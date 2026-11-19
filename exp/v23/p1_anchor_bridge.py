"""V23 P1 第五组 (4090): 内容一致的 Anchor-Bridge 混合训练（消除 train-inference 起点分布差异）。

零测试泄露的训练内部 Anchor-Bridge 配对构造:
1. 对训练集 4400 张真实画作 z1，构造去风格伪源图 z_ps（2D 高斯模糊削弱局部笔触 + 轻度随机跨类潜色度扰动）；
2. 用冻结 PRISM-1 学生模型对 z_ps 朝原画所属画派 sid 推理得到锚点 z_a（与推理期 A1_pk_prism1 完全同分布）；
3. 编辑目标 z* = z1（真实艺术品，内容结构 y16/d16 精确对齐、零伪影）。

严格遵循 V23 §1.6 数学定义:
  z_init = (1 - tau0)*eps + tau0*z_a
  z_t    = z_init + ((t - tau0) / (1 - tau0)) * (z* - z_init)
  v_tgt  = (z* - z_init) / (1 - tau0)

三组公平对照（从 Peak1500 热启动，900 updates，同 seed）:
- AB0_fm100 : 100% 原始 FM
- AB1_mix25 : 75% 原始 FM + 25% Anchor-Bridge
- AB2_mix50 : 50% 原始 FM + 50% Anchor-Bridge

用法 (4090):
  python -m exp.v23.p1_anchor_bridge --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, EVAL_SEED, ROOT, RawWriterV23, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v23, sanitize_cuda_env, save_json, seed_all, utc, validate_raw_meta
from .p1_fast_anchor import compute_prism1_anchor_750
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift, build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from src.utils import load_s10_decoder

PEAK_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"

AB_CONFIGS = {
    "AB0_fm100": 0.0,
    "AB1_mix25": 0.25,
    "AB2_mix50": 0.50,
}


@torch.no_grad()
def build_or_load_train_prism1_anchors(bundle: dict, device: str = "cuda") -> torch.Tensor:
    """在训练集 4400 张图上构造零泄露的 PRISM-1 内部锚点 z_a（缓存到 runs/v23/cache/train_prism1_za.pt）。"""
    cache_p = OUT / "cache/train_prism1_za.pt"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    if cache_p.exists():
        print(f"[V23 AB] Reusing {cache_p}", flush=True)
        return torch.load(cache_p, map_location="cpu", weights_only=False)["z_a"]

    from exp.v19.routed_student import SpatiallyRoutedChordNet

    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all = tr["z"].float()
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"].float()
    c64_all = tr["c_64"].float() if "c_64" in tr else None

    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    ck = torch.load(ROOT / "runs/v19/students/s2_prism1_v2.pt", map_location="cpu", weights_only=False)
    stu.load_state_dict(ck["model"])

    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_all = fiber_enc.encode(fnd["tr_fib"].to(device).float())
    bary = fiber_enc.class_barycenters()

    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]
    P_C = priors.get("P_C")

    # 若 tr 无 c_64 则直接由 luma_proj/P_C 或 y_64 计算近似 c_64
    if c64_all is None:
        c64_all = tr["y_64"].float()

    g = torch.Generator().manual_seed(20261350)
    z_a = torch.zeros_like(z_all, dtype=torch.float16)
    N = len(z_all)
    for st in range(0, N, 64):
        sl = slice(st, min(st + 64, N))
        zs = z_all[sl].to(device)
        sid_b = sid_all[sl].to(device)
        # 模拟“来自随机异风格源图的锚点输入”：先往随机其他画派施加色度漂移+轻度平滑去高频笔触，再推回 sid_b
        rand_sid = (sid_b + torch.randint(1, 5, (len(sid_b),), generator=g).to(device)) % 5
        z_ps = F.avg_pool2d(F.pad(zs, (1, 1, 1, 1), mode="replicate"), 3, 1)
        z_ps = apply_latent_chroma_shift(z_ps, rand_sid, priors, alpha=0.25)
        z_anc = apply_latent_chroma_shift(z_ps, sid_b, priors, alpha=0.20)
        u_ps = u_all[torch.randint(0, N, (len(sid_b),), generator=g).to(device)]
        chord = bary[sid_b] - u_ps
        z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
        c64_b = c64_all[sl].to(device)
        if c64_b.shape[1] == 1 and "c_64" not in tr:
            # fallback: 构造2通道近似结构张量 [0, 0] 若缺失
            c64_b = torch.cat([c64_b, torch.zeros_like(c64_b)], dim=1)
        delta = stu(z_anc, z_luma, c64_b, d16_all[sl].to(device), chord)
        z_a[sl] = (z_anc + delta).detach().cpu().half()

    del stu, fiber_enc
    torch.cuda.empty_cache()
    torch.save({"z_a": z_a}, cache_p)
    print(f"[V23 AB] Saved train anchors -> {cache_p} ({N} samples)", flush=True)
    return z_a


def train_ab_arm(bundle: dict, core_bank: dict, z_a_tr: torch.Tensor, arm: str,
                 bridge_prob: float, updates: int = 900, device: str = "cuda") -> Path:
    out_p = OUT / "anchor_bridge" / f"slf_{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 {arm}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261360)
    tr = bundle["v18_cache"]["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()
    z_all, sid_all = tr["z"], tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    model = SLFUNet(base_dim=144, u_dim=128).to(device)
    model.load_state_dict(torch.load(PEAK_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=8e-5, betas=(0.9, 0.99), weight_decay=1e-4)
    micro = 120  # 4090 24GB 可单批 120
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1.5e-5 + 0.5 * (8e-5 - 1.5e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        per_cls = micro // 5
        cur_pools = core_cls_idx if (step % 5 != 0) else all_cls_idx
        idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in cur_pools], dim=0)
        B = len(idx)

        z1 = z_all[idx].to(device, non_blocking=True).float()
        za = z_a_tr[idx].to(device, non_blocking=True).float()
        sid = sid_all[idx].to(device, non_blocking=True)
        d16 = d16_all[idx].to(device, non_blocking=True).float()
        y16 = y16_all[idx].to(device, non_blocking=True).float()
        uv = u_exact_all[idx].to(device, non_blocking=True).float()

        d16_aug = d16 + 0.08 * torch.randn_like(d16)
        d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
        y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

        u_r = torch.rand(B, device=device)
        ds = u_r < 0.12
        dc = (u_r >= 0.12) & (u_r < 0.42)
        both = u_r >= 0.95
        ds, dc = ds | both, dc | both
        lscale = torch.ones(B, device=device)

        eps = torch.randn_like(z1)
        is_ab = (torch.rand(B, device=device) < bridge_prob)

        # 标准直线 FM 路径
        tau_fm = torch.rand(B, device=device)
        zt_fm = (1.0 - tau_fm).view(B, 1, 1, 1) * eps + tau_fm.view(B, 1, 1, 1) * z1
        v_fm = z1 - eps

        # Anchor-Bridge 路径（V23 §1.6 精确定义）
        tau0 = 0.35 + 0.25 * torch.rand(B, device=device)  # [0.35, 0.60]
        z_init = (1.0 - tau0).view(B, 1, 1, 1) * eps + tau0.view(B, 1, 1, 1) * za
        tau_ab = tau0 + (1.0 - tau0) * torch.rand(B, device=device)  # [tau0, 1.0]
        frac = ((tau_ab - tau0) / (1.0 - tau0).clamp_min(1e-4)).view(B, 1, 1, 1)
        zt_ab = z_init + frac * (z1 - z_init)
        v_ab = (z1 - z_init) / (1.0 - tau0).clamp_min(1e-4).view(B, 1, 1, 1)

        m4 = is_ab.view(B, 1, 1, 1)
        tau = torch.where(is_ab, tau_ab, tau_fm)
        zt = torch.where(m4, zt_ab, zt_fm)
        target_v = torch.where(m4, v_ab, v_fm)

        with autocast(device):
            pred_v = model(zt, y16_aug, d16_aug, sid, uv, tau,
                           drop_style=ds, drop_content=dc, layout_scale=lscale)
            fm_loss = F.mse_loss(pred_v.float(), target_v)
            z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
            y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
            w_tau = ((tau > 0.25) & (~dc)).float()
            st_loss = (((y16_hat - y16).abs().mean(dim=(1, 2, 3))) * w_tau).sum() / w_tau.sum().clamp_min(1.0)
            loss = fm_loss + 0.10 * st_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V23 {arm}] {step:4d}/{updates} loss={loss.item():.4f} fm={fm_loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "arm": arm, "bridge_prob": bridge_prob}, out_p)
    print(f"[V23 {arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_ab_arm(bundle: dict, bank: dict, ckpt_path: Path, arm: str, ctx: dict, device: str = "cuda") -> list[dict]:
    core = PrismCore(bundle, bank, ckpt_path, device=device)
    tasks, z_src_750 = ctx["tasks"], ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    res = []
    for anc_tag, anc_t in (("p1", ctx["z_prism1"]), ("t4", ctx["z_t4"])):
        for g_val, wc_val, wp_tag in ((0.45, 1.0, "hiS"), (0.50, 1.5, "hiC")):
            name = f"{arm}_{anc_tag}_{wp_tag}"
            cfg_hash = config_hash({"name": name, "ckpt": file_sha(ckpt_path)[:12], "g": g_val, "wc": wc_val})
            if (OUT / "raw" / name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / name / "meta.json")):
                s = read_json(OUT / f"measured/{name}/summary_v23.json")
            else:
                gate = GateConfig(mode="fixed", fixed_g=g_val)
                cfg = CFGConfig(branches="full", rescale_phi=0.48)
                solver = dict(D5_SOLVER, wc_fig=wc_val, wc_abs=wc_val)
                wr = RawWriterV23(name, dec=ctx["dec"], device=device)
                t0 = time.time()
                for st in range(0, 750, 25):
                    sl = slice(st, min(st + 25, 750))
                    z_gen, _ = core.integrate_flow(
                        ctx["eps_750"][sl], anc_t[sl].to(device), ctx["y16_750"][sl], ctx["d16_750"][sl],
                        tid_all[sl].to(device), gate=gate, cfg=cfg, solver=solver,
                    )
                    z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), z_gen)
                    wr.write_batch(z_out, tasks[sl])
                meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_path),
                                   tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                                   seed=EVAL_SEED, g=g_val, wc=wc_val, anc=anc_tag, wall_sec=round(time.time() - t0, 1))
                assert validate_raw_meta(meta)
                s = run_measure_v23(name, device=device)
                export_raw_pngs(OUT / f"raw/{name}", OUT / f"png_750/{name}", max_workers=8)
            c = s["cross600"]
            res.append({"arm": name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                        "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                        "lpips": round(c["lpips"], 5)})
    del core
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=900)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    z_a_tr = build_or_load_train_prism1_anchors(bundle, device=args.device)

    ckpts = {}
    for arm, bp in AB_CONFIGS.items():
        ckpts[arm] = train_ab_arm(bundle, bank, z_a_tr, arm, bp, updates=args.updates, device=args.device)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    z_prism1 = compute_prism1_anchor_750(bundle, tasks, z_src_750, d16_750, device=args.device)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(args.device)
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "z_t4": z_t4, "z_prism1": z_prism1,
           "y16_750": y16_750, "d16_750": d16_750, "eps_750": eps_750, "dec": dec}

    results = []
    for arm in AB_CONFIGS:
        results.extend(eval_ab_arm(bundle, bank, ckpts[arm], arm, ctx, device=args.device))

    save_json(OUT / "p1_anchor_bridge.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P1 第五组（Anchor-Bridge 训练）汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:22s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
