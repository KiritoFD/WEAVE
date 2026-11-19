"""V23 P1 收官核心组合 (4090): ConvNeXt 中尺度特征损失 (C1) + 25% Anchor-Bridge (AB1) 联合训练。

前期结论:
- C1 (ConvNeXt 中尺度空间余弦损失) 推动 Pareto 外移 (+2.0pp Top-1 同时 DINO-C 微升);
- AB1 (25% Anchor-Bridge) 消除 train-inference 起点不匹配, 在 PRISM-1 锚点下抬高 DINO-C +0.007 & DINO-S +0.005 (Ukiyo-e +11~16pp);
- T1 (连续 layout) 拥有最高 DINO-C 基座 (0.60-0.62), Peak1500 (固定 layout) 拥有最高风格基座。

本实验验证两条联合主模型 (1200 updates, 同 seed):
1. Joint_T1_C1AB : 从 slf_T1.pt 出发 (保持 layout 连续训练以保内容 DINO-C≥0.60) + ConvNeXt 中尺度 (w_mid=0.25) + 25% Anchor-Bridge;
2. Joint_Pk_C1AB : 从 Peak1500 出发 (layout=1.0 固定) + 强化 ConvNeXt 中尺度 (w_mid=0.30) + 25% Anchor-Bridge。

每条联合模型在 PRISM-1 与 T4 锚点 × {hiS(g=0.45,wc=1.0), hiC(g=0.50,wc=1.5), bal(g=0.55,wc=1.6)} 评测，冲击 P3 联合目标: DINO-C>=0.60, DINO-S>=0.56, Top1>=60%。

用法 (4090):
  python -m exp.v23.p1_joint_c1_ab --pool ../pseudo_gt_24h/orig_mean_fp16
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
from .p1_anchor_bridge import build_or_load_train_prism1_anchors
from .p1_fast_anchor import compute_prism1_anchor_750
from .p1_midfreq_loss import _mid_spatial_cos_loss
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline
from src.utils import load_s10_decoder

PEAK_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"
T1_CKPT = ROOT / "runs/v22/tmatrix/slf_T1.pt"


def train_joint_arm(bundle: dict, core_bank: dict, z_a_tr: torch.Tensor, dec, cnx_feats: nn.Module,
                    arm: str, init_ckpt: Path, layout_train: bool, w_mid: float,
                    updates: int = 1200, device: str = "cuda") -> Path:
    out_p = OUT / "joint_c1ab" / f"slf_{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 {arm}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261800)
    tr = bundle["v18_cache"]["train"]
    z_all, sid_all = tr["z"], tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    model = SLFUNet(base_dim=144, u_dim=128).to(device)
    model.load_state_dict(torch.load(init_ckpt, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # 解冻条件与中后主干（保留前端结构稳定性）
    unfreeze = ("cond_mlp.", "dino_proj.", "mid1.", "mid_attn.", "mid2.",
                "up32.", "skip32.", "dec32.", "up64.", "skip64.", "dec64.", "out_norm.", "out_conv.")
    trainable = []
    for n, p in model.named_parameters():
        req = n.startswith(unfreeze)
        p.requires_grad_(req)
        if req:
            trainable.append(p)

    opt = torch.optim.AdamW(trainable, lr=5e-5, betas=(0.9, 0.99), weight_decay=1e-4)
    micro, accum = 10, 6  # 有效 batch = 60
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1e-5 + 0.5 * (5e-5 - 1e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        fm_v, mid_v = 0.0, 0.0
        for acc_i in range(accum):
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
            lscale = torch.where(dc, torch.zeros(B, device=device), 0.15 + 0.85 * torch.rand(B, device=device)) if layout_train else torch.ones(B, device=device)

            eps = torch.randn_like(z1)
            is_ab = (torch.rand(B, device=device) < 0.25)  # 25% Anchor-Bridge
            tau_fm = torch.rand(B, device=device)
            zt_fm = (1.0 - tau_fm).view(B, 1, 1, 1) * eps + tau_fm.view(B, 1, 1, 1) * z1
            v_fm = z1 - eps

            tau0 = 0.35 + 0.25 * torch.rand(B, device=device)
            z_init = (1.0 - tau0).view(B, 1, 1, 1) * eps + tau0.view(B, 1, 1, 1) * za
            tau_ab = tau0 + (1.0 - tau0) * torch.rand(B, device=device)
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

                mid_loss = z1.new_zeros(())
                if acc_i == 0:
                    valid = (tau > 0.35) & (~dc)
                    if valid.sum() >= 2:
                        sub = valid.nonzero(as_tuple=True)[0][:3]
                        z1_hat = (zt[sub] + (1.0 - tau[sub]).view(-1, 1, 1, 1) * pred_v[sub].float()).clamp(-5.0, 5.0)
                        mid_loss = _mid_spatial_cos_loss(cnx_feats, dec, z1_hat, z1[sub])

                loss = (fm_loss + w_mid * mid_loss * (accum if acc_i == 0 else 0.0)) / accum

            loss.backward()
            fm_v += float(fm_loss.item()) / accum
            if acc_i == 0:
                mid_v = float(mid_loss.item())

        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V23 {arm}] {step:4d}/{updates} fm={fm_v:.4f} mid={mid_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "arm": arm}, out_p)
    print(f"[V23 {arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_joint_arm(bundle: dict, bank: dict, ckpt_p: Path, arm: str, ctx: dict, device: str = "cuda") -> list[dict]:
    core = PrismCore(bundle, bank, ckpt_p, device=device)
    tasks, z_src_750 = ctx["tasks"], ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    res = []

    eval_points = (
        ("p1", ctx["z_prism1"], 0.45, 1.0, "hiS"),
        ("p1", ctx["z_prism1"], 0.50, 1.5, "hiC"),
        ("p1", ctx["z_prism1"], 0.58, 1.6, "bal"),
        ("t4", ctx["z_t4"],     0.50, 1.5, "hiC"),
    )
    for anc_tag, anc_t, g_val, wc_val, wp_tag in eval_points:
        name = f"{arm}_{anc_tag}_{wp_tag}"
        cfg_hash = config_hash({"name": name, "ckpt": file_sha(ckpt_p)[:12], "g": g_val, "wc": wc_val})
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
            meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_p),
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
    ap.add_argument("--updates", type=int, default=1200)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    z_a_tr = build_or_load_train_prism1_anchors(bundle, device=args.device)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)

    ck_t1 = train_joint_arm(bundle, bank, z_a_tr, dec, cnx, "Joint_T1_C1AB", T1_CKPT, True, 0.25, args.updates, args.device)
    ck_pk = train_joint_arm(bundle, bank, z_a_tr, dec, cnx, "Joint_Pk_C1AB", PEAK_CKPT, False, 0.30, args.updates, args.device)
    del cnx
    torch.cuda.empty_cache()

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    z_prism1 = compute_prism1_anchor_750(bundle, tasks, z_src_750, d16_750, device=args.device)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "z_t4": z_t4, "z_prism1": z_prism1,
           "y16_750": y16_750, "d16_750": d16_750, "eps_750": eps_750, "dec": dec}

    results = []
    results.extend(eval_joint_arm(bundle, bank, ck_t1, "Joint_T1_C1AB", ctx, device=args.device))
    results.extend(eval_joint_arm(bundle, bank, ck_pk, "Joint_Pk_C1AB", ctx, device=args.device))

    save_json(OUT / "p1_joint_c1_ab.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P1 联合突破 (C1 中尺度 + AB1 25% Anchor-Bridge) 汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:22s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
