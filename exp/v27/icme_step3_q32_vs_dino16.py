r"""ICME 收束第 3 步 (RTX 3060): Q32 vs dino_16 闭集单变量裁决.

目的 (用户 2026-11-23 收束清单 §三.3):
  在选定的闭集主力 Flow 配方 (V25_Tr_T1 配方: slf_final 同构初始化/从头或同起点匹配训练,
  drop_c=0.30, drop_s=0.12, layout=1.0, core 80%, struct_w=0.10, u_exact_all+sid) 上,
  仅将 16×16 瓶颈语义条件 `dino_16 (384ch)` 替换为 `Q32 (32ch)` 跑一次完全同条件匹配训练
  (1800 updates, micro=24 × accum=5 = 120 有效 batch, 适配 RTX 3060 12GB 显存 ≤ 6GB).

  两条对照臂 (严格同 seed / 同采样顺序 / 同优化器 / 同 solver 评测):
    1. DINO16_ctrl: SLFUNet (in_ch=384 → 160 → 80ch)
    2. Q32_test:    SLFUNet (in_ch=32  → 160 → 80ch, 结构其余逐参数一致)
  (从随机初始化或各自适配层同起点训练 2000 updates — 公平单变量对比, 不引入预训练不对称!)

  统一评测 (6-step cs_j D5_CFG + T4 锚, 750 全量 + src.eval.measure 隔离进程):
    - 若 Q32 ≥ DINO16 (Top-1 / DINO-C 持平或反超) → 切主干为 Q32, 理论/Fig.3/代码 100% 统一;
    - 若 DINO16 明显胜出 → 保留 dino_16 为闭集主干, 论文如实分层陈述 + 消融表交代.

用法 (3060):  python -m exp.v27.icme_step3_q32_vs_dino16 --updates 2000
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v27 import OUT, ROOT, RawWriterV27, run_measure_v27
from exp.v27.icme_step2_closedset_accel import D5_CFG, integrate_segmented
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast as autocast, read_json, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc


def make_bottleneck_net(in_ch: int) -> SLFUNet:
    """SLFUNet 仅修改 dino_proj 首层输入通道 (384 for dino_16 vs 32 for Q32), 其余完全同构."""
    m = SLFUNet(base_dim=144, u_dim=128)
    if in_ch != 384:
        m.dino_proj[0] = nn.Conv2d(in_ch, 160, 1)
    return m


def train_closedset_arm(arm: str, use_q32: bool, bundle: dict, c21: dict, bank: dict,
                        updates: int = 2000, device: str = "cuda", seed: int = 20267400,
                        smoke: bool = False) -> str:
    ckpt_p = OUT / f"icme_step3_{arm}.pt"
    if ckpt_p.exists() and not smoke:
        print(f"[Step3 {arm}] Reusing {ckpt_p}", flush=True)
        return str(ckpt_p)

    seed_all(seed)
    micro, accum = tier_micro_accum(120, default_micro_24gb=60, max_micro_12gb=24, device=device)
    tr = bundle["v18_cache"]["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

    z_all = tr["z"]
    sid_all = tr["sid"].long()
    feat_all = c21["train"]["q32"] if use_q32 else tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_ex_all = bank["sample_sub_u"]  # (4400, 128)
    from exp.v19.fiber_duality import StyleFiberEncoder
    fnd = bundle["fnd"]
    enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_exact = enc.encode(fnd["tr_fib"].to(device).float()).cpu().half()
    del enc

    all_cls = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls = [ci.long() for ci in bank["core_idx_per_style"]]

    model = make_bottleneck_net(32 if use_q32 else 384).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-4, betas=(0.9, 0.99), weight_decay=1e-4)

    t0 = time.time()
    print(f"[Step3 {arm}] start (use_q32={use_q32}, micro={micro}×accum={accum}, updates={updates})", flush=True)
    for step in range(1, updates + 1):
        if step <= 200:
            lr = 2.2e-4 * (step / 200.0)
        else:
            prog = (step - 200) / max(1, updates - 200)
            lr = 2.0e-5 + 0.5 * (2.2e-4 - 2.0e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        fm_v, st_v = 0.0, 0.0

        for _ in range(accum):
            per_cls = max(1, micro // 5)
            pools = core_cls if (step % 5 != 0) else all_cls
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in pools], dim=0)
            B = len(idx)

            z1 = z_all[idx].to(device, non_blocking=True).float()
            sid = sid_all[idx].to(device, non_blocking=True)
            f16 = feat_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            uv = torch.where(torch.rand(B, 1, device=device) < 0.75,
                             u_exact[idx].to(device).float(), u_ex_all[idx].to(device).float())

            f16_aug = f16 + (0.06 if use_q32 else 0.08) * torch.randn_like(f16)
            f16_aug = f16_aug - f16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device))
                       + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            u_r = torch.rand(B, device=device)
            ds = u_r < 0.12
            dc = (u_r >= 0.12) & (u_r < 0.42)  # 30% drop_c (V25_Tr_T1)
            both = u_r >= 0.95
            ds, dc = ds | both, dc | both
            lscale = torch.ones(B, device=device)

            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = model(zt, y16_aug, f16_aug, sid, uv, tau,
                               drop_style=ds, drop_content=dc, layout_scale=lscale)
                fm_loss = F.mse_loss(pred_v.float(), target_v)
                z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
                y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                w_tau = (tau > 0.25).float()
                st_loss = ((y16_hat - y16).abs().mean(dim=(1, 2, 3)) * w_tau).sum() / w_tau.sum().clamp_min(1.0)
                loss = (fm_loss + 0.10 * st_loss) / accum
            loss.backward()
            fm_v += float(fm_loss.item()) / accum
            st_v += float(st_loss.item()) / accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[Step3 {arm}] {step:4d}/{updates} fm={fm_v:.4f} st={st_v:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if smoke and step >= 50:
            break

    if not smoke:
        torch.save({"ema": ema.state_dict(), "updates": updates, "use_q32": use_q32}, ckpt_p)
        print(f"[Step3 {arm}] Saved {ckpt_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return str(ckpt_p)


@torch.no_grad()
def eval_step3_arm(arm: str, ckpt_p: str, use_q32: bool, bundle: dict, c21: dict, bank: dict,
                   tasks: list[dict], z_src_750: torch.Tensor, z_t4: torch.Tensor,
                   device: str = "cuda") -> dict:
    m_dir = OUT / "measured" / f"ICME_S3_{arm}"
    if (m_dir / "summary_v27.json").exists():
        s = read_json(m_dir / "summary_v27.json")
    else:
        y16_750, d16_750 = get_test750_y16_and_d16(bundle)
        feat_750 = c21["test750"]["q32"] if use_q32 else d16_750
        eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
        dec = load_s10_decoder(device)
        sid_all = torch.tensor([t["source_style_id"] for t in tasks])
        tid_all = torch.tensor([t["target_style_id"] for t in tasks])
        same_all = sid_all == tid_all

        model = make_bottleneck_net(32 if use_q32 else 384).to(device).eval()
        model.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=False)["ema"], strict=True)
        # Note: integrate_segmented uses d16 for scene matching AND passes f16 into model;
        # we match scene submodes using d16_750 (or q32), and pass feat_750 cleanly into model.
        wr = RawWriterV27(f"ICME_S3_{arm}", dec=dec, device=device)
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            zg = _integrate_with_feat(model, z_t4[sl].to(device).float(), eps_750[sl].to(device).float(),
                                      y16_750[sl].to(device).float(), d16_750[sl].to(device).float(),
                                      feat_750[sl].to(device).float(), tid_all[sl].to(device),
                                      bank, steps=6, cfg_kw=D5_CFG)
            wr.write_batch(torch.where(same_all[sl].view(-1, 1, 1, 1).to(device),
                                z_src_750[sl].to(device).float(), zg), tasks[sl])
        wr.finalize(arm=arm, use_q32=use_q32)
        del model
        torch.cuda.empty_cache()
        s = run_measure_v27(f"ICME_S3_{arm}", device=device)
    c = s["cross600"]
    return {
        "csd_top1": round(c["csd_top1"], 4), "dino_s": round(c["dino_s"], 4),
        "dino_c": round(c["dino_c"], 4), "style_affinity": round(c["style_affinity"], 4),
        "clip_s": round(c["clip_s"], 4), "lpips": round(c["lpips"], 4),
        "same150_top1": round(s["same150"]["csd_top1"], 4),
        "per_target": {k: round(v["csd_top1"], 4) for k, v in s.get("per_target_cross120", {}).items()},
    }


@torch.no_grad()
def _integrate_with_feat(model, z_t4, eps, y16, d16_match, f16_in, tid, bank, *,
                         steps: int, cfg_kw: dict) -> torch.Tensor:
    from exp.v20.submode_patchswd import match_scene_submode
    dev = z_t4.device
    B = len(z_t4)
    _, u_core = match_scene_submode(d16_match, tid, bank)
    sc_src = F.normalize(F.avg_pool2d(d16_match.float(), 4).flatten(1), dim=-1)
    best_c = torch.einsum("bd,bkd->bk", sc_src, bank["sub_scene"].to(dev).float()[tid]).argmax(dim=1)
    ex_dsd = bank["sub_dsd"].to(dev).float()[tid, best_c]
    g_fig = torch.sigmoid((ex_dsd - float(bank["mid_dsd"])) * 550.0)
    gf4 = g_fig.view(B, 1, 1, 1)
    lscale = (cfg_kw["lscale_abs"] + (1.0 - cfg_kw["lscale_abs"]) * g_fig).clamp(0.15, 1.0)
    c8_w = (1.0 - g_fig).clamp(0.0, 0.85).view(B, 1, 1, 1)
    f8 = F.avg_pool2d(f16_in, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    f8 = f8 - f8.mean(dim=(2, 3), keepdim=True)
    y8 = F.avg_pool2d(y16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    f_in = (1.0 - c8_w) * f16_in + c8_w * f8
    y_in = (1.0 - c8_w) * y16 + c8_w * y8

    tau_v = (cfg_kw["tau_min"] + (cfg_kw["tau_hi"] - cfg_kw["tau_min"]) * g_fig).clamp(0.0, 0.78)
    z = (1.0 - tau_v.view(B, 1, 1, 1)) * eps + tau_v.view(B, 1, 1, 1) * z_t4
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    dt = (1.0 - tau_v).view(B, 1, 1, 1) / float(steps)

    def _vel(zz, prog):
        tv = tau_v + prog * (1.0 - tau_v)
        with autocast(str(dev)):
            v_sc, v_0c = model(
                zz.repeat(2, 1, 1, 1), y_in.repeat(2, 1, 1, 1), f_in.repeat(2, 1, 1, 1),
                tid.repeat(2), u_core.repeat(2, 1), tv.repeat(2),
                drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                layout_scale=lscale.repeat(2),
            ).float().chunk(2, 0)
        gl = min(1.0, max(0.0, (prog - 0.36) / 0.32))
        ws_j = gf4 * (cfg_kw["ws_fig"] - 1.1 * gl) + (1.0 - gf4) * 1.4
        return v_0c + ws_j * (v_sc - v_0c)

    for i in range(steps):
        v1 = _vel(z, i / float(steps))
        zm = z + 0.5 * dt * v1
        v2 = _vel(zm, (i + 0.5) / float(steps))
        z = z + dt * v2
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)
    pack = build_or_load_v21_cache(args.pool, device=args.device)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    if args.smoke:
        train_closedset_arm("SMOKE_Q32", True, bundle, c21, bank, updates=50, device=args.device, smoke=True)
        print("[Step3] SMOKE_OK", flush=True)
        return
    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    res = {"task": "ICME Step3 (RTX 3060): Q32 vs dino_16 closed-set single-variable ablation",
           "updates": args.updates, "updated_utc": utc(), "arms": {}}
    for arm, uq in (("DINO16_ctrl", False), ("Q32_test", True)):
        ck = train_closedset_arm(arm, uq, bundle, c21, bank, updates=args.updates, device=args.device)
        res["arms"][arm] = eval_step3_arm(arm, ck, uq, bundle, c21, bank, tasks, z_src_750, z_t4, device=args.device)
        save_json(OUT / "icme_step3_q32_vs_dino16.json", res)
        print(f"[Step3 EVAL] {arm:14s} | {res['arms'][arm]}", flush=True)
    print(f"[Step3] DONE → {OUT / 'icme_step3_q32_vs_dino16.json'}", flush=True)


if __name__ == "__main__":
    main()
