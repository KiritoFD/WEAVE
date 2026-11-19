"""V23 方向校准 C (4090): 专用双分支 + 全批 ConvNeXt 中尺度约束 + 25% Anchor-Bridge (Dual_C1AB)。

为什么选专用双分支作为高 DINO-C 前沿主线:
1. F3_dual_fixed_orig (V22 Dual 修复双分支) 未经中尺度与 AB 训练即达 DINO-S=0.5522 @ DINO-C=0.6096,
   其 DINO-S 比同 DINO-C 区的四分支/T1 模型高出 +0.020~0.030!
2. 原因是训练期 drop_content=False 把 100% 模型容量专注于 (v_sc, v_0c) 两个内容保留分支,
   彻底不训练毁内容的 v_s0 / v_00 纯风格方向;
3. 在此之上叠加 100% 微批密度的 ConvNeXt 中尺度空间余弦损失 (w_mid=0.35) + 25% Anchor-Bridge
   (tau0 in [0.38, 0.68]), 并扫描 ws_fig in {4.5, 5.5, 6.5} 与 gate in {original, fixed 0.50},
   冲击 DINO-C >= 0.612 & DINO-S >= 0.562 & Top-1 >= 58% 全面支配 StyleGallery!

用法 (4090):
  python -m exp.v23.p1_dual_c1ab_4090 --pool ../pseudo_gt_24h/orig_mean_fp16
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

from . import OUT, EVAL_SEED, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from .p1_anchor_bridge import build_or_load_train_prism1_anchors
from .p1_fast_anchor import compute_prism1_anchor_750
from .p1_high_dinoc_frontier import run_point
from .p1_midfreq_loss import _mid_spatial_cos_loss
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline
from src.utils import load_s10_decoder

BASE_CKPT = ROOT / "runs/v20/slf_flow/slf_final.pt"


def train_dual_c1ab(bundle: dict, core_bank: dict, z_a_tr: torch.Tensor, dec, cnx_feats: nn.Module,
                    w_mid: float = 0.35, updates: int = 1500, device: str = "cuda") -> Path:
    out_p = OUT / "dual_c1ab" / "slf_dual_c1ab.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 DualC1AB] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20262000)
    tr = bundle["v18_cache"]["train"]
    z_all, sid_all = tr["z"], tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    model = SLFUNet(base_dim=144, u_dim=128).to(device)
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    micro, accum = 10, 6  # 有效 batch = 60, 每微批最多4张全量进 S10+ConvNeXt
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1.5e-5 + 0.5 * (1.0e-4 - 1.5e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        fm_v, mid_v, n_mid = 0.0, 0.0, 0
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

            # 专用双分支协议: 内容永不 dropout, layout_scale=1.0 固定, 只 drop_style 15% 供 v_0c 使用
            ds = torch.rand(B, device=device) < 0.15
            dc = torch.zeros(B, dtype=torch.bool, device=device)
            lscale = torch.ones(B, device=device)

            eps = torch.randn_like(z1)
            is_ab = (torch.rand(B, device=device) < 0.25)
            tau_fm = torch.rand(B, device=device)
            zt_fm = (1.0 - tau_fm).view(B, 1, 1, 1) * eps + tau_fm.view(B, 1, 1, 1) * z1
            v_fm = z1 - eps

            tau0 = 0.38 + 0.30 * torch.rand(B, device=device)
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
                valid = (tau > 0.30)
                if valid.sum() >= 2:
                    sub = valid.nonzero(as_tuple=True)[0][:4]
                    z1_hat = (zt[sub] + (1.0 - tau[sub]).view(-1, 1, 1, 1) * pred_v[sub].float()).clamp(-5.0, 5.0)
                    mid_loss = _mid_spatial_cos_loss(cnx_feats, dec, z1_hat, z1[sub])
                    mid_v += float(mid_loss.item())
                    n_mid += 1

                loss = (fm_loss + w_mid * mid_loss) / accum

            loss.backward()
            fm_v += float(fm_loss.item()) / accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V23 DualC1AB] {step:4d}/{updates} fm={fm_v:.4f} mid={mid_v/max(1,n_mid):.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "w_mid": w_mid}, out_p)
    print(f"[V23 DualC1AB] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1500)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    z_a_tr = build_or_load_train_prism1_anchors(bundle, device=args.device)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)

    ckpt_p = train_dual_c1ab(bundle, bank, z_a_tr, dec, cnx, w_mid=0.35, updates=args.updates, device=args.device)
    del cnx
    torch.cuda.empty_cache()

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    z_prism1 = compute_prism1_anchor_750(bundle, tasks, z_src_750, d16_750, device=args.device)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750, "d16_750": d16_750,
           "eps_750": eps_750, "dec": dec}

    core = PrismCore(bundle, bank, ckpt_p, device=args.device)
    # 扫描修复后真双分支 cs_j 的 ws_fig 强度与门控 (T4 与 PRISM-1 双锚点)
    points = [
        ("DualC1AB_orig_ws42_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="cs_j"), dict(D5_SOLVER, ws_fig=4.2)),
        ("DualC1AB_orig_ws52_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="cs_j"), dict(D5_SOLVER, ws_fig=5.2)),
        ("DualC1AB_orig_ws62_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="cs_j"), dict(D5_SOLVER, ws_fig=6.2)),
        ("DualC1AB_orig_ws52_p1", z_prism1, GateConfig(mode="original"), CFGConfig(branches="cs_j"), dict(D5_SOLVER, ws_fig=5.2)),
        ("DualC1AB_g50_ws52_t4",  z_t4,     GateConfig(mode="fixed", fixed_g=0.50), CFGConfig(branches="cs_j"), dict(D5_SOLVER, ws_fig=5.2)),
        ("DualC1AB_g50_ws52_p1",  z_prism1, GateConfig(mode="fixed", fixed_g=0.50), CFGConfig(branches="cs_j"), dict(D5_SOLVER, ws_fig=5.2)),
    ]

    results = []
    for arm_name, anc, g, cfg, solv in points:
        results.append(run_point(core, ckpt_p, ctx, arm_name, anc, g, cfg, solv, device=args.device))

    save_json(OUT / "p1_dual_c1ab.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 Dual_C1AB 汇总 (红线 DINO-C >= 0.610, 对标 SG: 52.50% / 0.5604 / 0.6113) ===", flush=True)
    for r in results:
        flag = " [DOMINATES_SG!]" if (r["dino_c"] >= 0.610 and r["dino_s"] >= 0.560) else (" [D-C>=0.61]" if r["dino_c"] >= 0.610 else "")
        print(f"{r['arm']:22s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}{flag}", flush=True)


if __name__ == "__main__":
    main()
