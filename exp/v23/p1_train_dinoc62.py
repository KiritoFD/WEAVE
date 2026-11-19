"""V23 方向校准 B (4090): 专门面向 DINO-C >= 0.615 高内容前沿的重构训练 (SLF-Content62)。

针对前期教训的系统修正:
1. 不再使用偏低内容的 Peak1500 底座, 改从高内容底座 (Joint_T1_C1AB 或 slf_T1) 热启动;
2. 解决此前 ConvNeXt 中尺度损失梯度密度过稀问题 (此前仅 3/60=5% 样本算 mid_loss):
   本实验在 micro=10, accum=6 下对每个微批中所有 tau > 0.30 & ~drop_content 样本全量计算 ConvNeXt 中尺度空间余弦损失 (密度提升 ~15×, w_mid=0.40);
3. 强化 Anchor-Bridge 混合 (25% 比例) 且覆盖高内容推理起止区间 tau0 in [0.40, 0.70];
4. 保留 layout 连续随机训练 (保证强内容响应) + 提升风格条件强度下限，在 DINO-C >= 0.610 红线上角逐 DINO-S >= 0.565 & Top-1 >= 60%!

用法 (4090):
  python -m exp.v23.p1_train_dinoc62 --pool ../pseudo_gt_24h/orig_mean_fp16
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

T1_CKPT = ROOT / "runs/v22/tmatrix/slf_T1.pt"


def train_dinoc62(bundle: dict, core_bank: dict, z_a_tr: torch.Tensor, dec, cnx_feats: nn.Module,
                  w_mid: float = 0.40, updates: int = 1200, device: str = "cuda") -> Path:
    out_p = OUT / "dinoc62" / "slf_dinoc62.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 D62] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261900)
    tr = bundle["v18_cache"]["train"]
    z_all, sid_all = tr["z"], tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    model = SLFUNet(base_dim=144, u_dim=128).to(device)
    model.load_state_dict(torch.load(T1_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # 全网络轻量微调 (偏向中后解码与条件注入)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, betas=(0.9, 0.99), weight_decay=1e-4)
    micro, accum = 10, 6  # 每微批最多 ~7 张进 S10+ConvNeXt 全量算 mid_loss
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 8e-6 + 0.5 * (4e-5 - 8e-6) * (1.0 + math.cos(math.pi * prog))
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

            u_r = torch.rand(B, device=device)
            ds = u_r < 0.14  # 略增 drop_style 让 v_0c 分支更准
            dc = (u_r >= 0.14) & (u_r < 0.40)
            both = u_r >= 0.95
            ds, dc = ds | both, dc | both
            # 保持 layout 连续训练 (强内容保护分布)
            lscale = torch.where(dc, torch.zeros(B, device=device), 0.20 + 0.80 * torch.rand(B, device=device))

            eps = torch.randn_like(z1)
            is_ab = (torch.rand(B, device=device) < 0.25)
            tau_fm = torch.rand(B, device=device)
            zt_fm = (1.0 - tau_fm).view(B, 1, 1, 1) * eps + tau_fm.view(B, 1, 1, 1) * z1
            v_fm = z1 - eps

            # 高保真 Anchor-Bridge 起点覆盖 [0.40, 0.70]
            tau0 = 0.40 + 0.30 * torch.rand(B, device=device)
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
                valid = (tau > 0.30) & (~dc)
                if valid.sum() >= 2:
                    # 4090 上 micro<=10 时每微批全量过 ConvNeXt (限制 <=4 张以稳控制在 16GB 内, 100% 微批参与!)
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

        if step == 1 or step % 200 == 0 or step == updates:
            print(f"[V23 D62] {step:4d}/{updates} fm={fm_v:.4f} mid={mid_v/max(1,n_mid):.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "w_mid": w_mid}, out_p)
    print(f"[V23 D62] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


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

    ckpt_p = train_dinoc62(bundle, bank, z_a_tr, dec, cnx, w_mid=0.40, updates=args.updates, device=args.device)
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
    s_csj65 = dict(D5_SOLVER, tau_min=0.34, tau_hi=0.70, ws_fig=6.5)
    s_lowwp = dict(D5_SOLVER, tau_min=0.34, tau_hi=0.70, wc_fig=1.80, wc_abs=1.40,
                   ws_fig=5.0, wsp_fig_late=0.9, wsp_abs=1.8, lscale_abs=0.42)
    s_bal   = dict(D5_SOLVER, wc_fig=1.50, wc_abs=1.50)

    points = [
        ("D62_csj65_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="cs_j"), s_csj65),
        ("D62_lowwp_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="full", rescale_phi=0.48), s_lowwp),
        ("D62_lowwp_p1", z_prism1, GateConfig(mode="original"), CFGConfig(branches="full", rescale_phi=0.48), s_lowwp),
        ("D62_g52_t4",   z_t4,     GateConfig(mode="fixed", fixed_g=0.52), CFGConfig(branches="full", rescale_phi=0.48), s_bal),
        ("D62_g52_p1",   z_prism1, GateConfig(mode="fixed", fixed_g=0.52), CFGConfig(branches="full", rescale_phi=0.48), s_bal),
    ]

    results = []
    for arm_name, anc, g, cfg, solv in points:
        results.append(run_point(core, ckpt_p, ctx, arm_name, anc, g, cfg, solv, device=args.device))

    save_json(OUT / "p1_dinoc62.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 SLF-Content62 汇总 (红线 DINO-C >= 0.610) ===", flush=True)
    for r in results:
        flag = " [DOMINATES_SG!]" if (r["dino_c"] >= 0.610 and r["dino_s"] >= 0.560) else (" [D-C>=0.61]" if r["dino_c"] >= 0.610 else "")
        print(f"{r['arm']:18s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}{flag}", flush=True)


if __name__ == "__main__":
    main()
