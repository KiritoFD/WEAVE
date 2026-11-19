"""V23 方向校准终极冲刺 (4090): 感知局部笔触 Patch-SWD 微调专用双分支模型 (Dual_SWD)。

原理:
FM 纯 L2/MSE 损失在 DINO-C >= 0.61 区间会对局部高频笔触隐式平均，使 DINO-S 停在 ~0.553。
本实验从 Dual_AB25 (50.83% / D-S 0.5513 / D-C 0.6068) 热启动, 经可微 S10 施加独立于裁判的
MultiScalePatchSWDExtractor (ConvNeXt-Tiny + RGB 3x3 局部笔触 Sliced-Wasserstein 分布损失, w_swd=0.12),
对比两组 900-update 微调:
- DualSWD_pure : w_swd=0.12, w_mid=0.00 (纯还原局部锐利笔触分布)
- DualSWD_both : w_swd=0.12, w_mid=0.15 (局部笔触分布 + 中尺度空间结构对齐)

评测在 orig gate (ws_fig in {4.2, 5.2}) 与 高内容起点 (tau_min=0.32, tau_hi=0.68, ws=5.2) 进行，
冲击 DINO-C >= 0.612 & DINO-S >= 0.562 全面支配 StyleGallery!

用法 (4090):
  python -m exp.v23.p1_dual_swd_train --pool ../pseudo_gt_24h/orig_mean_fp16
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
from .p1_dual_frontier_sprint import eval_decoupled_dual
from .p1_midfreq_loss import _mid_spatial_cos_loss
from exp.v22.core import PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, build_or_load_core_knn_bank, load_convnext_tiny_offline
from src.utils import decode_s10, load_s10_decoder

INIT_CKPT = OUT / "dual_sprint/slf_Dual_AB25.pt"


def train_dual_swd_arm(bundle: dict, core_bank: dict, z_a_tr: torch.Tensor, dec,
                       swd_ext: MultiScalePatchSWDExtractor, cnx_feats: nn.Module,
                       arm: str, w_swd: float, w_mid: float, updates: int = 900,
                       device: str = "cuda") -> Path:
    out_p = OUT / "dual_swd" / f"slf_{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 {arm}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20262400)
    tr = bundle["v18_cache"]["train"]
    z_all, sid_all = tr["z"], tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    model = SLFUNet(base_dim=144, u_dim=128).to(device)
    init_p = INIT_CKPT if INIT_CKPT.exists() else (ROOT / "runs/v20/slf_flow/slf_final.pt")
    model.load_state_dict(torch.load(init_p, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # 解冻条件层 + 中后层解码块以精细化笔触输出
    unfreeze = ("cond_mlp.", "dino_proj.", "mid1.", "mid_attn.", "mid2.",
                "up32.", "skip32.", "dec32.", "up64.", "skip64.", "dec64.", "out_norm.", "out_conv.")
    trainable = []
    for n, p in model.named_parameters():
        req = n.startswith(unfreeze)
        p.requires_grad_(req)
        if req:
            trainable.append(p)

    opt = torch.optim.AdamW(trainable, lr=4e-5, betas=(0.9, 0.99), weight_decay=1e-4)
    micro, accum = 10, 6  # 有效 batch = 60
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 8e-6 + 0.5 * (4e-5 - 8e-6) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        fm_v, swd_v, mid_v, n_p = 0.0, 0.0, 0.0, 0
        for _ in range(accum):
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

            ds = torch.rand(B, device=device) < 0.14
            dc = torch.zeros(B, dtype=torch.bool, device=device)
            lscale = torch.ones(B, device=device)

            eps = torch.randn_like(z1)
            is_ab = torch.rand(B, device=device) < 0.25
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

                swd_loss = z1.new_zeros(())
                mid_loss = z1.new_zeros(())
                valid = tau > 0.35
                if valid.sum() >= 2:
                    sub = valid.nonzero(as_tuple=True)[0][:4]
                    z1_hat = (zt[sub] + (1.0 - tau[sub]).view(-1, 1, 1, 1) * pred_v[sub].float()).clamp(-5.0, 5.0)
                    rgb_hat = decode_s10(dec, z1_hat, batch=4, enable_grad=True, output_cpu=False)
                    with torch.no_grad():
                        rgb_gt = decode_s10(dec, z1[sub], batch=4, enable_grad=False, output_cpu=False)
                        ref_toks = swd_ext.extract_tokens(rgb_gt)
                    cur_toks = swd_ext.extract_tokens(rgb_hat)
                    from exp.v20.submode_patchswd import sliced_wasserstein_1d
                    swd_loss = (
                        1.2 * sliced_wasserstein_1d(cur_toks["rgb"], ref_toks["rgb"], swd_ext.p_rgb)
                        + 0.35 * sliced_wasserstein_1d(cur_toks["s1"], ref_toks["s1"], swd_ext.p_s1)
                        + 0.45 * sliced_wasserstein_1d(cur_toks["s2"], ref_toks["s2"], swd_ext.p_s2)
                        + 0.55 * sliced_wasserstein_1d(cur_toks["s3"], ref_toks["s3"], swd_ext.p_s3)
                    ).mean()
                    swd_v += float(swd_loss.item())
                    if w_mid > 0:
                        # 复用已提取的 s2 (32×32, 192ch) 与 s3 (16×16, 384ch) 空间特征算余弦对齐，零额外 S10+ConvNeXt 显存！
                        p2, g2 = F.normalize(cur_toks["s2"], dim=-1), F.normalize(ref_toks["s2"], dim=-1)
                        p3, g3 = F.normalize(cur_toks["s3"], dim=-1), F.normalize(ref_toks["s3"], dim=-1)
                        mid_loss = 0.5 * ((1.0 - (p2 * g2).sum(-1)).mean() + (1.0 - (p3 * g3).sum(-1)).mean())
                        mid_v += float(mid_loss.item())
                    n_p += 1

                loss = (fm_loss + w_swd * swd_loss + w_mid * mid_loss) / accum

            loss.backward()
            fm_v += float(fm_loss.item()) / accum

        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V23 {arm}] {step:4d}/{updates} fm={fm_v:.4f} swd={swd_v/max(1,n_p):.4f} mid={mid_v/max(1,n_p):.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "w_swd": w_swd, "w_mid": w_mid}, out_p)
    print(f"[V23 {arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


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
    dec = load_s10_decoder(args.device)
    swd_ext = MultiScalePatchSWDExtractor(n_slices=96).to(args.device).eval()
    for p in swd_ext.parameters():
        p.requires_grad_(False)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)

    ck_pure = train_dual_swd_arm(bundle, bank, z_a_tr, dec, swd_ext, cnx, "DualSWD_pure", 0.12, 0.0, args.updates, args.device)
    ck_both = train_dual_swd_arm(bundle, bank, z_a_tr, dec, swd_ext, cnx, "DualSWD_both", 0.12, 0.15, args.updates, args.device)
    del swd_ext, cnx
    torch.cuda.empty_cache()

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "z_t4": z_t4, "y16_750": y16_750,
           "d16_750": d16_750, "eps_750": eps_750, "dec": dec}

    results = []
    for tag, ck in (("DualSWD_pure", ck_pure), ("DualSWD_both", ck_both)):
        core = PrismCore(bundle, bank, ck, device=args.device)
        results.append(eval_decoupled_dual(core, ck, ctx, f"{tag}_orig_ws42", None, None, None, 4.2, device=args.device))
        results.append(eval_decoupled_dual(core, ck, ctx, f"{tag}_orig_ws52", None, None, None, 5.2, device=args.device))
        results.append(eval_decoupled_dual(core, ck, ctx, f"{tag}_hiT_ws52", None, None, None, 5.2, tau_min=0.32, tau_hi=0.68, device=args.device))
        del core
        torch.cuda.empty_cache()

    save_json(OUT / "p1_dual_swd.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 Dual_SWD 感知笔触微调汇总 (红线 DINO-C >= 0.610, 对标 SG 52.5% / 0.5604 / 0.6113) ===", flush=True)
    for r in results:
        flag = " [DOMINATES_SG!]" if (r["dino_c"] >= 0.610 and r["dino_s"] >= 0.560) else (" [D-C>=0.61]" if r["dino_c"] >= 0.610 else "")
        print(f"{r['arm']:24s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}{flag}", flush=True)


if __name__ == "__main__":
    main()
