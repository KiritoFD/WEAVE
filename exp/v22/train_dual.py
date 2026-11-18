"""V22-P2-0：双分支重训（取消 Content Dropout，layout_scale=1.0）—— One-Shot 的内容条件正确化基座。

设计（writing-block P2 §2.1/2.2/6.2）:
  - 训练期 drop_content=False 恒成立（不再训练 v_s0 / v_00 主分支）；
  - drop_style 保留 0.12（供推理期空风格分支 v_0c 使用）;
  - layout_scale = 1.0 固定（内容约束永不缩放）;
  - 80% Core-kNN 核心采样保持;
  - 从 V20 slf_final.pt（5000-step 预训练锚点）出发，1800 updates，AdamW 1.3e-4→1.5e-5。

可在 3060 (12GB) 上独立运行：python -m exp.v22.train_dual --pool ...
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

from . import OUT, ROOT, autocast, read_json, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc
from .t4 import T4_HASH, build_or_load_t4_latents_750
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v19.fiber_duality import build_or_load_v19_foundation

BASE_CKPT = ROOT / "runs/v20/slf_flow/slf_final.pt"


def train_dual(bundle: dict, core_bank: dict, updates: int = 1800, micro: int | None = None,
               accum: int | None = None, device: str = "cuda") -> Path:
    out_p = OUT / "attn_flow/prism_dual_branch.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V22 Dual] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261201)
    tr = bundle["v18_cache"]["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    model = SLFUNet(base_dim=144, u_dim=128).to(device)
    init_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)["ema"]
    model.load_state_dict(init_state, strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    if micro is None or accum is None:
        micro, accum = tier_micro_accum(120, default_micro_24gb=120, max_micro_12gb=24, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1.3e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1.5e-5 + 0.5 * (1.3e-4 - 1.5e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_fm_v, loss_st_v = 0.0, 0.0

        for _ in range(accum):
            per_cls = max(1, micro // 5)
            pools = core_cls_idx if (step % 5 != 0) else all_cls_idx
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in pools], dim=0)
            B = len(idx)

            z1 = z_all[idx].to(device, non_blocking=True).float()
            sid = sid_all[idx].to(device, non_blocking=True)
            d16 = d16_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            uv = u_exact_all[idx].to(device, non_blocking=True).float()

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            # P2 §2.1: 只 drop_style；内容永不 dropout
            ds = torch.rand(B, device=device) < 0.12
            dc = torch.zeros(B, dtype=torch.bool, device=device)
            lscale = torch.ones(B, device=device)  # P2 §2.2: layout_scale = 1.0

            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = model(zt, y16_aug, d16_aug, sid, uv, tau,
                               drop_style=ds, drop_content=dc, layout_scale=lscale)
                fm_loss = F.mse_loss(pred_v.float(), target_v)

                z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
                y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                st_loss = ((y16_hat - y16).abs().mean(dim=(1, 2, 3))).mean()
                loss = (fm_loss + 0.10 * st_loss) / accum

            loss.backward()
            loss_fm_v += float(fm_loss.item()) / accum
            loss_st_v += float(st_loss.item()) / accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 200 == 0 or step == updates:
            print(f"[V22 Dual] {step:4d}/{updates} fm={loss_fm_v:.4f} y16={loss_st_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "params": ema.n_params(),
                "base_ckpt": str(BASE_CKPT), "training": "dual_branch_no_content_dropout"},
               out_p)
    print(f"[V22 Dual] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1800)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    core_bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    ckpt = train_dual(bundle, core_bank, updates=args.updates, device=args.device)
    save_json(OUT / "train_dual_done.json", {
        "updated_utc": utc(), "ckpt": str(ckpt),
        "base_ckpt": str(BASE_CKPT), "updates": args.updates, "t4_hash": str(T4_HASH),
    })


if __name__ == "__main__":
    main()
