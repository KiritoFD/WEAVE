"""V22-P2 训练消融矩阵 T1–T5（同预算公平对照，writing-block(2) §五）。

统一协议: 从 slf_final.pt 出发, 1800 updates, AdamW 1.3e-4→1.5e-5, wd 1e-4,
EMA 0.996, clip 1.0, 有效 batch 120 (5 画派×24), 同 seed 共享采样顺序/噪声。

| 臂 | Core采样 | 内容Dropout | Layout训练 | 结构损失 |
|---|---|---|---|---|
| T1 | 80% Core | 30% | 连续 | 0.10 |
| T2 | 0% Core | 30% | 连续 | 0.10 |
| T3 | 80% Core | 8%(原始) | 连续 | 0.10 |
| T4 | 80% Core | 30% | 固定 1.0 | 0.10 |
| T5 | 80% Core | 30% | 连续 | 0 |

T0 = slf_final 本身（预训练锚点，无需训练）。
训练完成后用固定门控 wc11/wc14 两档评测（Core-Simple 冻结配置）。
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

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v19.fiber_duality import build_or_load_v19_foundation

BASE_CKPT = ROOT / "runs/v20/slf_flow/slf_final.pt"

T_CONFIGS = {
    "T1": dict(core_frac=0.8, drop_c_prob=0.30, layout_train=True, struct_w=0.10),
    "T2": dict(core_frac=0.0, drop_c_prob=0.30, layout_train=True, struct_w=0.10),
    "T3": dict(core_frac=0.8, drop_c_prob=0.08, layout_train=True, struct_w=0.10),
    "T4": dict(core_frac=0.8, drop_c_prob=0.30, layout_train=False, struct_w=0.10),
    "T5": dict(core_frac=0.8, drop_c_prob=0.30, layout_train=True, struct_w=0.0),
    "T4L": dict(core_frac=0.8, drop_c_prob=0.30, layout_train=False, struct_w=0.10),  # 长训练臂，--updates 3600
    "T4S": dict(core_frac=0.8, drop_c_prob=0.30, layout_train=False, struct_w=0.10),  # 快照曲线臂，--updates 3600，每 300 步存快照
    "E99": dict(core_frac=0.8, drop_c_prob=0.30, layout_train=False, struct_w=0.10, ema=0.999),  # EMA 滞后检验
    "E97": dict(core_frac=0.8, drop_c_prob=0.30, layout_train=False, struct_w=0.10, ema=0.997),
}


def train_t_arm(bundle: dict, core_bank: dict, arm: str, updates: int = 1800,
                micro: int = 120, device: str = "cuda") -> Path:
    cfg = T_CONFIGS[arm]
    out_p = OUT / "tmatrix" / f"slf_{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V22 {arm}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261205)  # 全臂同 seed → 共享采样顺序/噪声
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
    pools = core_cls_idx if cfg["core_frac"] > 0 else all_cls_idx
    core_every = 5 if cfg["core_frac"] > 0 else 1  # T2: 永远全池

    model = SLFUNet(base_dim=144, u_dim=128).to(device)
    init_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)["ema"]
    model.load_state_dict(init_state, strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    ema_decay = cfg.get("ema", 0.996)

    accum = 1
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
            use_core = (cfg["core_frac"] > 0) and (step % core_every != 0)
            cur_pools = core_cls_idx if use_core else all_cls_idx
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in cur_pools], dim=0)
            B = len(idx)

            z1 = z_all[idx].to(device, non_blocking=True).float()
            sid = sid_all[idx].to(device, non_blocking=True)
            d16 = d16_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            uv = u_exact_all[idx].to(device, non_blocking=True).float()

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            u_r = torch.rand(B, device=device)
            ds = u_r < 0.12
            dc = (u_r >= 0.12) & (u_r < 0.12 + cfg["drop_c_prob"])
            both = u_r >= 0.95
            ds = ds | both
            dc = dc | both

            if cfg["layout_train"]:
                lscale = torch.where(dc, torch.zeros(B, device=device),
                                     0.15 + 0.85 * torch.rand(B, device=device))
            else:
                lscale = torch.ones(B, device=device)

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
                w_tau = (tau > 0.25).float()
                if cfg["layout_train"]:
                    w_tau = w_tau * ((~dc).float())
                st_loss = (((y16_hat - y16).abs().mean(dim=(1, 2, 3))) * w_tau).sum() / w_tau.sum().clamp_min(1.0)
                loss = (fm_loss + cfg["struct_w"] * st_loss) / accum

            loss.backward()
            loss_fm_v += float(fm_loss.item()) / accum
            loss_st_v += float(st_loss.item()) / accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(ema_decay).add_(pw, alpha=1.0 - ema_decay)

        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[V22 {arm}] {step:4d}/{updates} fm={loss_fm_v:.4f} y16={loss_st_v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if arm.endswith("S") and step % 300 == 0:  # 快照臂：训练曲线定位
            torch.save({"ema": ema.state_dict(), "updates": step, "arm": arm, "cfg": cfg},
                       OUT / "tmatrix" / f"slf_{arm}snap{step}.pt")

    torch.save({"ema": ema.state_dict(), "updates": updates, "arm": arm, "cfg": cfg}, out_p)
    print(f"[V22 {arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1800)
    ap.add_argument("--arms", default="T1,T2,T3,T4,T5")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    core_bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)

    for arm in args.arms.split(","):
        train_t_arm(bundle, core_bank, arm.strip(), updates=args.updates, device=args.device)
    save_json(OUT / "tmatrix_done.json", {"updated_utc": utc(), "arms": args.arms, "updates": args.updates})


if __name__ == "__main__":
    main()
