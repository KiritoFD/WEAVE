"""V25 Round 2 (4090): 最小受控训练消融矩阵 Tr_T0–T3 —— 单变量对照 (Content Dropout × Anchor-Bridge)。

落实 docs/19_pull_together/01_convergence_plan.md §3.2 (修正 C4):
- 配方对比一律在共同有效的 cs_j 双分支两档水位评测 (DINO-C≈0.65: D5 τ[0.28,0.66] / ≈0.70: D6-high τ[0.48,0.78]);
- drop_c=0 的臂 (T0/T2) 未训无内容分支, 不参与 full 四分支 CFG 对照 (full 仅限 T1/T3 附加分析);
- 统一协议: slf_final.pt 出发, 1500 updates, AdamW 1.2e-4→1.5e-5, wd 1e-4, EMA 0.996, clip 1.0,
  micro=120 (5 画派×24), core 80%, layout=1.0 固定, struct_w=0.10, 全臂同 seed=20263500 共享采样顺序/噪声。

| 臂  | drop_c | Anchor-Bridge |
|-----|--------|---------------|
| T0  | 0%     | 0%            |
| T1  | 30%    | 0%            |
| T2  | 0%     | 25%           |
| T3  | 30%    | 25%           |

AB 定义 (对齐推理起点分布, 同 V23 srcA0 桥): 以 25% 概率把样本的流匹配段改为
[tau0, 1] 区间 (tau0 ~ U(0.40, 0.78) 匹配推理 tau_v 范围), 起点 z_init = (1-tau0)·noise + tau0·z_src,
目标 z1 不变 —— 教会模型从"部分加噪的源锚"出发编辑 (推理时锚 = T4/PRISM-1 latent)。

用法 (4090, Round 1 结束后自动接力):
  python -m exp.v25.r2_train_ablation --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v22.core import D5_SOLVER
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v22.tmatrix import BASE_CKPT
from exp.v25.r1_inference_ablation import D6H_SOLVER, eval_latent_v25, run_core_flow
from src.utils import load_s10_decoder

R2_CONFIGS = {
    "Tr_T0": dict(drop_c_prob=0.00, ab_prob=0.00),
    "Tr_T1": dict(drop_c_prob=0.30, ab_prob=0.00),
    "Tr_T2": dict(drop_c_prob=0.00, ab_prob=0.25),
    "Tr_T3": dict(drop_c_prob=0.30, ab_prob=0.25),
}


def train_r2_arm(bundle: dict, core_bank: dict, arm: str, updates: int = 1500,
                 micro: int = 120, device: str = "cuda") -> str:
    """返回训练好的权重路径 (runs/v25/r2_tmatrix/slf_<arm>.pt)。"""
    cfg = R2_CONFIGS[arm]
    out_p = OUT / "r2_tmatrix" / f"slf_{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V25 {arm}] Reusing {out_p}", flush=True)
        return str(out_p)

    seed_all(20263500)  # 全臂同 seed → 共享采样顺序/噪声
    tr = bundle["v18_cache"]["train"]
    device_t = torch.device(device)
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
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1.5e-5 + 0.5 * (1.2e-4 - 1.5e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_fm_v, loss_st_v = 0.0, 0.0

        per_cls = max(1, micro // 5)
        use_core = (step % 5 != 0)  # 80% Core 采样
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
        lscale = torch.ones(B, device=device)  # layout=1.0 固定 (V22 T4 结论)

        z0n = torch.randn_like(z1)
        tau = torch.rand(B, device=device)
        is_ab = torch.rand(B, device=device) < cfg["ab_prob"]
        if is_ab.any():
            tau0 = 0.40 + 0.38 * torch.rand(B, device=device)  # U(0.40, 0.78) 匹配推理 tau_v 范围
            za = z1  # 真 z0_tr: 训练集干净 latent 即源锚
            z_init = (1.0 - tau0).view(B, 1, 1, 1) * z0n + tau0.view(B, 1, 1, 1) * za
            tau_ab = tau0 + (1.0 - tau0) * torch.rand(B, device=device)
            frac = ((tau_ab - tau0) / (1.0 - tau0).clamp_min(1e-4)).view(B, 1, 1, 1)
            zt_ab = z_init + frac * (z1 - z_init)
            v_ab = (z1 - z_init) / (1.0 - tau0).clamp_min(1e-4).view(B, 1, 1, 1)
            tau = torch.where(is_ab, tau_ab, tau)
            zt = torch.where(is_ab.view(B, 1, 1, 1), zt_ab, (1.0 - tau).view(B, 1, 1, 1) * z0n + tau.view(B, 1, 1, 1) * z1)
            target_v = torch.where(is_ab.view(B, 1, 1, 1), v_ab, z1 - z0n)
        else:
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0n + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0n

        with autocast(device):
            pred_v = model(zt, y16_aug, d16_aug, sid, uv, tau,
                           drop_style=ds, drop_content=dc, layout_scale=lscale)
            fm_loss = F.mse_loss(pred_v.float(), target_v)

            z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
            y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
            w_tau = (tau > 0.25).float()  # layout 固定 (V22 T4 语义): 不乘 (~dc)
            st_loss = (((y16_hat - y16).abs().mean(dim=(1, 2, 3))) * w_tau).sum() / w_tau.sum().clamp_min(1.0)
            loss = fm_loss + 0.10 * st_loss

        loss.backward()
        loss_fm_v += float(fm_loss.item())
        loss_st_v += float(st_loss.item())

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V25 {arm}] {step:4d}/{updates} fm={loss_fm_v:.4f} y16={loss_st_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "arm": arm, "cfg": cfg}, out_p)
    print(f"[V25 {arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return str(out_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1500)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261119))
    dec = load_s10_decoder(args.device)

    out_json = OUT / "r2_train_ablation.json"
    results = []
    for arm in ("Tr_T0", "Tr_T1", "Tr_T2", "Tr_T3"):
        ckpt_p = train_r2_arm(bundle, bank, arm, updates=args.updates, device=args.device)
        # C4 规则: 配方对比一律 cs_j 双分支 × 两档水位 (T4 锚点 + original Gate)
        for tag, solver in (("d5csj", dict(D5_SOLVER)), ("d6hcsj", dict(D6H_SOLVER))):
            z, _ = run_core_flow(bundle, bank, ckpt_p, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750,
                                 branches="cs_j", solver=solver, device=args.device)
            rec = eval_latent_v25(f"V25_{arm}_{tag}", z, tasks, dec, device=args.device)
            rec["train_arm"], rec["eval_solver"] = arm, tag
            results.append(rec)
            save_json(out_json, {"updated_utc": utc(), "results": results})

    print("\n=== V25 Round 2 训练消融矩阵汇总 (cs_j 两档) ===", flush=True)
    for r in results:
        print(f"{r['arm']:24s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
