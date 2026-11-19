"""V23 方向校准 D (4090): 围绕 F3 (0.5522 @ 0.6096) 做双分支高 DINO-C 前沿突破。

洞察:
- F3_dual_fixed_orig (V22 Dual 配方, 无 w_mid 压制) 是全仓除 StyleGallery 外 DINO-S@DINO-C>=0.61 最高点 (0.5522 @ 0.6096)。
- DualC1AB 加了 w_mid=0.35 把 DINO-S 压回了 0.546-0.550。
- 本实验:
  1. 在 4090 上以完全对等协议重训两套轻量双分支模型 (1500 updates, drop_content=False, layout=1.0):
     - Dual_Pure: 纯 V22 Dual 配方 (0% w_mid, 0% AB, 验证复现 F3 并做解耦门控冲刺)
     - Dual_AB25: 0% w_mid + 25% Anchor-Bridge (tau0 in [0.38, 0.68], 修复起点不匹配同时不锁纹理)
  2. 测试高内容起点 + 松布局解耦门控组合:
     - 高 g_tau (tau0 偏高, 0.48~0.58) 锁死 DINO-C >= 0.612;
     - 低/中 g_layout (lscale ~ 0.45~0.75) 释放 16x16 结构束缚拉高 DINO-S;
     - ws_fig in {4.8, 5.6, 6.4} 增强内容下风格梯度!

用法 (4090):
  python -m exp.v23.p1_dual_frontier_sprint --pool ../pseudo_gt_24h/orig_mean_fp16
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
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore, build_content_condition, make_initial_state, predict_guided_velocity, rk2_integrate
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from src.utils import load_s10_decoder

BASE_CKPT = ROOT / "runs/v20/slf_flow/slf_final.pt"


def train_dual_variant(bundle: dict, core_bank: dict, z_a_tr: torch.Tensor | None,
                       tag: str, ab_prob: float, updates: int = 1600, device: str = "cuda") -> Path:
    out_p = OUT / "dual_sprint" / f"slf_{tag}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 {tag}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20262300)
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
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    micro = 120
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1.5e-5 + 0.5 * (1.2e-4 - 1.5e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        per_cls = micro // 5
        cur_pools = core_cls_idx if (step % 5 != 0) else all_cls_idx
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

        ds = torch.rand(B, device=device) < 0.14
        dc = torch.zeros(B, dtype=torch.bool, device=device)
        lscale = torch.ones(B, device=device)
        eps = torch.randn_like(z1)

        if ab_prob > 0 and z_a_tr is not None:
            za = z_a_tr[idx].to(device, non_blocking=True).float()
            is_ab = torch.rand(B, device=device) < ab_prob
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
        else:
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * eps + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - eps

        with autocast(device):
            pred_v = model(zt, y16_aug, d16_aug, sid, uv, tau,
                           drop_style=ds, drop_content=dc, layout_scale=lscale)
            fm_loss = F.mse_loss(pred_v.float(), target_v)
            z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
            y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
            st_loss = (y16_hat - y16).abs().mean()
            loss = fm_loss + 0.10 * st_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[V23 {tag}] {step:4d}/{updates} fm={fm_loss.item():.4f} lcn={st_loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "ab_prob": ab_prob}, out_p)
    print(f"[V23 {tag}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_decoupled_dual(core: PrismCore, ckpt_path: Path, ctx: dict, arm_name: str,
                        g_tau: float | None, g_layout: float | None, g_guid: float,
                        ws_fig: float, tau_min: float = 0.28, tau_hi: float = 0.66,
                        device: str = "cuda") -> dict:
    """支持独立解耦 (g_tau, g_layout, g_guid) 的双分支 cs_j 评测（None 表示沿用 original sigmoid 门控）。"""
    cfg_hash = config_hash({"arm": arm_name, "ckpt": file_sha(ckpt_path)[:12],
                            "gt": g_tau, "gl": g_layout, "gg": g_guid, "ws": ws_fig, "tmin": tau_min, "thi": tau_hi})
    if (OUT / "raw" / arm_name / "meta.json").exists():
        old = read_json(OUT / "raw" / arm_name / "meta.json")
        if old.get("cfg_hash") == cfg_hash and validate_raw_meta(old):
            s = read_json(OUT / f"measured/{arm_name}/summary_v23.json")
            c = s["cross600"]
            return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                    "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "lpips": round(c["lpips"], 5)}

    tasks, z_src_750 = ctx["tasks"], ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    solver = dict(D5_SOLVER, tau_min=tau_min, tau_hi=tau_hi, ws_fig=ws_fig)
    cfg = CFGConfig(branches="cs_j", rescale_phi=0.48)
    wr = RawWriterV23(arm_name, dec=ctx["dec"], device=device)
    t0 = time.time()

    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        d16_b = ctx["d16_750"][sl].to(device).float()
        y16_b = ctx["y16_750"][sl].to(device).float()
        tid_b = tid_all[sl].to(device)
        u_style, dsd_ref = core.build_style_condition(d16_b, tid_b, mode="core8")
        g_orig = torch.sigmoid((dsd_ref - float(core.bank["mid_dsd"])) * 550.0)
        gt = g_orig if g_tau is None else torch.full_like(dsd_ref, g_tau)
        gl = g_orig if g_layout is None else torch.full_like(dsd_ref, g_layout)
        gg = g_orig if g_guid is None else torch.full_like(dsd_ref, g_guid)

        lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * gl).clamp(0.15, 1.0)
        content = build_content_condition(y16_b, d16_b, gl)
        tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * gt).clamp(0.0, 0.78)
        z = make_initial_state(ctx["eps_750"][sl].to(device).float(), ctx["z_t4"][sl].to(device).float(), tau_v)

        z_gen = rk2_integrate(
            lambda zz, prog: predict_guided_velocity(
                core.model, zz, tau_v + prog * (1.0 - tau_v), content, tid_b, u_style,
                lscale, gg, cfg, solver, prog,
            ),
            z, tau_v, int(solver["steps"]),
        )
        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), z_gen)
        wr.write_batch(z_out, tasks[sl])

    meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_path),
                       tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                       seed=EVAL_SEED, wall_sec=round(time.time() - t0, 1))
    assert validate_raw_meta(meta)
    s = run_measure_v23(arm_name, device=device)
    export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
    c = s["cross600"]
    return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
            "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
            "lpips": round(c["lpips"], 5), "same_top1": round(s["same150"]["csd_top1"], 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    z_a_tr = build_or_load_train_prism1_anchors(bundle, device=args.device)

    ck_pure = train_dual_variant(bundle, bank, None, "Dual_Pure", 0.0, updates=1600, device=args.device)
    ck_ab25 = train_dual_variant(bundle, bank, z_a_tr, "Dual_AB25", 0.25, updates=1600, device=args.device)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(args.device)
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "z_t4": z_t4, "y16_750": y16_750,
           "d16_750": d16_750, "eps_750": eps_750, "dec": dec}

    results = []
    for tag, ck in (("DualPure", ck_pure), ("DualAB25", ck_ab25)):
        core = PrismCore(bundle, bank, ck, device=args.device)
        # 1) orig gate 增强风格强度 ws_fig=5.2
        results.append(eval_decoupled_dual(core, ck, ctx, f"{tag}_orig_ws52", None, None, None, 5.2, device=args.device))
        # 2) 解耦门控: 高 g_tau=0.68 保 DINO-C>=0.612 + 松 g_layout=0.35 放风格 + ws_fig=5.8
        results.append(eval_decoupled_dual(core, ck, ctx, f"{tag}_dec_gt68_gl35_w58", 0.68, 0.35, 0.70, 5.8, device=args.device))
        # 3) 解耦门控: g_tau=0.72 + g_layout=0.45 + ws_fig=6.4
        results.append(eval_decoupled_dual(core, ck, ctx, f"{tag}_dec_gt72_gl45_w64", 0.72, 0.45, 0.75, 6.4, device=args.device))
        del core
        torch.cuda.empty_cache()

    save_json(OUT / "p1_dual_frontier_sprint.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 双分支高 DINO-C 前沿冲刺汇总 (对标 SG: 52.50% / 0.5604 / 0.6113) ===", flush=True)
    for r in results:
        flag = " [DOMINATES_SG!]" if (r["dino_c"] >= 0.610 and r["dino_s"] >= 0.560) else (" [D-C>=0.61]" if r["dino_c"] >= 0.610 else "")
        print(f"{r['arm']:26s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}{flag}", flush=True)


if __name__ == "__main__":
    main()
