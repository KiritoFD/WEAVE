"""Unified Student Trainer & Evaluator for Phase 3 (M0-M3), Phase 4 (6-Arm Duel), and Phase 5 (5k Dual-Seed)."""
from __future__ import annotations

import argparse
import copy
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exp.v18 import STYLES, tasks_and_sources
from exp.v27_single.bare_q32 import load_cache
from exp.v27_single.unified.train import conditioning_stats
from exp.v28_unified import (
    OUT,
    ROOT,
    SEED_TRAIN,
    RawWriterV28,
    assert_vram_budget,
    atomic_save,
    hard_hits,
    run_measure_v28,
)
from exp.v28_unified.student_models import PixelBridgeV28, build_student_model
from src.eval.metrics import paired_cluster_ci
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import read_json, sanitize_cuda_env, save_json, seed_all
from src.utils.structure import dwt2_haar

# 7 evaluation levels spanning content -> balanced -> strong -> extrapolation
EVAL_LEVELS_V28 = (0.0, 0.25, 0.40, 0.60, 0.75, 1.00, 1.15)
MODE_PATTERN_2TIER = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
MODE_PATTERN_3TIER = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])


def _sample_strengths_and_targets(
    modes_base: torch.Tensor,
    n_cross: int,
    gen: torch.Generator,
    interp_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (strengths (n_cross,), tier_code (n_cross,)) for cross tasks.

    For 2tier: 0->0.45(hiC), 1->1.00(hiS), 2->[0.05,0.45](low lerp), 3->[0.45,1.00](mid lerp).
    For 3tier: 0->0.40±0.03(hiC), 1->0.75±0.04(mid), 2->1.00[-0.04,0](hiS), 3->[0.04,0.40](low lerp).
    """
    r = torch.rand(n_cross, generator=gen)
    jit = torch.rand(n_cross, generator=gen)
    exact_flag = torch.rand(n_cross, generator=gen) < 0.55
    if interp_mode == "2tier":
        s = torch.where(
            modes_base == 0, 0.45,
            torch.where(modes_base == 1, 1.00,
                        torch.where(modes_base == 2, 0.05 + 0.40 * r, 0.45 + 0.55 * r)),
        )
        return s, modes_base
    # 3tier: anchor around real endpoints {0.40, 0.75, 1.00} + low-end [0.04, 0.40]
    s0 = torch.where(exact_flag, torch.full_like(r, 0.40), 0.37 + 0.06 * jit)
    s1 = torch.where(exact_flag, torch.full_like(r, 0.75), 0.71 + 0.08 * jit)
    s2 = torch.where(exact_flag, torch.full_like(r, 1.00), 0.96 + 0.04 * jit)
    s3 = 0.04 + 0.36 * r
    s = torch.where(
        modes_base == 0, s0,
        torch.where(modes_base == 1, s1, torch.where(modes_base == 2, s2, s3)),
    )
    return s, modes_base


def _compute_targets(
    source: torch.Tensor,
    hi_c: torch.Tensor,
    mid: torch.Tensor | None,
    hi_s: torch.Tensor,
    strength: torch.Tensor,
    tier_code: torch.Tensor,
    interp_mode: str,
) -> torch.Tensor:
    alpha = strength[:, None, None, None]
    if interp_mode == "2tier":
        low = source.lerp(hi_c, (alpha / 0.45).clamp(0.0, 1.0))
        high = hi_c.lerp(hi_s, ((alpha - 0.45) / 0.55).clamp(0.0, 1.0))
        return torch.where(alpha <= 0.45, low, high)
    # 3tier: zero blind interpolation between hiC/mid/hiS!
    low = source.lerp(hi_c, (alpha / 0.40).clamp(0.0, 1.0))
    t0 = torch.where(tier_code[:, None, None, None] == 0, hi_c,
         torch.where(tier_code[:, None, None, None] == 1, mid if mid is not None else hi_c.lerp(hi_s, 0.55),
         torch.where(tier_code[:, None, None, None] == 2, hi_s, low)))
    return t0


def _latent_loss_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    levels: torch.Tensor,
    priors: dict,
    loss_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (smooth_l1_term, full_latent_loss)."""
    direct_ps = F.smooth_l1_loss(pred, target, beta=0.06, reduction="none").mean(dim=(1, 2, 3))
    if loss_mode == "hiswd":
        # Relax high-lambda latent Smooth L1 averaging contraction so pixel SWD can recover textures
        high_damp = torch.where(levels > 0.6, torch.full_like(levels, 0.55), torch.ones_like(levels))
        direct = (direct_ps * high_damp).mean()
    else:
        direct = direct_ps.mean()
    fp, ft = dwt2_haar(pred), dwt2_haar(target)
    bands = sum(w * F.l1_loss(a, b) for w, a, b in zip((1.0, 1.0, 0.5), fp[1:], ft[1:]))
    dy = torch.einsum("ij,bjhw->bihw", priors["P_Y"], pred - target).abs().mean(dim=(1, 2, 3))
    if loss_mode == "base":
        luma = dy.mean()
        return direct, direct + 0.35 * bands + 0.25 * luma
    # Rebalanced M2/M3/Phase4.5: luma L1 weight 0.25 -> 0.10 AND active only at low lambda!
    low_gate = ((0.55 - levels) / 0.55).clamp(0.0, 1.0)
    luma = (dy * low_gate).mean()
    return direct, direct + 0.35 * bands + 0.10 * luma


@torch.no_grad()
def evaluate_student_v28(
    model: torch.nn.Module,
    arch: str,
    raw_stats: torch.Tensor,
    priors: dict,
    ts: dict,
    q: torch.Tensor,
    *,
    tag: str,
    levels: tuple[float, ...] = EVAL_LEVELS_V28,
    wrong: bool = False,
    device: str = "cuda",
) -> dict:
    tasks, src, _ = tasks_and_sources()
    sid = torch.tensor([t["source_style_id"] for t in tasks])
    tid = torch.tensor([t["target_style_id"] for t in tasks])
    labels = pick_wrong_cross_tid(sid, tid) if wrong else tid
    dec = load_s10_decoder(device)
    eps_fixed = torch.randn(src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    out = {}
    name_tags = ("wrong",) if wrong else tuple(f"l{int(round(l * 100)):03d}" for l in levels)
    model.eval()
    for name, strength in zip(name_tags, levels):
        arm = f"{tag}_{name}"
        identity = 0.0
        meta = OUT / "raw" / arm / "meta.json"
        if not (OUT / "measured" / arm / "summary_v28.json").exists():
            writer = RawWriterV28(arm, dec=dec, device=device)
            for st in range(0, len(tasks), 32):
                sl = slice(st, min(st + 32, len(tasks)))
                zs = src[sl].to(device).float()
                tgt = labels[sl].to(device)
                st_t = torch.full_like(tgt, strength, dtype=torch.float32)
                if arch in ("unet", "bridge4"):
                    z = model(zs, ts["c_64"][sl].to(device).float(), q[sl].to(device).float(),
                              tgt, st_t, raw_stats, priors, noise=eps_fixed[sl].to(device).float())
                else:
                    z = model(zs, ts["c_64"][sl].to(device).float(), q[sl].to(device).float(),
                              tgt, st_t, raw_stats, priors)
                if strength == 0.0:
                    identity = max(identity, float((z - zs).abs().max()))
                    if identity > 0.0:
                        raise RuntimeError(f"Lambda-zero latent identity failed: {identity}")
                writer.write_batch(z, tasks[sl])
            writer.finalize(arm=arm, strength=strength, nfe=4 if arch == "bridge4" else 1, same_bypass=False,
                            lambda_zero_max_abs=identity if strength == 0.0 else None)
        row = run_measure_v28(arm, device=device)
        row.update(strength=strength, wrong=wrong,
                   lambda_zero_max_abs=read_json(meta).get("lambda_zero_max_abs") if meta.exists() else None)
        out[name] = row
    del dec
    torch.cuda.empty_cache()
    return out


def run_final_eval(
    model: torch.nn.Module,
    arch: str,
    raw_stats: torch.Tensor,
    priors: dict,
    ts: dict,
    q: torch.Tensor,
    steps: int,
    *,
    run_dir: Path,
    tag: str,
    checkpoint_name: str,
    device: str = "cuda",
) -> dict:
    levels = evaluate_student_v28(model, arch, raw_stats, priors, ts, q, tag=tag, levels=EVAL_LEVELS_V28, device=device)
    levels.update(evaluate_student_v28(model, arch, raw_stats, priors, ts, q, tag=tag, levels=(1.0,), wrong=True, device=device))

    # Paired bootstrap CI of l100 vs l040 and vs old ema_s4000 if measured exists
    rows_l100 = read_json(OUT / "measured" / f"{tag}_l100" / "eval.json")["rows"]
    rows_l075 = read_json(OUT / "measured" / f"{tag}_l075" / "eval.json")["rows"]
    rows_l040 = read_json(OUT / "measured" / f"{tag}_l040" / "eval.json")["rows"]

    high, mid, strong = levels["l040"], levels["l075"], levels["l100"]
    monotonic_dino_s = (levels["l025"]["dino_s"] <= high["dino_s"] <= mid["dino_s"] <= strong["dino_s"])
    monotonic_top1 = (levels["l025"]["csd_top1"] <= high["csd_top1"] <= mid["csd_top1"] <= strong["csd_top1"])
    no_l060_hole = (
        levels["l060"]["dino_s"] >= high["dino_s"] - 0.002
        and levels["l060"]["csd_top1"] >= high["csd_top1"] - 0.02
        and mid["csd_top1"] >= levels["l060"]["csd_top1"] - 0.02
    )

    report = {
        "arm_tag": tag,
        "arch": arch,
        "checkpoint": checkpoint_name,
        "student_updates": steps,
        "nfe": 4 if arch == "bridge4" else 1,
        "same_bypass": False,
        "arms": levels,
        "bootstrap_ci": {
            "l100_minus_l040": paired_cluster_ci(rows_l100, rows_l040),
            "l075_minus_l040": paired_cluster_ci(rows_l075, rows_l040),
        },
        "acceptance": {
            "exact_latent_identity": levels["l000"]["lambda_zero_max_abs"] == 0.0,
            "monotonic_dino_s": monotonic_dino_s,
            "monotonic_top1": monotonic_top1,
            "monotonic_no_l060_hole": no_l060_hole,
            "high_content_l040": (
                high["dino_c"] >= 0.72 and high["dino_s"] >= 0.49
                and high["csd_top1"] >= 0.52 and high["same150_top1"] >= 0.95
            ),
            "balanced_l075": (
                mid["dino_c"] >= 0.65 and mid["dino_s"] >= 0.53 and mid["csd_top1"] >= 0.58
            ),
            "strong_style_l100": (
                strong["dino_c"] >= 0.58 and strong["dino_s"] >= 0.55
                and strong["csd_top1"] >= 0.63
                and strong["hits"]["ER"][0] >= 45
                and strong["hits"]["Uki"][0] >= 45
                and strong["same150_top1"] >= 0.93
            ),
            "wrong_target": levels["wrong"]["csd_top1"] <= 0.020,
        },
    }
    save_json(run_dir / "results.json", report)
    return report


def train_student_arm(
    arm_name: str,
    pool_path: Path,
    *,
    arch: str = "unet",
    interp_mode: str = "3tier",
    loss_mode: str = "calibrated",
    pool_variant: str = "main",     # main | rand | old
    pool_c_path: Path | None = None,
    pool_c_field: str = "z_real",   # z_real | z_fwd | z_shuf
    curriculum_steps: int | None = None,
    updates: int = 2800,
    seed: int = SEED_TRAIN,
    eff_batch: int = 128,
    micro: int = 41,
    pixel_micro: int = 6,
    lr: float = 1.15e-3,
    min_lr: float = 6e-5,
    warmup_steps: int = 150,
    weight_decay: float = 2.0e-4,
    ema_decay: float = 0.006,
    smoke_steps: int = 0,
    eval_every: int = 700,
    device: str = "cuda",
) -> dict:
    """Train one student arm with optional 20-step smoke check, resume support, and full test750 eval."""
    sanitize_cuda_env()
    seed_all(seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 8
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    run_dir = OUT / "students" / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = run_dir / f"{arm_name}_final.pt"

    tr = load_cache("runs/v18/cache/ccf_pool_cache.pt")["train"]
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    c21 = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")
    foundation = load_cache("runs/v19/cache/v19_foundation.pt")
    priors = {k: foundation["priors"][k].to(device).float() for k in ("P_Y", "P_C", "mu_c", "std_c")}

    pixel = PixelBridgeV28(device=device)
    raw_stats = conditioning_stats(tr, foundation, pixel)
    del foundation

    pairs = torch.load(pool_path, map_location="cpu", weights_only=False)
    if pool_variant == "rand":
        pair_hic, pair_mid, pair_his = pairs["rand_hiC"], pairs["rand_mid"], pairs["rand_hiS"]
    else:
        pair_hic = pairs["hiC"]
        pair_mid = pairs.get("mid", None)
        pair_his = pairs["hiS"]

    pool_c = torch.load(pool_c_path, map_location="cpu", weights_only=False) if pool_c_path is not None else None

    # Standardize pixel moments on deterministic slice of target latents
    ids_std = (~pairs["same"]).nonzero().flatten()[::125][:48]
    with torch.no_grad():
        xx = torch.cat([pixel.encode(pair_his[idx].to(device).float(), False)[0] for idx in ids_std.split(4)])
        pixel.scale = xx.std(0, unbiased=False).clamp_min(0.03)

    model = build_student_model(arch, raw_stats.shape[-1], device=device)
    num_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed + 1)

    if final_ckpt.exists() and smoke_steps == 0 and (run_dir / "results.json").exists():
        print(f"[Train {arm_name}] Already complete -> {run_dir / 'results.json'}", flush=True)
        return read_json(run_dir / "results.json")

    same_ids = pairs["same"].nonzero().flatten()
    cross_ids = (~pairs["same"]).nonzero().flatten()

    # 50-step Auto-Gradient Norm Calibration for M2/M3/Phase 4/Phase 4.5/Phase 5
    pixel_scale = 1.0
    if loss_mode in ("calibrated", "hiswd"):
        calib_p = run_dir / "calib.json"
        if calib_p.exists() and smoke_steps == 0:
            pixel_scale = float(read_json(calib_p)["pixel_scale"])
        else:
            cal_swd_w = 0.35 if loss_mode == "hiswd" else 0.15
            cal_ratio = 1.0 if loss_mode == "hiswd" else 0.65
            g_lat_sum, g_pix_sum = 0.0, 0.0
            n_cal = 12 if smoke_steps > 0 else 50
            cal_params = [p for p in model.parameters() if p.requires_grad]
            for c_st in range(n_cal):
                c_idx = cross_ids[torch.randint(0, len(cross_ids), (pixel_micro,), generator=gen)]
                src_i = pairs["src"][c_idx]
                zs = tr["z"][src_i].to(device).float()
                lv = torch.full((pixel_micro,), 1.0, device=device)
                tgt = pair_his[c_idx].to(device).float()
                pg_c = torch.Generator(device=device).manual_seed(seed + 90000 + c_st)
                proj_c = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=pg_c), mode="reduced").Q
                               for c in (64, 128, 256))
                if arch == "unet":
                    pred = model(zs, tr["c_64"][src_i].to(device).float(), c21["train"]["q32"][src_i].to(device).float(),
                                 pairs["tid"][c_idx].to(device), lv, raw_stats, priors, noise=torch.randn_like(zs))
                else:
                    pred = model(zs, tr["c_64"][src_i].to(device).float(), c21["train"]["q32"][src_i].to(device).float(),
                                 pairs["tid"][c_idx].to(device), lv, raw_stats, priors)
                l_lat, _ = _latent_loss_terms(pred, tgt, lv, priors, loss_mode)
                l_pix = pixel.loss(pred, tgt, proj_c, swd_weight=cal_swd_w)
                gl = torch.autograd.grad(l_lat, cal_params, retain_graph=True, allow_unused=True)
                gp = torch.autograd.grad(l_pix, cal_params, allow_unused=True)
                g_lat_sum += float(torch.sqrt(sum((g.float().square().sum() for g in gl if g is not None), torch.tensor(0.0, device=device))).item())
                g_pix_sum += float(torch.sqrt(sum((g.float().square().sum() for g in gp if g is not None), torch.tensor(0.0, device=device))).item())
            pixel_scale = float(max(0.05, min(10.0, cal_ratio * (g_lat_sum / max(g_pix_sum, 1e-6)))))
            save_json(calib_p, {
                "n_cal": n_cal,
                "mean_grad_latent": round(g_lat_sum / n_cal, 4),
                "mean_grad_pixel": round(g_pix_sum / n_cal, 4),
                "pixel_scale": round(pixel_scale, 4),
                "target_ratio": cal_ratio,
            })
            print(f"[{arm_name}] Auto-Grad Calib ({n_cal} steps): pixel_scale={pixel_scale:.4f} "
                  f"(ratio {cal_ratio:.2f}:1.0)", flush=True)

    resume_p = run_dir / "resume.pt"
    first = 1
    if resume_p.exists() and smoke_steps == 0:
        state = torch.load(resume_p, map_location="cpu", weights_only=False)
        if state["updates"] != updates:
            raise ValueError(f"Saved updates {state['updates']} != requested {updates}")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        gen.set_state(state["sampling"])
        first = state["step"] + 1

    target_steps = smoke_steps if smoke_steps > 0 else updates
    mode_pat = MODE_PATTERN_2TIER if interp_mode == "2tier" else MODE_PATTERN_3TIER
    c_tail = curriculum_steps if curriculum_steps is not None else max(500, updates // 4)
    curriculum_start = max(0, updates - c_tail)  # Last tail steps for Pool C TeleStyle mixing

    print(f"[{arm_name}] START {first}/{target_steps} | arch={arch} ({num_params:,} params) "
          f"interp={interp_mode} loss={loss_mode} bs={eff_batch} micro={micro} pix_micro={pixel_micro} "
          f"lr={lr:.2e} wd={weight_decay:.1e} ema={ema_decay}", flush=True)

    t0 = time.monotonic()
    for step in range(first, target_steps + 1):
        frac = max(0, step - warmup_steps) / max(1, updates - warmup_steps)
        cur_lr = (
            lr * step / max(1, warmup_steps)
            if step <= warmup_steps
            else min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * frac))
        )
        if pool_c is not None and step > curriculum_start:
            cur_lr *= 0.30
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)

        # Sample eff_batch indices (with 20% Pool C curriculum mix in last 25% steps if enabled)
        use_c_mix = (pool_c is not None) and (step > curriculum_start)
        n_pool_c = int(round(0.20 * eff_batch)) if use_c_mix else 0
        n_main = eff_batch - n_pool_c
        prev_same, next_same = ((step - 1) * n_main) // 5, (step * n_main) // 5
        n_same = next_same - prev_same
        n_cross = n_main - n_same

        si = same_ids[torch.randint(len(same_ids), (n_same,), generator=gen)]
        ci = cross_ids[torch.randint(len(cross_ids), (n_cross,), generator=gen)]
        prev_cross = (step - 1) * n_main - prev_same
        modes_b = mode_pat[torch.arange(prev_cross, prev_cross + n_cross) % len(mode_pat)]
        modes_b = modes_b[torch.randperm(n_cross, generator=gen)]
        cross_l, cross_tc = _sample_strengths_and_targets(modes_b, n_cross, gen, interp_mode)

        same_l = torch.rand(n_same, generator=gen)
        same_tc = torch.zeros(n_same, dtype=torch.long)

        main_idx = torch.cat((ci, si))
        main_l = torch.cat((cross_l, same_l))
        main_tc = torch.cat((cross_tc, same_tc))
        is_c_flag = torch.zeros(n_main, dtype=torch.bool)

        if n_pool_c > 0:
            c_rows = torch.randint(len(pool_c["tid"]), (n_pool_c,), generator=gen)
            all_idx = torch.cat((main_idx, c_rows))
            all_l = torch.cat((main_l, torch.ones(n_pool_c)))
            all_tc = torch.cat((main_tc, torch.full((n_pool_c,), 2, dtype=torch.long)))
            is_c_flag = torch.cat((is_c_flag, torch.ones(n_pool_c, dtype=torch.bool)))
        else:
            all_idx, all_l, all_tc = main_idx, main_l, main_tc

        # Ensure pixel slice (first pixel_micro items) consists of cross samples
        perm = torch.randperm(eff_batch, generator=gen)
        is_cross_all = torch.cat((torch.ones(n_cross, dtype=torch.bool), torch.zeros(n_same, dtype=torch.bool),
                                  torch.ones(n_pool_c, dtype=torch.bool)))
        pix_candidates = perm[is_cross_all[perm]][:pixel_micro]
        rest = perm[~torch.isin(perm, pix_candidates)]
        order = torch.cat((pix_candidates, rest))

        all_idx = all_idx[order]
        all_l = all_l[order]
        all_tc = all_tc[order]
        is_c_flag = is_c_flag[order]

        pg = torch.Generator(device=device).manual_seed(seed + step)
        proj = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=pg), mode="reduced").Q
                     for c in (64, 128, 256))

        def _predict_slice(sl: slice):
            b_idx = all_idx[sl]
            b_l = all_l[sl].to(device)
            b_tc = all_tc[sl].to(device)
            b_isc = is_c_flag[sl]

            if not b_isc.any():
                s_i = pairs["src"][b_idx]
                zs = tr["z"][s_i].to(device).float()
                c64b = tr["c_64"][s_i].to(device).float()
                q32b = c21["train"]["q32"][s_i].to(device).float()
                tidb = pairs["tid"][b_idx].to(device)
                tgt = _compute_targets(
                    zs, pair_hic[b_idx].to(device).float(),
                    pair_mid[b_idx].to(device).float() if pair_mid is not None else None,
                    pair_his[b_idx].to(device).float(), b_l, b_tc, interp_mode,
                )
                id_mask = pairs["same"][b_idx].to(device)[:, None, None, None]
                tgt = torch.where(id_mask, zs, tgt)
            elif b_isc.all():
                zs = pool_c["z_syn"][b_idx].to(device).float()
                c64b = pool_c["c_64"][b_idx].to(device).float()
                q32b = pool_c["q_32"][b_idx].to(device).float()
                tidb = pool_c["tid"][b_idx].to(device)
                tgt = pool_c[pool_c_field][b_idx].to(device).float()
            else:
                # Mixed micro-batch: split per item cleanly
                zs_l, c_l, q_l, t_l, tg_l = [], [], [], [], []
                for k_i in range(len(b_idx)):
                    bi = b_idx[k_i:k_i + 1]
                    if b_isc[k_i]:
                        zs_l.append(pool_c["z_syn"][bi].to(device).float())
                        c_l.append(pool_c["c_64"][bi].to(device).float())
                        q_l.append(pool_c["q_32"][bi].to(device).float())
                        t_l.append(pool_c["tid"][bi].to(device))
                        tg_l.append(pool_c[pool_c_field][bi].to(device).float())
                    else:
                        si_k = pairs["src"][bi]
                        z_k = tr["z"][si_k].to(device).float()
                        zs_l.append(z_k)
                        c_l.append(tr["c_64"][si_k].to(device).float())
                        q_l.append(c21["train"]["q32"][si_k].to(device).float())
                        t_l.append(pairs["tid"][bi].to(device))
                        tg_k = _compute_targets(
                            z_k, pair_hic[bi].to(device).float(),
                            pair_mid[bi].to(device).float() if pair_mid is not None else None,
                            pair_his[bi].to(device).float(), b_l[k_i:k_i + 1], b_tc[k_i:k_i + 1], interp_mode,
                        )
                        if bool(pairs["same"][bi]):
                            tg_k = z_k
                        tg_l.append(tg_k)
                zs = torch.cat(zs_l, 0)
                c64b = torch.cat(c_l, 0)
                q32b = torch.cat(q_l, 0)
                tidb = torch.cat(t_l, 0)
                tgt = torch.cat(tg_l, 0)

            if arch == "unet":
                pred = model(zs, c64b, q32b, tidb, b_l, raw_stats, priors, noise=torch.randn_like(zs))
            else:
                pred = model(zs, c64b, q32b, tidb, b_l, raw_stats, priors)
            return pred, tgt, b_l

        total = torch.zeros(2, device=device)
        # Pixel perceptual slice: base mode runs 2*pixel_micro on even steps; calibrated/hiswd run pixel_micro every step!
        run_pix_now = (loss_mode in ("calibrated", "hiswd")) or (step % 2 == 0)
        pix_n = pixel_micro if loss_mode in ("calibrated", "hiswd") else (2 * pixel_micro if run_pix_now else 0)
        if pix_n > 0:
            for st in range(0, pix_n, pixel_micro):
                sl = slice(st, min(pix_n, st + pixel_micro))
                pred, tgt, lv = _predict_slice(sl)
                _, l_lat = _latent_loss_terms(pred, tgt, lv, priors, loss_mode)
                l_lat = l_lat * (len(lv) / float(eff_batch))
                if loss_mode == "hiswd":
                    swd_sample_w = torch.where(lv > 0.6, torch.full_like(lv, 2.2), torch.ones_like(lv))
                    l_pix = pixel.loss(pred, tgt, proj, swd_weight=0.35, sample_weights=swd_sample_w)
                else:
                    l_pix = pixel.loss(pred, tgt, proj)
                l_pix = l_pix * (pixel_scale * (len(lv) / float(pix_n)))
                (l_lat + l_pix).backward()
                total[0] += l_lat.detach()
                total[1] += l_pix.detach()

        for st in range(pix_n, eff_batch, micro):
            stop = min(eff_batch, st + micro)
            pred, tgt, lv = _predict_slice(slice(st, stop))
            _, l_lat = _latent_loss_terms(pred, tgt, lv, priors, loss_mode)
            l_lat = l_lat * ((stop - st) / float(eff_batch))
            l_lat.backward()
            total[0] += l_lat.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in model.parameters()], ema_decay)

        if step == first or step % 50 == 0 or step == target_steps:
            dt_step = (time.monotonic() - t0) / max(1, step - first + 1)
            print(f"[{arm_name}] {step:4d}/{target_steps} lat={float(total[0]):.4f} pix={float(total[1]):.4f} "
                  f"grad={float(grad_norm):.3f} ({dt_step:.3f} s/step)", flush=True)

        if smoke_steps > 0:
            continue

        if step % 100 == 0 or step == updates:
            atomic_save({
                "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "sampling": gen.get_state(), "updates": updates, "step": step,
            }, resume_p)
        if (eval_every > 0 and step % eval_every == 0 and step < updates):
            atomic_save({"ema": ema.state_dict(), "raw_stats": raw_stats, "priors": priors,
                         "step": step, "params": num_params, "arch": arch}, run_dir / f"ema_s{step:04d}.pt")

    vram_gb = assert_vram_budget(device=device, tag=arm_name)
    elapsed = time.monotonic() - t0
    if smoke_steps > 0:
        sec_per_step = elapsed / float(smoke_steps)
        smoke_rep = {
            "arm": arm_name, "arch": arch, "params": num_params, "smoke_steps": smoke_steps,
            "eff_batch": eff_batch, "micro": micro, "pixel_micro": pixel_micro,
            "lr": lr, "weight_decay": weight_decay, "ema_decay": ema_decay,
            "elapsed_s": round(elapsed, 2), "sec_per_step": round(sec_per_step, 4),
            "est_screen_min": round(sec_per_step * updates / 60.0, 1),
            "peak_vram_gb": round(vram_gb, 3), "pixel_scale": round(pixel_scale, 4),
        }
        save_json(run_dir / "smoke.json", smoke_rep)
        print(f"[SMOKE {arm_name}] {smoke_rep}", flush=True)
        del pixel, model, opt, ema
        gc.collect()
        torch.cuda.empty_cache()
        return smoke_rep

    atomic_save({
        "ema": ema.state_dict(), "raw_stats": raw_stats, "priors": priors,
        "step": updates, "params": num_params, "arch": arch, "seed": seed,
    }, final_ckpt)
    del pixel, model, opt
    gc.collect()
    torch.cuda.empty_cache()

    rep = run_final_eval(ema, arch, raw_stats, priors, ts, c21["test750"]["q32"], updates,
                         run_dir=run_dir, tag=arm_name, checkpoint_name=final_ckpt.name, device=device)
    rep["peak_vram_gb"] = round(vram_gb, 3)
    rep["train_elapsed_min"] = round(elapsed / 60.0, 2)
    save_json(run_dir / "results.json", rep)
    del ema
    gc.collect()
    torch.cuda.empty_cache()
    return rep
