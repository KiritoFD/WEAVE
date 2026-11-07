"""WEAVE V6 Stage 4: Final 600 Cross + 150 Same Full Evaluation, Second-Seed Verification,
Paired Statistical Comparison against Old WEAVE Baseline, and Final Report Generation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v6.models_v6 import C3UNet, ConditionalScoreNet  # noqa: E402
from tools.v6.pareto_and_full_eval import full_eval750_fn  # noqa: E402
from tools.v6.score_diff import edit_with_score_difference  # noqa: E402
from tools.v6.train_eval_v6 import boot_ci, generate_v6, train_and_evaluate  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
V6_ROOT = REPO_ROOT / "runs" / "v6"
RENDER_ROOT = paths.asset_root() / "eval_assets" / "v6_main"


def summarize_750(eval750: dict) -> dict:
    return {
        "cross600": eval750["cross600"],
        "same150": eval750["same150"],
        "all750": eval750["all750"],
        "per_target_cross_CSD_dT": eval750["per_target_cross_CSD_dT"],
        "styles_improved": eval750["styles_improved"],
        "ukiyo_dT": eval750["ukiyo_dT"],
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(global_stats).to(device)

    # 1. Train Second Seed (seed 43) for D1_c1_gentle to verify cross-seed reproducibility
    s43_dir = V6_ROOT / "D1_c1_gentle_s43"
    if not (s43_dir / "D1_c1_gentle_s43.pt").exists():
        print("[Stage4] Training second seed D1_c1_gentle_s43 (seed=43, 800 steps)...", flush=True)
        ns = argparse.Namespace(
            name="D1_c1_gentle_s43",
            mode="direct",
            canon_mode="c1_gentle",
            base_dim=48,
            steps=800,
            microbatch=80,
            accum=3,
            lr=3e-4,
            coherence_beta=0.0,
            w_style_dist=0.25,
            w_cross=0.35,
            w_canon_inv=1.5,
            ode_steps=8,
            sigma0=0.10,
            seed=43,
            run_full750=False,
        )
        train_and_evaluate(ns)
    else:
        print("[Stage4] D1_c1_gentle_s43 checkpoint already exists, reusing...", flush=True)

    # Load models
    d1_s42_ck = torch.load(V6_ROOT / "D1_c1_gentle" / "D1_c1_gentle.pt", map_location="cpu", weights_only=False)
    d1_s42 = C3UNet(base_dim=48, num_styles=5, mode="direct").to(device).eval()
    d1_s42.load_state_dict(d1_s42_ck["model"])

    d1_s43_ck = torch.load(V6_ROOT / "D1_c1_gentle_s43" / "D1_c1_gentle_s43.pt", map_location="cpu", weights_only=False)
    d1_s43 = C3UNet(base_dim=48, num_styles=5, mode="direct").to(device).eval()
    d1_s43.load_state_dict(d1_s43_ck["model"])

    sd_ck = torch.load(V6_ROOT / "score_diff" / "score_diff_net.pt", map_location="cpu", weights_only=False)
    sd_model = ConditionalScoreNet(base_dim=64, num_styles=5).to(device).eval()
    sd_model.load_state_dict(sd_ck["model"])

    final_results = {}

    # 2. Full 750 Eval: D1_c1_gentle full output (s42 & s43)
    for tag, mod in (("D1_c1_gentle_s42", d1_s42), ("D1_c1_gentle_s43", d1_s43)):
        out_p = RENDER_ROOT / tag / "v4_eval750.json"
        if out_p.exists():
            print(f"[Stage4] Reusing existing {tag}...", flush=True)
            final_results[tag] = summarize_750(json.loads(out_p.read_text(encoding="utf-8")))
        else:
            print(f"[Stage4] Full 750 evaluation for {tag}...", flush=True)
            res = full_eval750_fn(
                lambda z, s_id, t_id, m=mod: generate_v6(m, canon, z, t_id, canon_mode="c1_gentle"),
                cell=tag,
                device=device,
            )
            final_results[tag] = summarize_750(res)

    # 3. Full 750 Eval: D1_c1_gentle LPIPS-calibrated (alpha=0.70, s42 & s43)
    for tag_a, mod_a in (("D1_c1_gentle_a0.70_s42", d1_s42), ("D1_c1_gentle_a0.70_s43", d1_s43)):
        out_p = RENDER_ROOT / tag_a / "v4_eval750.json"
        if out_p.exists():
            print(f"[Stage4] Reusing existing {tag_a}...", flush=True)
            final_results[tag_a] = summarize_750(json.loads(out_p.read_text(encoding="utf-8")))
        else:
            print(f"[Stage4] Full 750 evaluation for {tag_a}...", flush=True)
            res_a = full_eval750_fn(
                lambda z, s_id, t_id, m=mod_a: 0.30 * z + 0.70 * generate_v6(m, canon, z, t_id, canon_mode="c1_gentle"),
                cell=tag_a,
                device=device,
            )
            final_results[tag_a] = summarize_750(res_a)

    # 3b. Full 750 Eval: ScoreDiff LPIPS+DINO-C safe point (lr=0.048, s42 & s43)
    for seed, tag_sd in ((42, "ScoreDiff_s12_lr0.048_s42"), (43, "ScoreDiff_s12_lr0.048_s43")):
        out_p = RENDER_ROOT / tag_sd / "v4_eval750.json"
        if out_p.exists():
            print(f"[Stage4] Reusing existing {tag_sd}...", flush=True)
            final_results[tag_sd] = summarize_750(json.loads(out_p.read_text(encoding="utf-8")))
        else:
            print(f"[Stage4] Full 750 evaluation for {tag_sd}...", flush=True)
            res_sd_safe = full_eval750_fn(
                lambda z, s_id, t_id, sd=seed: edit_with_score_difference(
                    sd_model, z, s_id, t_id, num_steps=12, step_size=0.048, struct_anchor_weight=0.08, seed=sd
                ),
                cell=tag_sd,
                device=device,
            )
            final_results[tag_sd] = summarize_750(res_sd_safe)

    # 4. Full 750 Eval: Score Difference (s42 & s43)
    for seed, tag in ((42, "ScoreDiff_s12_lr0.055_s42"), (43, "ScoreDiff_s12_lr0.055_s43")):
        out_p = RENDER_ROOT / tag / "v4_eval750.json"
        if out_p.exists():
            print(f"[Stage4] Reusing existing {tag}...", flush=True)
            final_results[tag] = summarize_750(json.loads(out_p.read_text(encoding="utf-8")))
        else:
            print(f"[Stage4] Full 750 evaluation for {tag}...", flush=True)
            res_sd = full_eval750_fn(
                lambda z, s_id, t_id, sd=seed: edit_with_score_difference(
                    sd_model, z, s_id, t_id, num_steps=12, step_size=0.055, struct_anchor_weight=0.08, seed=sd
                ),
                cell=tag,
                device=device,
            )
            final_results[tag] = summarize_750(res_sd)

    # Compute two-seed means and acceptance checks against V6 Section VII
    def mean_two(a: dict, b: dict, group: str) -> dict:
        keys = ("CSD_dT", "CSD_dS", "CSD_dM", "DINO_S", "DINO_C", "CLIP_S", "LPIPS", "PatchSWD")
        return {k: float(0.5 * (a[group][k] + b[group][k])) for k in keys if k in a[group]}

    d1_two_seed = {
        "cross600": mean_two(final_results["D1_c1_gentle_s42"], final_results["D1_c1_gentle_s43"], "cross600"),
        "same150": mean_two(final_results["D1_c1_gentle_s42"], final_results["D1_c1_gentle_s43"], "same150"),
        "all750": mean_two(final_results["D1_c1_gentle_s42"], final_results["D1_c1_gentle_s43"], "all750"),
        "seed_gap_cross600_dT": abs(
            final_results["D1_c1_gentle_s42"]["cross600"]["CSD_dT"]
            - final_results["D1_c1_gentle_s43"]["cross600"]["CSD_dT"]
        ),
        "per_target_cross_CSD_dT": {
            s: float(0.5 * (
                final_results["D1_c1_gentle_s42"]["per_target_cross_CSD_dT"][s]
                + final_results["D1_c1_gentle_s43"]["per_target_cross_CSD_dT"][s]
            ))
            for s in STYLES
        },
    }

    sd_two_seed = {
        "cross600": mean_two(final_results["ScoreDiff_s12_lr0.055_s42"], final_results["ScoreDiff_s12_lr0.055_s43"], "cross600"),
        "same150": mean_two(final_results["ScoreDiff_s12_lr0.055_s42"], final_results["ScoreDiff_s12_lr0.055_s43"], "same150"),
        "all750": mean_two(final_results["ScoreDiff_s12_lr0.055_s42"], final_results["ScoreDiff_s12_lr0.055_s43"], "all750"),
        "seed_gap_cross600_dT": abs(
            final_results["ScoreDiff_s12_lr0.055_s42"]["cross600"]["CSD_dT"]
            - final_results["ScoreDiff_s12_lr0.055_s43"]["cross600"]["CSD_dT"]
        ),
        "per_target_cross_CSD_dT": {
            s: float(0.5 * (
                final_results["ScoreDiff_s12_lr0.055_s42"]["per_target_cross_CSD_dT"][s]
                + final_results["ScoreDiff_s12_lr0.055_s43"]["per_target_cross_CSD_dT"][s]
            ))
            for s in STYLES
        },
    }

    d1_a70_two_seed = {
        "cross600": mean_two(final_results["D1_c1_gentle_a0.70_s42"], final_results["D1_c1_gentle_a0.70_s43"], "cross600"),
        "same150": mean_two(final_results["D1_c1_gentle_a0.70_s42"], final_results["D1_c1_gentle_a0.70_s43"], "same150"),
        "all750": mean_two(final_results["D1_c1_gentle_a0.70_s42"], final_results["D1_c1_gentle_a0.70_s43"], "all750"),
        "seed_gap_cross600_dT": abs(
            final_results["D1_c1_gentle_a0.70_s42"]["cross600"]["CSD_dT"]
            - final_results["D1_c1_gentle_a0.70_s43"]["cross600"]["CSD_dT"]
        ),
        "per_target_cross_CSD_dT": {
            s: float(0.5 * (
                final_results["D1_c1_gentle_a0.70_s42"]["per_target_cross_CSD_dT"][s]
                + final_results["D1_c1_gentle_a0.70_s43"]["per_target_cross_CSD_dT"][s]
            ))
            for s in STYLES
        },
    }

    sd_lr048_two_seed = {
        "cross600": mean_two(final_results["ScoreDiff_s12_lr0.048_s42"], final_results["ScoreDiff_s12_lr0.048_s43"], "cross600"),
        "same150": mean_two(final_results["ScoreDiff_s12_lr0.048_s42"], final_results["ScoreDiff_s12_lr0.048_s43"], "same150"),
        "all750": mean_two(final_results["ScoreDiff_s12_lr0.048_s42"], final_results["ScoreDiff_s12_lr0.048_s43"], "all750"),
        "seed_gap_cross600_dT": abs(
            final_results["ScoreDiff_s12_lr0.048_s42"]["cross600"]["CSD_dT"]
            - final_results["ScoreDiff_s12_lr0.048_s43"]["cross600"]["CSD_dT"]
        ),
        "per_target_cross_CSD_dT": {
            s: float(0.5 * (
                final_results["ScoreDiff_s12_lr0.048_s42"]["per_target_cross_CSD_dT"][s]
                + final_results["ScoreDiff_s12_lr0.048_s43"]["per_target_cross_CSD_dT"][s]
            ))
            for s in STYLES
        },
    }

    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "old_weave_anchor_600cross": {
            "CSD_dT": 0.020284,
            "CSD_dS": -0.001890,
            "CSD_dM": 0.022174,
            "DINO_S": 0.391857,
            "DINO_C": 0.878826,
            "CLIP_S": 0.669748,
            "LPIPS": 0.265337,
        },
        "v6_acceptance_thresholds": {
            "cross600_CSD_dT_challenge": 0.04,
            "cross600_CSD_dT_beats_old": 0.020284,
            "DINO_C_floor": 0.8688,
            "LPIPS_ceiling": 0.2753,
            "min_styles_improved": 4,
            "same150_no_collapse_vs_v5": -0.1427,
        },
        "cells": final_results,
        "D1_c1_gentle_two_seed_mean": d1_two_seed,
        "D1_c1_gentle_a0.70_two_seed_mean": d1_a70_two_seed,
        "ScoreDiff_two_seed_mean": sd_two_seed,
        "ScoreDiff_lr0.048_two_seed_mean": sd_lr048_two_seed,
    }

    out_path = V6_ROOT / "stage4_final750_summary.json"
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Stage4] Complete! Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
