"""CLEAVE V34 Stage B (2h budget): Continuous Style Conditioning vs K=8 Discrete Centroids.

Breaks the 40 discrete K=8 cluster center bottleneck with instance-level training statistics
and continuous scene-matched convex combinations of training instances at inference:
  - B0: K=8 discrete centroids baseline (417-D)
  - B1: Instance-level 417-D statistics + continuous convex mixup
  - B2: Instance-level 417-D + 684-D multi-scale ConvNeXt+Haar texture distribution (1101-D)

No RCSM, no reference Value injection, no extra local matching loss.
Includes causal diversity probe holding (content, noise, Q32, E-Core draft) fixed while varying u_style.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from exp.v34_frontier import (
    OUT,
    interp_s_at_c,
    paired_ci_vs_baseline,
    v32_frontier_points,
)
from exp.v34_frontier.train_common import (
    build_or_load_style_instances,
    evaluate_v34_arm,
    measure_style_diversity,
    train_pcore_v34,
)
from src.utils.runtime import read_json, save_json, utc


STAGE_B_ARMS: dict[str, str] = {
    "B0_k8_discrete": "k8",
    "B1_inst_417": "inst_417",
    "B2_inst_tex1101": "inst_1101",
}


def summarize_stage_b(arm_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare B0/B1/B2 on diversity responsiveness and matched-C (DINO-C, DINO-S) frontier."""
    v32_pts = v32_frontier_points()
    b0_lv = arm_results["B0_k8_discrete"]["levels"]
    b0_pts = sorted(
        [(float(m["dino_c"]), float(m["dino_s"])) for k, m in b0_lv.items() if "wrong" not in k],
        key=lambda x: x[0],
    )
    b0_top1_pts = sorted(
        [(float(m["dino_c"]), float(m["csd_top1"])) for k, m in b0_lv.items() if "wrong" not in k],
        key=lambda x: x[0],
    )
    b0_s085_eval = OUT / "measured/B0_k8_discrete_s085/eval.json"

    comparisons: dict[str, Any] = {}
    best_name = "B0_k8_discrete"
    best_gain = 0.0

    for arm_name, payload in arm_results.items():
        lv = payload["levels"]
        div = payload["diversity"]
        c_targets = (0.50, 0.61)
        pts_s = sorted([(float(m["dino_c"]), float(m["dino_s"])) for k, m in lv.items() if "wrong" not in k], key=lambda x: x[0])
        pts_t = sorted([(float(m["dino_c"]), float(m["csd_top1"])) for k, m in lv.items() if "wrong" not in k], key=lambda x: x[0])
        delta_s_b0 = [round(interp_s_at_c(c, pts_s) - interp_s_at_c(c, b0_pts), 4) for c in c_targets]
        delta_top1_b0 = [round(interp_s_at_c(c, pts_t) - interp_s_at_c(c, b0_top1_pts), 4) for c in c_targets]
        delta_s_v32 = [round(interp_s_at_c(c, pts_s) - interp_s_at_c(c, v32_pts), 4) for c in c_targets]
        mean_ds_b0 = round(sum(delta_s_b0) / len(delta_s_b0), 4)

        ci_s085 = None
        arm_s085_eval = OUT / f"measured/{arm_name}_s085/eval.json"
        if arm_name != "B0_k8_discrete" and b0_s085_eval.exists() and arm_s085_eval.exists():
            ci_s085 = paired_ci_vs_baseline(arm_s085_eval, b0_s085_eval, cross_only=True)

        comparisons[arm_name] = {
            "style_mode": STAGE_B_ARMS[arm_name],
            "mean_delta_dino_s_vs_B0_at_C50_61": mean_ds_b0,
            "delta_dino_s_vs_B0_at_C50_C61": delta_s_b0,
            "delta_top1_vs_B0_at_C50_C61": delta_top1_b0,
            "delta_dino_s_vs_V32_at_C50_C61": delta_s_v32,
            "diversity": div,
            "ci_s085_vs_B0": ci_s085,
        }
        if arm_name != "B0_k8_discrete" and mean_ds_b0 > best_gain:
            best_gain = mean_ds_b0
            best_name = arm_name

    b0_div = float(arm_results["B0_k8_discrete"]["diversity"]["pix_l1_uA_vs_uB"])
    b1_div = float(arm_results["B1_inst_417"]["diversity"]["pix_l1_uA_vs_uB"])
    b2_div = float(arm_results["B2_inst_tex1101"]["diversity"]["pix_l1_uA_vs_uB"])
    best_div = max(b1_div, b2_div)

    return {
        "updated_utc": utc(),
        "stage": "B_continuous_style_cond",
        "winner_arm": best_name,
        "winner_style_mode": STAGE_B_ARMS[best_name],
        "best_mean_delta_dino_s_vs_B0": round(best_gain, 4),
        "frontier_verified_effective": bool(best_gain >= 0.002),
        "diversity_verified_responsive": bool(best_div >= max(0.018, b0_div)),
        "comparisons": comparisons,
        "arms": arm_results,
    }


def run_stage_b(
    *,
    updates: int = 3500,
    batch_size: int = 128,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run all Stage B arms (B0, B1, B2) -> diversity probe -> test750 eval -> report."""
    rep_p = OUT / "stage_b_report.json"
    if rep_p.exists() and smoke_steps == 0:
        print(f"[Stage B] Reusing completed report {rep_p}", flush=True)
        return read_json(rep_p)

    build_or_load_style_instances(device=device)

    arm_results: dict[str, dict[str, Any]] = {}
    for idx, (arm_name, style_mode) in enumerate(STAGE_B_ARMS.items()):
        arm_dir = OUT / "stage_b" / arm_name
        arm_rep_p = arm_dir / "arm_report.json"
        if arm_rep_p.exists() and smoke_steps == 0:
            arm_results[arm_name] = read_json(arm_rep_p)
            print(f"[Stage B] Reusing {arm_name} from {arm_rep_p}", flush=True)
            continue

        tr_info = train_pcore_v34(
            arm_name,
            arm_dir,
            base_dim=144,
            cond_dim=320,
            style_mode=style_mode,
            use_adapter=False,
            updates=updates,
            lr=3.0e-4,
            batch_size=batch_size,
            seed_offset=100,
            smoke_steps=smoke_steps,
            device=device,
        )
        if smoke_steps > 0:
            arm_results[arm_name] = tr_info
            continue

        ckpt_p = Path(tr_info["ckpt"])
        div = measure_style_diversity(arm_name, ckpt_p, n_tasks=50, device=device)
        levels = evaluate_v34_arm(
            arm_name,
            ckpt_p,
            knob_list=(0.75, 0.85, 1.00, 1.25, 1.50),
            include_tau0=False,
            eval_wrong=True,
            device=device,
        )
        payload = {**tr_info, "diversity": div, "levels": levels}
        save_json(arm_rep_p, payload)
        arm_results[arm_name] = payload

    if smoke_steps > 0:
        return {"stage": "B_smoke", "arms": arm_results}

    rep = summarize_stage_b(arm_results)
    save_json(rep_p, rep)
    print(
        f"[Stage B Done] Winner={rep['winner_arm']} | delta_S_vs_B0={rep['best_mean_delta_dino_s_vs_B0']:+.4f} "
        f"| frontier_ok={rep['frontier_verified_effective']} | diversity_ok={rep['diversity_verified_responsive']}",
        flush=True,
    )
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=3500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_stage_b(updates=args.updates, batch_size=args.batch_size, smoke_steps=args.smoke_steps, device=args.device)


if __name__ == "__main__":
    main()
