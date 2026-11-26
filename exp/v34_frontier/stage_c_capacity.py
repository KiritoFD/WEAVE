"""CLEAVE V34 Stage C (4h budget): 2x2 Joint Training & Capacity Expansion (22M/44M x Q32/V33-S2).

Disentangles backbone freeze, network capacity, and semantic control with identical
data, update count, and training objective across all four arms:
  - C0: 22M + Q32 (V32 control baseline)
  - C1: 22M + 7M V33 S2 joint training
  - C2: 44M + Q32 capacity expansion
  - C3: 44M + 7M V33 S2 joint training + capacity expansion
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
    evaluate_v34_arm,
    train_pcore_v34,
)
from src.utils.runtime import read_json, save_json, utc


STAGE_C_SPECS: dict[str, dict[str, Any]] = {
    "C0_22m_q32": {"base_dim": 144, "cond_dim": 320, "use_adapter": False, "label": "22M + Q32"},
    "C1_22m_s2_joint": {"base_dim": 144, "cond_dim": 320, "use_adapter": True, "label": "22M + 7M V33 S2 joint"},
    "C2_44m_q32": {"base_dim": 208, "cond_dim": 384, "use_adapter": False, "label": "44M + Q32"},
    "C3_44m_s2_joint": {"base_dim": 208, "cond_dim": 384, "use_adapter": True, "label": "44M + 7M V33 S2 joint"},
}


def summarize_stage_c(arm_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Perform 2x2 factorial decomposition at matched DINO-C (0.50, 0.61, 0.70) across C0..C3."""
    v32_pts = v32_frontier_points()
    c_grid = (0.50, 0.61, 0.70)

    def _curve(arm: str, metric: str) -> list[tuple[float, float]]:
        lv = arm_results[arm]["levels"]
        return sorted([(float(m["dino_c"]), float(m[metric])) for k, m in lv.items() if "wrong" not in k], key=lambda x: x[0])

    c0_s_pts = _curve("C0_22m_q32", "dino_s")
    c0_t_pts = _curve("C0_22m_q32", "csd_top1")
    c0_s085_eval = OUT / "measured/C0_22m_q32_s085/eval.json"

    arm_summary: dict[str, Any] = {}
    matched_s: dict[str, list[float]] = {}
    best_arm = "C0_22m_q32"
    best_gain = 0.0

    for arm_name, payload in arm_results.items():
        pts_s = _curve(arm_name, "dino_s")
        pts_t = _curve(arm_name, "csd_top1")
        s_at_c = [round(interp_s_at_c(c, pts_s), 4) for c in c_grid]
        t_at_c = [round(interp_s_at_c(c, pts_t), 4) for c in c_grid]
        ds_c0 = [round(s_at_c[i] - interp_s_at_c(c_grid[i], c0_s_pts), 4) for i in range(len(c_grid))]
        dt_c0 = [round(t_at_c[i] - interp_s_at_c(c_grid[i], c0_t_pts), 4) for i in range(len(c_grid))]
        ds_v32 = [round(s_at_c[i] - interp_s_at_c(c_grid[i], v32_pts), 4) for i in range(len(c_grid))]
        mean_ds_c0 = round(sum(ds_c0) / len(ds_c0), 4)
        mean_dt_c0 = round(sum(dt_c0) / len(dt_c0), 4)
        matched_s[arm_name] = s_at_c

        ci_s085 = None
        arm_s085_eval = OUT / f"measured/{arm_name}_s085/eval.json"
        if arm_name != "C0_22m_q32" and c0_s085_eval.exists() and arm_s085_eval.exists():
            ci_s085 = paired_ci_vs_baseline(arm_s085_eval, c0_s085_eval, cross_only=True)

        arm_summary[arm_name] = {
            "spec": STAGE_C_SPECS[arm_name],
            "cfg": payload["cfg"],
            "c_grid": list(c_grid),
            "dino_s_at_c_grid": s_at_c,
            "csd_top1_at_c_grid": t_at_c,
            "delta_s_vs_C0": ds_c0,
            "delta_top1_vs_C0": dt_c0,
            "delta_s_vs_V32": ds_v32,
            "mean_delta_s_vs_C0": mean_ds_c0,
            "mean_delta_top1_vs_C0": mean_dt_c0,
            "ci_s085_vs_C0": ci_s085,
        }
        # Select winner by highest positive matched-C frontier gain (+ top-1 tiebreaker)
        score = mean_ds_c0 + 0.1 * mean_dt_c0
        if arm_name != "C0_22m_q32" and score > best_gain:
            best_gain = score
            best_arm = arm_name

    # Exact 2x2 factorial effects on matched DINO-S
    eff_s2_on_22m = [round(matched_s["C1_22m_s2_joint"][i] - matched_s["C0_22m_q32"][i], 4) for i in range(len(c_grid))]
    eff_cap_on_q32 = [round(matched_s["C2_44m_q32"][i] - matched_s["C0_22m_q32"][i], 4) for i in range(len(c_grid))]
    eff_s2_on_44m = [round(matched_s["C3_44m_s2_joint"][i] - matched_s["C2_44m_q32"][i], 4) for i in range(len(c_grid))]
    eff_cap_on_s2 = [round(matched_s["C3_44m_s2_joint"][i] - matched_s["C1_22m_s2_joint"][i], 4) for i in range(len(c_grid))]
    interaction = [round(eff_s2_on_44m[i] - eff_s2_on_22m[i], 4) for i in range(len(c_grid))]

    return {
        "updated_utc": utc(),
        "stage": "C_capacity_and_joint_training",
        "winner_arm": best_arm,
        "winner_spec": STAGE_C_SPECS[best_arm],
        "winner_effective": bool(arm_summary[best_arm]["mean_delta_s_vs_C0"] >= 0.002 or arm_summary[best_arm]["mean_delta_top1_vs_C0"] >= 0.01),
        "factorial_effects_at_C50_61_70": {
            "c_grid": list(c_grid),
            "semantic_control_on_22M (C1-C0)": eff_s2_on_22m,
            "capacity_on_Q32 (C2-C0)": eff_cap_on_q32,
            "semantic_control_on_44M (C3-C2)": eff_s2_on_44m,
            "capacity_on_S2 (C3-C1)": eff_cap_on_s2,
            "interaction ((C3-C2)-(C1-C0))": interaction,
        },
        "arm_summary": arm_summary,
        "arms": arm_results,
    }


def run_stage_c(
    *,
    updates: int = 3500,
    batch_size: int = 128,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train & evaluate all 4 Stage C arms (C0, C1, C2, C3) with identical data, updates, and loss."""
    rep_p = OUT / "stage_c_report.json"
    if rep_p.exists() and smoke_steps == 0:
        print(f"[Stage C] Reusing completed report {rep_p}", flush=True)
        return read_json(rep_p)

    arm_results: dict[str, dict[str, Any]] = {}
    for arm_name, spec in STAGE_C_SPECS.items():
        arm_dir = OUT / "stage_c" / arm_name
        arm_rep_p = arm_dir / "arm_report.json"
        if arm_rep_p.exists() and smoke_steps == 0:
            arm_results[arm_name] = read_json(arm_rep_p)
            print(f"[Stage C] Reusing {arm_name} from {arm_rep_p}", flush=True)
            continue

        tr_info = train_pcore_v34(
            arm_name,
            arm_dir,
            base_dim=int(spec["base_dim"]),
            cond_dim=int(spec["cond_dim"]),
            style_mode="k8",
            use_adapter=bool(spec["use_adapter"]),
            updates=updates,
            lr=3.0e-4,
            batch_size=batch_size,
            seed_offset=200,
            smoke_steps=smoke_steps,
            device=device,
        )
        if smoke_steps > 0:
            arm_results[arm_name] = tr_info
            continue

        levels = evaluate_v34_arm(
            arm_name,
            Path(tr_info["ckpt"]),
            knob_list=(0.75, 0.85, 1.00, 1.25, 1.50),
            include_tau0=True,
            eval_wrong=True,
            device=device,
        )
        payload = {**tr_info, "levels": levels}
        save_json(arm_rep_p, payload)
        arm_results[arm_name] = payload

    if smoke_steps > 0:
        return {"stage": "C_smoke", "arms": arm_results}

    rep = summarize_stage_c(arm_results)
    save_json(rep_p, rep)
    w = rep["winner_arm"]
    ws = rep["arm_summary"][w]
    print(
        f"[Stage C Done] Winner={w} | delta_S_vs_C0={ws['mean_delta_s_vs_C0']:+.4f} "
        f"| delta_Top1_vs_C0={ws['mean_delta_top1_vs_C0']*100:+.2f}pp | effective={rep['winner_effective']}",
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
    run_stage_c(updates=args.updates, batch_size=args.batch_size, smoke_steps=args.smoke_steps, device=args.device)


if __name__ == "__main__":
    main()
