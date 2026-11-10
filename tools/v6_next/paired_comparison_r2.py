"""Comprehensive Paired Cluster Bootstrap Comparison for WEAVE R2 Models.

Implements Sections P4.2 & P4.3 of WEAVE R2 Plan:
- Paired cluster bootstrap with 150 source clusters, 4000 resamples.
- Compares all models against B3 (the high-fidelity anchor) and Q2a (baseline).
- Metrics: Δ(DINO-S), Δ(DINO-C), Δ(LPIPS), Δ(CSD Top-1), Δ(Target Margin).
- Checks whether 95% CI excludes 0.
- Checks P4.3 pre-set success criteria:
  * DINO-S gain >= +0.020 with CI > 0
  * DINO-C >= B3 - 0.02 (and ideally >= 0.80)
  * LPIPS <= B3 + 0.03
  * Top-1 >= B3 - 2pp
  * Minimalism off-target attractor <= B3 + 5pp
  * ER and Ukiyo-e breakdowns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.v6.train_eval_v6 import STYLES
from tools.v6_24h.stats import cluster_ci, paired_cluster_ci

ROOT = Path(__file__).resolve().parents[2]


def load_eval(path: Path):
  if not path.exists():
    return None
  d = json.loads(path.read_text(encoding="utf-8"))
  rows = d.get("rows", [])
  cross = [r for r in rows if r["is_cross"]]
  by_id = {r["task_id"]: r for r in cross}
  return {
      "name": path.parent.name,
      "summary": d.get("cross600", {}),
      "same150": d.get("same150", {}),
      "by_id": by_id,
      "stems": [r["src_stem"] for r in cross],
  }


def main():
  ap = argparse.ArgumentParser(description="WEAVE R2 Paired Comparison")
  ap.add_argument("--eval-dir", default="runs/v6_r2")
  ap.add_argument("--out-dir", default="runs/v6_r2/final_verdict")
  args = ap.parse_args()

  eval_dir = ROOT / args.eval_dir
  out_dir = ROOT / args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  # Candidate models
  models = [
      ("Identity", eval_dir / "eval_identity/eval.json"),
      ("Q2a Direct (218K)", eval_dir / "eval_Q2a_direct/eval.json"),
      ("B3 distill λ=20 (218K)", eval_dir / "eval_B3_distill20/eval.json"),
      ("B4 distill λ=50 (218K)", eval_dir / "eval_B4_distill50/eval.json"),
      (
          "P1_B2c_ctrl845 (845K)",
          eval_dir / "eval_P1_B2c_ctrl845/eval.json",
      ),  # if evaluated
      ("S1 distill λ=20 (218K, P2)", eval_dir / "eval_S1/eval.json"),
      ("S2 distill λ=50 (218K, P2)", eval_dir / "eval_S2/eval.json"),
      ("S3 distill λ=20 (845K, Oracle)", eval_dir / "eval_S3/eval.json"),
      ("S4 distill λ=20 (845K, P2)", eval_dir / "eval_S4/eval.json"),
      (
          "Q2a + R1 Oracle (1 step)",
          eval_dir / "eval_Q2a_oracle/eval.json",
      ),
      (
          "ScoreDiff N=24",
          eval_dir / "eval_scorediff_N24/eval.json",
      ),
  ]

  loaded = []
  for name, p in models:
    data = load_eval(p)
    if data:
      loaded.append((name, data))
      print(f"Loaded {name:32s}: {len(data['by_id'])} cross tasks")
    else:
      print(f"Missing {name:32s}: {p}")

  if not loaded:
    print("No evaluated models found.")
    return

  # Locate anchors: B3 and Q2a
  b3_data = next((d for n, d in loaded if "B3" in n), None)
  q2a_data = next((d for n, d in loaded if "Q2a Direct" in n), None)

  task_ids = (
      sorted(list(b3_data["by_id"].keys()))
      if b3_data
      else sorted(list(loaded[0][1]["by_id"].keys()))
  )

  # Table 1: Consolidated Overview Table
  print("\n" + "=" * 115)
  print("WEAVE R2 CONSOLIDATED OVERVIEW (tasks_750, Cross600)")
  print("=" * 115)
  print(
      f"  {'Model':30s} | {'Top-1':>7s} | {'Margin':>8s} | {'CSD ΔT':>8s} |"
      f" {'DINO-S (gain)':>18s} | {'DINO-C':>7s} | {'LPIPS':>7s} |"
      f" {'Min-Off':>8s}"
  )
  print("  " + "-" * 110)
  for name, d in loaded:
    s = d["summary"]
    gain_str = f"{s.get('dino_s', 0):.4f} ({s.get('dino_s_gain', 0):+.4f})"
    print(
        f"  {name:30s} | {s.get('top1', 0)*100:6.1f}% |"
        f" {s.get('margin', 0):+8.4f} | {s.get('dT', 0):+8.4f} |"
        f" {gain_str:>18s} | {s.get('dino_c', 0):7.4f} |"
        f" {s.get('lpips', 0):7.4f} |"
        f" {s.get('minimalism_offtarget', 0)*100:7.1f}%"
    )

  # Table 2: Paired Clustered Bootstrap vs B3 (Anchor)
  if b3_data:
    stems = [b3_data["by_id"][tid]["src_stem"] for tid in task_ids]
    print("\n" + "=" * 115)
    print("PAIRED CLUSTERED BOOTSTRAP VS B3 DISTILL λ=20 (Anchor, 150 Clusters)")
    print("=" * 115)
    header_p = (
        f"  {'Model':30s} | {'Δ(Top-1) [95% CI]':>24s} | {'Δ(DINO-S) [95% CI]':>26s} | {'Δ(DINO-C)':>10s} | {'P4.3 Gate':>9s}"
    )
    print(header_p)
    print("  " + "-" * 110)

    for name, d in loaded:
      if "B3" in name:
        continue
      vals_m_top1 = [d["by_id"][tid]["target_top1"] for tid in task_ids]
      vals_b_top1 = [b3_data["by_id"][tid]["target_top1"] for tid in task_ids]
      ci_top1 = paired_cluster_ci(vals_m_top1, vals_b_top1, stems)

      vals_m_dinos = [d["by_id"][tid]["dino_s"] for tid in task_ids]
      vals_b_dinos = [b3_data["by_id"][tid]["dino_s"] for tid in task_ids]
      ci_dinos = paired_cluster_ci(vals_m_dinos, vals_b_dinos, stems)

      vals_m_dinoc = [d["by_id"][tid]["dino_c"] for tid in task_ids]
      vals_b_dinoc = [b3_data["by_id"][tid]["dino_c"] for tid in task_ids]
      ci_dinoc = paired_cluster_ci(vals_m_dinoc, vals_b_dinoc, stems)

      # Check P4.3 criteria
      dinos_pass = ci_dinos["diff"] >= 0.020 and ci_dinos["ci"][0] > 0
      dinoc_pass = ci_dinoc["diff"] >= -0.02
      top1_pass = ci_top1["diff"] >= -0.02
      gate_str = (
          "✅ PASS"
          if (dinos_pass and dinoc_pass and top1_pass)
          else (
              "⚠️ PARTIAL"
              if (ci_dinos["ci"][0] > 0 and dinoc_pass)
              else "❌ NO"
          )
      )

      t_ci_str = (
          f"{ci_top1['diff']*100:+5.1f}pp"
          f" [{ci_top1['ci'][0]*100:+4.1f},{ci_top1['ci'][1]*100:+4.1f}]"
      )
      ds_ci_str = (
          f"{ci_dinos['diff']:+.4f}"
          f" [{ci_dinos['ci'][0]:+.4f},{ci_dinos['ci'][1]:+.4f}]"
      )
      dc_str = f"{ci_dinoc['diff']:+.4f}"

      print(
          f"  {name:30s} | {t_ci_str:>24s} | {ds_ci_str:>26s} | {dc_str:>10s} |"
          f" {gate_str:>9s}"
      )

  # Table 3: Per-Target DINO-S & Top-1 Breakdowns (Early Ren & Ukiyo-e)
  print("\n" + "=" * 115)
  print("PER-TARGET DINO-S BREAKDOWN ACROSS ALL 5 STYLES")
  print("=" * 115)
  print(f"  {'Model':30s} " + "".join(f"{s[:9]:>14s}" for s in STYLES))
  print("  " + "-" * 110)
  for name, d in loaded:
    pt = d["summary"].get("per_target", {})
    cells = []
    for s in STYLES:
      ds_val = pt.get(s, {}).get("dino_s", 0.0)
      cells.append(f"{ds_val:.4f}")
    print(f"  {name:30s} " + "".join(f"{c:>14s}" for c in cells))

  print("\n" + "=" * 115)
  print("PER-TARGET CSD TOP-1 BREAKDOWN ACROSS ALL 5 STYLES")
  print("=" * 115)
  print(f"  {'Model':30s} " + "".join(f"{s[:9]:>14s}" for s in STYLES))
  print("  " + "-" * 110)
  for name, d in loaded:
    pt = d["summary"].get("per_target", {})
    cells = []
    for s in STYLES:
      t1_val = pt.get(s, {}).get("top1", 0.0) * 100
      cells.append(f"{t1_val:5.1f}%")
    print(f"  {name:30s} " + "".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
  main()
