"""WEAVE W — paired clustered bootstrap comparison of W arms vs R2 anchors.

Reads unified r2_eval outputs from BOTH runs/v6_r2 (R2 anchors: Q2a, B3, B4,
Oracle) and runs/v6_w (new W arms). Reports:
  - consolidated Cross600 overview,
  - paired Δ(Top-1) / Δ(DINO-S) / Δ(DINO-C) with 95% CI vs B3 anchor,
  - P4.3 pre-set success gate,
  - per-target Top-1 / DINO-S breakdown (ER & Ukiyo-e highlighted).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.v6.train_eval_v6 import STYLES
from tools.v6_24h.stats import cluster_ci, paired_cluster_ci

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "runs/v6_r2"
W = ROOT / "runs/v6_w"

ANCHORS = [
    ("Q2a Direct (R2)", R2 / "eval_Q2a_direct/eval.json"),
    ("Q2a + R1 Oracle (R2)", R2 / "eval_Q2a_oracle/eval.json"),
    ("B3 λ=20 (R2 anchor)", R2 / "eval_B3_distill20/eval.json"),
    ("B4 λ=50 (R2)", R2 / "eval_B4_distill50/eval.json"),
    ("ScoreDiff N24 (R2)", R2 / "eval_scorediff_N24/eval.json"),
]
W_ARMS = [
    ("W1 FeatDistill 218K", W / "eval_W1_FeatDistill_218/eval.json"),
    ("W2 LossClean 218K", W / "eval_W2_LossClean_218/eval.json"),
    ("W3 LossClean 218K λ50", W / "eval_W3_LossClean_218_l50/eval.json"),
    ("W4 LossClean 845K+SK", W / "eval_W4_LossClean_845_SK/eval.json"),
    ("W5 S10 ImgCalib", W / "eval_W5_ImgCalib/eval.json"),
    ("V1 A0-pool LossClean", W / "eval_V1_A0_LossClean/eval.json"),
    ("V2 A0-pool PixelMSE", W / "eval_V2_A0_PixelMSE/eval.json"),
    ("R1 only", W / "eval_guided_r1/eval.json"),
    ("Score only", W / "eval_guided_score/eval.json"),
    ("R1 + Score", W / "eval_guided_both/eval.json"),
    ("P2 Gated Full", W / "eval_gated_full/eval.json"),
    ("P2 Gated HF", W / "eval_gated_hf/eval.json"),
    ("P2 Gated TopK", W / "eval_gated_topk/eval.json"),
    ("P2 Gated Content", W / "eval_gated_content/eval.json"),
    ("P2 Gated StyleEnergy", W / "eval_gated_style_energy/eval.json"),
    ("P2 TopK10", W / "eval_gated_topk10/eval.json"),
    ("P2 TopK50", W / "eval_gated_topk50/eval.json"),
    ("P2 TopK25 S8", W / "eval_gated_topk25_s8/eval.json"),
]


def load(p):
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    cross = [r for r in d["rows"] if r["is_cross"]]
    return {
        "summary": d.get("cross600", {}),
        "by_id": {r["task_id"]: r for r in cross},
        "stems": [r["src_stem"] for r in cross],
    }


def main():
    rows = []
    for name, p in ANCHORS + W_ARMS:
        d = load(p)
        if d:
            rows.append((name, d))
            print(f"Loaded {name:28s}: {len(d['by_id'])} cross tasks")
        else:
            print(f"Missing {name:28s}: {p}")

    if not rows:
        print("No evals found"); return

    print("\n" + "=" * 118)
    print("WEAVE W CONSOLIDATED OVERVIEW (tasks_750 Cross600)")
    print("=" * 118)
    hdr = (f"  {'Model':28s} | {'Top-1':>7s} | {'Margin':>8s} | {'CSD ΔT':>7s} | "
           f"{'DINO-S':>7s} | {'gain':>7s} | {'DINO-C':>7s} | {'LPIPS':>7s} | {'MinOff':>7s}")
    print(hdr); print("  " + "-" * 112)
    for name, d in rows:
        s = d["summary"]
        print(f"  {name:28s} | {s['top1']*100:6.1f}% | {s['margin']:+8.4f} | "
              f"{s['dT']:+7.4f} | {s['dino_s']:7.4f} | {s['dino_s_gain']:+7.4f} | "
              f"{s['dino_c']:7.4f} | {s['lpips']:7.4f} | {s['minimalism_offtarget']*100:6.1f}%")

    # anchor B3
    b3 = next((d for n, d in rows if "B3" in n), None)
    if not b3:
        print("B3 anchor missing"); return
    task_ids = sorted(b3["by_id"].keys())

    def paired(name, d, metric, stems):
        a = [d["by_id"][t][metric] for t in task_ids]
        b = [b3["by_id"][t][metric] for t in task_ids]
        return paired_cluster_ci(a, b, stems)

    stems = b3["stems"]
    print("\n" + "=" * 118)
    print("PAIRED CLUSTERED BOOTSTRAP vs B3 λ=20 ANCHOR (150 clusters, 4000 resamples)")
    print("=" * 118)
    print(f"  {'Model':28s} | {'ΔTop-1 [95% CI]':>24s} | {'ΔDINO-S [95% CI]':>26s} | "
          f"{'ΔDINO-C':>8s} | {'P4.3 Gate':>10s}")
    print("  " + "-" * 112)
    gate_results = {}
    paired_details = {}
    for name, d in rows:
        if "B3" in name or d["by_id"].keys() != b3["by_id"].keys():
            if "B3" not in name:
                print(f"  {name:28s} | task-id mismatch -> skip paired (different task set)")
            continue
        ci_t = paired(name, d, "target_top1", stems)
        ci_ds = paired(name, d, "dino_s", stems)
        ci_dc = paired(name, d, "dino_c", stems)
        ci_lp = paired(name, d, "lpips", stems)

        dinos_ok = ci_ds["diff"] >= 0.020 and ci_ds["ci"][0] > 0
        dinoc_ok = ci_dc["diff"] >= -0.02
        lpips_ok = ci_lp["diff"] <= 0.03
        top1_ok = ci_t["diff"] >= -0.02
        gate = "PASS" if (dinos_ok and dinoc_ok and lpips_ok and top1_ok) else (
            "PARTIAL" if ci_ds["ci"][0] > 0 else "NO")
        gate_results[name] = gate
        paired_details[name] = {
            "top1": ci_t,
            "dino_s": ci_ds,
            "dino_c": ci_dc,
            "lpips": ci_lp,
        }
        t_str = f"{ci_t['diff']*100:+5.1f}pp [{ci_t['ci'][0]*100:+4.1f},{ci_t['ci'][1]*100:+4.1f}]"
        ds_str = f"{ci_ds['diff']:+.4f} [{ci_ds['ci'][0]:+.4f},{ci_ds['ci'][1]:+.4f}]"
        print(f"  {name:28s} | {t_str:>24s} | {ds_str:>26s} | {ci_dc['diff']:+8.4f} | {gate:>10s}")

    print("\n" + "=" * 118)
    print("PER-TARGET CSD TOP-1 (each n=120)")
    print("=" * 118)
    print(f"  {'Model':28s} " + "".join(f"{s[:9]:>11s}" for s in STYLES))
    print("  " + "-" * 100)
    for name, d in rows:
        pt = d["summary"].get("per_target", {})
        print(f"  {name:28s} " + "".join(
            f"{pt.get(s, {}).get('top1', 0)*100:10.1f}%" for s in STYLES))

    print("\n" + "=" * 118)
    print("PER-TARGET DINO-S")
    print("=" * 118)
    print(f"  {'Model':28s} " + "".join(f"{s[:9]:>11s}" for s in STYLES))
    print("  " + "-" * 100)
    for name, d in rows:
        pt = d["summary"].get("per_target", {})
        print(f"  {name:28s} " + "".join(
            f"{pt.get(s, {}).get('dino_s', 0):11.4f}" for s in STYLES))

    # verdict json
    out = {"gates_vs_B3": gate_results,
           "p43_primary_gate_any_pass": any(v == "PASS" for v in gate_results.values()),
           "paired_details_vs_B3": paired_details}
    (W / "w_paired_verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {W / 'w_paired_verdict.json'} : {out}")


if __name__ == "__main__":
    main()
