"""Plan section 7 - pick the best arm per declared tier.

The plan asks for three presets (fidelity / balanced / style-first) and insists on Pareto
rather than a single number, because "best" depends on the declared content budget.

Each tier is a content budget; within it we maximise dT:
  fidelity  DINO-C >= 0.910  and LPIPS <= 0.370   (stay close to A0's content quality)
  balanced  DINO-C >= 0.900  and LPIPS <= 0.390
  style     DINO-C >= 0.880  and LPIPS <= 0.410   (allow visible change for style)

An arm that fails every tier is reported as such rather than silently dropped, and the
report states which arms are Pareto-dominated so a "best" claim cannot be made from a
single axis.

Usage
    python3 -m tools.v6_24h.tier_select
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TIERS = {
    "fidelity": {"dino_min": 0.910, "lpips_max": 0.370},
    "balanced": {"dino_min": 0.900, "lpips_max": 0.390},
    "style":    {"dino_min": 0.880, "lpips_max": 0.410},
}


def collect() -> dict:
    """All arms with an eval200, from the queue state plus the historical A0 report."""
    arms = {}
    a0 = ROOT / "runs/v6/D1_c1_gentle/report.json"
    if a0.exists():
        e = json.loads(a0.read_text())["eval200"]
        arms["A0"] = {"dT": e["mean_dT"], "dino": e["mean_dino_c"],
                      "lpips": e["mean_lpips"], "sty": e["styles_improved"],
                      "params": json.loads(a0.read_text()).get("n_params")}
    st = ROOT / "runs/v6_24h/state.json"
    if st.exists():
        for k, v in json.loads(st.read_text()).items():
            e = v.get("eval200") or {}
            if e.get("mean_dT") is None:
                continue
            arms[k] = {"dT": e["mean_dT"], "dino": e["mean_dino_c"],
                       "lpips": e["mean_lpips"], "sty": e.get("styles_improved"),
                       "params": v.get("n_params")}
    return arms


def dominates(a: dict, b: dict) -> bool:
    """a dominates b if it is at least as good on all three axes and better on one."""
    ge = (a["dT"] >= b["dT"] and a["dino"] >= b["dino"] and a["lpips"] <= b["lpips"])
    gt = (a["dT"] > b["dT"] or a["dino"] > b["dino"] or a["lpips"] < b["lpips"])
    return bool(ge and gt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v6_24h/tier_selection")
    args = ap.parse_args()

    arms = collect()
    if not arms:
        raise SystemExit("no arms with eval200 yet")

    print(f"{'arm':16s} {'dT':>9s} {'DINO-C':>8s} {'LPIPS':>8s} {'sty':>5s} {'params':>9s}")
    for k, v in sorted(arms.items(), key=lambda x: -x[1]["dT"]):
        p = f"{v['params']:,}" if v.get("params") else "—"
        print(f"{k:16s} {v['dT']:+9.4f} {v['dino']:8.4f} {v['lpips']:8.4f} "
              f"{str(v['sty'])+'/5':>5s} {p:>9s}")

    print(f"\n=== tier winners (maximise dT inside each content budget) ===")
    winners = {}
    for name, t in TIERS.items():
        ok = {k: v for k, v in arms.items()
              if v["dino"] >= t["dino_min"] and v["lpips"] <= t["lpips_max"]}
        if not ok:
            print(f"  {name:9s} (dino>={t['dino_min']}, lpips<={t['lpips_max']}): "
                  f"NO ARM QUALIFIES")
            winners[name] = None
            continue
        best = max(ok, key=lambda k: ok[k]["dT"])
        winners[name] = best
        print(f"  {name:9s} (dino>={t['dino_min']}, lpips<={t['lpips_max']}): "
              f"**{best}** dT={ok[best]['dT']:+.4f} dino={ok[best]['dino']:.4f} "
              f"lpips={ok[best]['lpips']:.4f}  ({len(ok)} qualifying)")

    # Pareto front
    front = [k for k in arms if not any(dominates(arms[j], arms[k])
                                        for j in arms if j != k)]
    dominated = sorted(set(arms) - set(front))
    print(f"\n=== Pareto front ({len(front)}) ===")
    for k in sorted(front, key=lambda x: -arms[x]["dT"]):
        flag = ""
        if arms[k]["dT"] <= 0:
            # A near-identity operator scores excellent content metrics precisely BECAUSE
            # it changes almost nothing. Such an arm is Pareto-optimal on
            # (dT, DINO-C, LPIPS) while being useless, so it is flagged rather than
            # presented as a candidate.
            flag = "  <-- DEGENERATE (dT <= 0: barely changes the image)"
        print(f"  {k:16s} dT={arms[k]['dT']:+.4f} dino={arms[k]['dino']:.4f} "
              f"lpips={arms[k]['lpips']:.4f}{flag}")
    if dominated:
        print(f"\n=== Pareto-dominated ({len(dominated)}) ===")
        for k in dominated:
            dom = [j for j in arms if j != k and dominates(arms[j], arms[k])]
            print(f"  {k:16s} dominated by {', '.join(dom)}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tier_selection.json").write_text(json.dumps(
        {"tiers": TIERS, "winners": winners, "pareto_front": front,
         "dominated": dominated, "arms": arms}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nwrote {out}/tier_selection.json")


if __name__ == "__main__":
    main()
