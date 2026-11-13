"""Test the prototype-collapse hypothesis for the target-similarity / SWD anti-correlation.

Observation: e2 has the BEST target acquisition (dT -0.0289, Top-1 0.130) and the WORST
distribution match (SWD 0.02770), while e1 is the mirror image. If aggressive transport
pushes every image toward the target prototype, per-image cosine (dT) rises while the
cloud loses diversity and moves away from the real target distribution (SWD).

This computes, per target style, the mean pairwise cosine distance inside each arm's
generated CSD cloud, normalised by the same statistic for the real reference cloud.
ratio << 1 means the generated cloud has collapsed toward a narrow region.

Pure CPU: reads only the stored csd_embedding of each row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SH = {"Early_Renaissance": "ER", "Impressionism": "Imp", "Minimalism": "Min",
      "Rococo": "Roc", "Ukiyo_e": "Uki"}


def spread(cloud: np.ndarray) -> float:
    x = cloud / np.clip(np.linalg.norm(cloud, axis=1, keepdims=True), 1e-12, None)
    n = len(x)
    if n < 2:
        return 0.0
    sim = x @ x.T
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sim[iu].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/blog_sbflow/v11")
    ap.add_argument("--arms", default="ctrl,d1,d2,e1,e2")
    ap.add_argument("--out", default="runs/blog_sbflow/v11/spread_check.json")
    args = ap.parse_args()

    refs = torch.load(ROOT / "eval_protocol/cache/references_v1_csd_cpu.pt",
                      map_location="cpu", weights_only=False)
    ref_spread = {s: spread(np.asarray(refs["styles"][s]["csd"], dtype=np.float32))
                  for s in STYLES}

    paths = {
        "ctrl": ROOT / "runs/blog_sbflow/suite40/score_01_quick200/style_scores.json",
        "d1": ROOT / "runs/blog_sbflow/v11/score_d1_quick200/style_scores.json",
        "d2": ROOT / "runs/blog_sbflow/v11/score_d2_quick200/style_scores.json",
        "e1": ROOT / "runs/blog_sbflow/v11/score_e1_quick200/style_scores.json",
        "e2": ROOT / "runs/blog_sbflow/v11/score_e2_quick200/style_scores.json",
    }

    out = {"ref_spread": ref_spread, "arms": {}}
    print(f"{'arm':5s}" + "".join(f"{SH[s]:>8s}" for s in STYLES) + f"{'mean':>9s}")
    print(f"{'real':5s}" + "".join(f"{ref_spread[s]:8.4f}" for s in STYLES)
          + f"{np.mean(list(ref_spread.values())):9.4f}")
    for a in args.arms.split(","):
        p = paths.get(a)
        if p is None or not p.exists():
            print(f"  {a}: missing")
            continue
        rows = json.loads(p.read_text())["rows"]
        per, ratios = {}, {}
        for s in STYLES:
            cloud = np.stack([np.asarray(r["csd_embedding"], dtype=np.float32)
                              for r in rows if r["target"] == s], axis=0)
            if len(cloud) < 2:
                continue
            per[s] = spread(cloud)
            ratios[s] = per[s] / max(ref_spread[s], 1e-12)
        out["arms"][a] = {"spread": per, "ratio": ratios,
                          "mean_spread": float(np.mean(list(per.values()))),
                          "mean_ratio": float(np.mean(list(ratios.values())))}
        print(f"{a:5s}" + "".join(f"{per[s]:8.4f}" if s in per else f"{'-':>8s}"
                                 for s in STYLES)
              + f"{out['arms'][a]['mean_spread']:9.4f}")
    print()
    print(f"{'ratio':5s}" + "".join(f"{SH[s]:>8s}" for s in STYLES) + f"{'mean':>9s}")
    for a, v in out["arms"].items():
        print(f"{a:5s}" + "".join(f"{v['ratio'][s]:8.3f}" if s in v["ratio"] else f"{'-':>8s}"
                                 for s in STYLES) + f"{v['mean_ratio']:9.3f}")
    print("\nratio << 1 -> the generated cloud has collapsed toward a narrow region,")
    print("i.e. higher per-image cosine to the prototype can coexist with worse SWD.")

    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
