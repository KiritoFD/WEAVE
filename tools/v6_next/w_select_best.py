"""Select the best W student from the 750 eval and emit its checkpoint path.

Criterion (pre-set): highest Top-1 among arms with DINO-C >= 0.78; tie-break by
DINO-S. Falls back to W2 if no candidate qualifies, so the pipeline never stalls.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "runs/v6_w"
ARMS = ["W1_FeatDistill_218", "W2_LossClean_218",
        "W3_LossClean_218_l50"]  # W4 (845K+SK) disabled: 50-step stability failure


def main():
    cands = []
    for a in ARMS:
        p = W / f"eval_{a}" / "eval.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))["cross600"]
        cands.append((a, s["top1"], s["dino_s"], s["dino_c"]))
    if not cands:
        print("W2_LossClean_218")  # safe default
        return
    qualified = [c for c in cands if c[3] >= 0.78] or cands
    best = max(qualified, key=lambda c: (c[1], c[2]))
    print(f"runs/v6_w/{best[0]}/{best[0]}.pt")
    # record choice
    (W / "w_best_arm.json").write_text(json.dumps({
        "arm": best[0], "top1": best[1], "dino_s": best[2], "dino_c": best[3],
        "qualified_count": len(qualified),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
