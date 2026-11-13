#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path

STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
print("%-6s%-12s%5s%9s%9s%8s%8s%8s%6s" % ("model", "tier", "n", "dT", "dS", "top1",
                                          "DINO-C", "LPIPS", "sty+"))
for m in ("L11", "L10", "L9", "L8", "L2"):
    for tier in ("heldout400", "same150"):
        p = Path(f"runs/v6_24h/eval_{tier}_{m}/eval.json")
        if not p.exists():
            print("%-6s%-12s  MISSING" % (m, tier)); continue
        d = json.loads(p.read_text())
        print("%-6s%-12s%5d%+9.4f%+9.4f%8.3f%8.4f%8.4f%6s" % (
            m, tier[:11], d["n"], d["mean_dT"], d["mean_dS"], d["mean_top1"],
            d["mean_dino_c"], d["mean_lpips"], "%d/5" % d["styles_improved"]))
    print()

print("per-target top1 (heldout400)")
print("%-6s" % "model" + "".join("%10s" % s[:9] for s in STY))
for m in ("L11", "L10", "L9", "L8", "L2"):
    p = Path(f"runs/v6_24h/eval_heldout400_{m}/eval.json")
    if not p.exists(): continue
    d = json.loads(p.read_text())
    rows = d["rows"]
    cells = []
    for s in STY:
        sub = [r for r in rows if r["tgt_style"] == s]
        cells.append(sum(r["target_top1"] for r in sub) / max(1, len(sub)))
    print("%-6s" % m + "".join("%10.3f" % c for c in cells))
PY
