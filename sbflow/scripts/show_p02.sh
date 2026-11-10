#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path

STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
print("%-16s%5s%9s%9s%8s%8s%8s%8s%6s" % ("run", "n", "dT", "dS", "top1", "margin",
                                         "DINO-C", "LPIPS", "sty+"))
for tag in ("bench", "full"):
    for name in ("sdA", "sdB"):
        p = Path(f"runs/v6_next/eval_{name}_{tag}/eval.json")
        if not p.exists():
            print("%-16s  missing" % f"{name}_{tag}"); continue
        d = json.loads(p.read_text())
        print("%-16s%5d%+9.4f%+9.4f%8.3f%8.4f%8.4f%8.4f%6s" % (
            f"{name}_{tag}", d["n"], d["mean_dT"], d["mean_dS"], d["mean_top1"],
            d["mean_margin"], d["mean_dino_c"], d["mean_lpips"],
            "%d/5" % d["styles_improved"]))

print("\nper-target dT")
print("%-16s" % "run" + "".join("%10s" % s[:9] for s in STY))
for tag in ("bench", "full"):
    for name in ("sdA", "sdB"):
        p = Path(f"runs/v6_next/eval_{name}_{tag}/eval.json")
        if not p.exists(): continue
        d = json.loads(p.read_text())
        pt = d["per_target_dT"]
        print("%-16s" % f"{name}_{tag}" + "".join("%+10.4f" % pt[s] for s in STY))

print("\nper-target top1")
print("%-16s" % "run" + "".join("%10s" % s[:9] for s in STY))
for tag in ("bench", "full"):
    for name in ("sdA", "sdB"):
        p = Path(f"runs/v6_next/eval_{name}_{tag}/eval.json")
        if not p.exists(): continue
        d = json.loads(p.read_text())
        rows = d["rows"]
        cells = []
        for s in STY:
            sub = [r for r in rows if r["tgt_style"] == s]
            cells.append(sum(r["target_top1"] for r in sub) / max(1, len(sub)))
        print("%-16s" % f"{name}_{tag}" + "".join("%10.3f" % c for c in cells))

print("\n5x5 confusion (rows=target, cols=pred) for the full runs")
for name in ("sdA", "sdB"):
    p = Path(f"runs/v6_next/eval_{name}_full/eval.json")
    if not p.exists(): continue
    d = json.loads(p.read_text())
    conf = d.get("confusion_target_vs_pred")
    if not conf: continue
    print(f"-- {name}")
    print("%-20s" % "" + "".join("%9s" % s[:8] for s in STY))
    for i, s in enumerate(STY):
        print("%-20s" % s + "".join("%9d" % c for c in conf[i]))
PY
