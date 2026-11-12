#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONIOENCODING=utf-8
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("runs/v6/evaluator_v2_all_results.json").read_text())

print("=== verifications (parity anchors) ===")
for k, v in d["verifications"].items():
    print(f"\n{k}:")
    if isinstance(v, dict):
        for kk, vv in v.items():
            if isinstance(vv, dict):
                keep = {a: b for a, b in vv.items()
                        if a in ("target_top1", "mean_dT", "mean_dino_c", "mean_lpips",
                                 "target_margin", "n", "protocol", "zero_point")}
                print(f"    {kk}: {keep if keep else list(vv.keys())[:8]}")
            else:
                print(f"    {kk}: {vv}")
    else:
        print(f"    {v}")

print("\n=== methods_evaluated: keys and a sample ===")
for name in list(d["methods_evaluated"])[:3]:
    v = d["methods_evaluated"][name]
    print(f"\n{name}: keys={list(v.keys())[:12]}")
    for k in ("target_top1", "mean_dT", "mean_dino_c", "mean_lpips", "n", "protocol"):
        if k in v:
            print(f"    {k}: {v[k]}")

print("\n=== Identity row (the zero-point anchor) ===")
ident = d["methods_evaluated"].get("Identity")
if ident:
    for k, v in ident.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k}: {v}")
PY
