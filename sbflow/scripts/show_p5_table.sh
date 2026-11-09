#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path

STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
models = ["L10", "L8", "L9", "L2"]
tiers = ["heldout400", "same150"]

print("=" * 96)
for tier in tiers:
    print(f"\n### {tier}")
    hdr = "%-6s%8s%8s%7s%9s%8s%8s%6s" % ("model", "n", "dT", "dS", "top1", "DINO-C",
                                         "LPIPS", "sty+")
    print(hdr)
    for m in models:
        p = Path(f"runs/v6_24h/eval_{tier}_{m}/eval.json")
        if not p.exists():
            print("%-6s  missing" % m); continue
        d = json.loads(p.read_text())
        ci = d.get("dT_cluster_ci") or {}
        print("%-6s%8d%8s%8s%9.3f%8.4f%8.4f%6s" % (
            m, d["n"], "%+.4f" % d["mean_dT"], "%+.4f" % d["mean_dS"],
            d["mean_top1"], d["mean_dino_c"], d["mean_lpips"],
            "%d/5" % d["styles_improved"]))
        if ci:
            print("        dT cluster CI [%+.4f, %+.4f]  (%d clusters)"
                  % (ci["ci"][0], ci["ci"][1], ci["n_clusters"]))

print("\n### per-target dT")
print("%-6s%8s" % ("tier", "model") + "".join("%10s" % s[:9] for s in STY))
for tier in tiers:
    for m in models:
        p = Path(f"runs/v6_24h/eval_{tier}_{m}/eval.json")
        if not p.exists(): continue
        d = json.loads(p.read_text())
        pt = d["per_target_dT"]
        print("%-6s%8s" % (tier[:5], m) + "".join("%+10.4f" % pt[s] for s in STY))

print("\n### per-target top1")
print("%-6s%8s" % ("tier", "model") + "".join("%10s" % s[:9] for s in STY))
for tier in tiers:
    for m in models:
        p = Path(f"runs/v6_24h/eval_{tier}_{m}/eval.json")
        if not p.exists(): continue
        d = json.loads(p.read_text())
        rows = d["rows"]
        print("%-6s%8s" % (tier[:5], m) + "".join(
            "%10.3f" % (sum(r["target_top1"] for r in rows if r["tgt_style"] == s) /
                        max(1, sum(1 for r in rows if r["tgt_style"] == s)))
            for s in STY))
PY
