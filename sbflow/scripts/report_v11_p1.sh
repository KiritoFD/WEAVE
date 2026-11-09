#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
STYLES = ["Early_Renaissance","Impressionism","Minimalism","Rococo","Ukiyo_e"]
SH = {"Early_Renaissance":"ER","Impressionism":"Imp","Minimalism":"Min","Rococo":"Roc","Ukiyo_e":"Uki"}
rows = {}
for a in ("d1","d2","e1","e2"):
    p = Path(f"runs/blog_sbflow/v11/score_{a}_quick200/style_scores.json")
    if p.exists():
        rows[a] = json.loads(p.read_text())["aggregate"]
# control rescored with the corrected aggregate
import torch, sys
sys.path.insert(0, ".")
from sbflow.score_cpu import aggregate
refs = torch.load("eval_protocol/cache/references_v1_csd_cpu.pt", map_location="cpu",
                  weights_only=False)
c = json.loads(Path("runs/blog_sbflow/suite40/score_01_quick200/style_scores.json").read_text())
rows["ctrl"] = aggregate(c["rows"], refs, float(refs["rbf_gamma"]))

print(f"{'arm':6s} {'top1':>7s} {'margin':>8s} {'dT':>8s} {'dS':>8s} {'swd':>8s} {'mmd':>8s} {'dino':>6s} {'lpips':>6s}")
for k in ("ctrl","d1","d2","e1","e2"):
    if k not in rows: continue
    o = rows[k]["overall"]
    print(f"{k:6s} {o['target_top1']:7.4f} {o['target_margin']:+8.4f} {o['csd_dT']:+8.4f} "
          f"{o['csd_dS']:+8.4f} {o['csd_swd']:8.5f} {o['csd_mmd2_unbiased']:8.5f} "
          f"{o['dino_c']:6.4f} {o['lpips_to_input']:6.4f}")

for metric in ("csd_dT","csd_swd","target_top1"):
    print(f"\n### {metric}")
    print(f"{'arm':6s}" + "".join(f"{SH[s]:>8s}" for s in STYLES))
    for k in ("ctrl","d1","d2","e1","e2"):
        if k not in rows: continue
        bt = rows[k]["by_target"]
        print(f"{k:6s}" + "".join(f"{bt[s][metric]:8.4f}" if s in bt else f"{'-':>8s}"
                                  for s in STYLES))
print(f"\nSWD floor (real vs ref) = 0.01111")
PY
