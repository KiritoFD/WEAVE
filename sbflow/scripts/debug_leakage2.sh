#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import json
from pathlib import Path
import torch

REPO = "/mnt/i/Latent_Style/WEAVE"
CACHE = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]

def norm(p):
    """Repo-relative POSIX path, lower-cased, for a fair comparison."""
    p = str(p).replace("\\", "/")
    if p.startswith(REPO + "/"):
        p = p[len(REPO) + 1:]
    return p.lower()

train = set()
for f in FILES:
    d = torch.load(CACHE / f, map_location="cpu", weights_only=False)
    train |= {norm(x) for x in (d.get("files") or d.get("ids") or [])}

r = json.loads(Path("eval_protocol/references_v1.json").read_text())
ref = set()
for s, its in r["references"].items():
    for it in its:
        ref.add(norm(it["rel_path"] if isinstance(it, dict) else it))

t = json.loads(Path("eval_protocol/tasks_750.json").read_text())
src = {norm(x.get("source_rel_path", "")) for x in t["tasks"]}

print(f"train unique={len(train)}  ref unique={len(ref)}  src unique={len(src)}")
print()
print("=== normalized overlaps ===")
print(f"  refs  & train = {len(ref & train)}")
print(f"  srcs  & train = {len(src & train)}")
print(f"  refs  & srcs  = {len(ref & src)}")
if ref & train:
    ex = sorted(ref & train)[:5]
    print(f"  LEAK: eval references that ARE in the training pool:")
    for e in ex:
        print(f"        {e}")
if src & train:
    ex = sorted(src & train)[:5]
    print(f"  LEAK: eval sources that ARE in the training pool:")
    for e in ex:
        print(f"        {e}")
if ref & src:
    ex = sorted(ref & src)[:5]
    print(f"  LEAK: eval references that ARE eval sources:")
    for e in ex:
        print(f"        {e}")

# per-style overlap, to catch a single-style leak
print()
print("=== per-style refs & train ===")
for s in r["references"]:
    rs = {norm(it["rel_path"] if isinstance(it, dict) else it) for it in r["references"][s]}
    print(f"  {s:20s} refs={len(rs):3d}  overlap_with_train={len(rs & train)}")
PY
