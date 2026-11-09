#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import json
from pathlib import Path
import torch

CACHE = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]

print("=== sample training-cache path strings ===")
train_raw = set()
for f in FILES:
    d = torch.load(CACHE / f, map_location="cpu", weights_only=False)
    ids = d.get("files") or d.get("ids") or []
    print(f"  {f}: keys={[k for k in d.keys() if k in ('files','ids','subdir','schema')]}")
    print(f"    subdir={d.get('subdir')!r}  first3={ids[:3]}")
    train_raw |= {str(x) for x in ids}

print("\n=== sample eval reference / source paths ===")
r = json.loads(Path("eval_protocol/references_v1.json").read_text())
s0 = next(iter(r["references"]))
items = r["references"][s0]
rel = [it["rel_path"] if isinstance(it, dict) else it for it in items]
print(f"  refs[{s0}] first3={rel[:3]}")

t = json.loads(Path("eval_protocol/tasks_750.json").read_text())
print(f"  task keys={list(t['tasks'][0].keys())}")
print(f"  first task source_rel_path={t['tasks'][0].get('source_rel_path')!r}")
print(f"  first task source_filename={t['tasks'][0].get('source_filename')!r}")

# compare on FULL strings and on (parent, name) pairs
ref_full = set()
for s, its in r["references"].items():
    for it in its:
        ref_full.add(str(it["rel_path"] if isinstance(it, dict) else it))
src_full = {str(x.get("source_rel_path", x.get("source_filename", ""))) for x in t["tasks"]}

print("\n=== overlap by FULL string ===")
print(f"  refs vs train : {len(ref_full & train_raw)}")
print(f"  srcs vs train : {len(src_full & train_raw)}")
print(f"  refs vs srcs  : {len(ref_full & src_full)}")
if ref_full & train_raw:
    print(f"  example ref/train overlap: {sorted(ref_full & train_raw)[:3]}")
if src_full & train_raw:
    print(f"  example src/train overlap: {sorted(src_full & train_raw)[:3]}")

print("\n=== prefix families ===")
from collections import Counter
def fam(xs):
    return Counter(str(x).split("/")[0] for x in xs).most_common(5)
print(f"  train families: {fam(train_raw)}")
print(f"  ref   families: {fam(ref_full)}")
print(f"  src   families: {fam(src_full)}")
PY
