#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("runs/blog_sbflow/v10/exp_*/train_summary.json")):
    d = json.loads(open(f).read())
    h = d.get("loss_history", [])
    tag = f.split("/")[-2]
    print(f"{tag}: updates={d.get('updates')} batch={d.get('batch')} "
          f"pairs={len(d.get('unordered_pairs', []))} "
          f"s/upd={d.get('sec_per_update')} peak={d.get('peak_vram_mib')}MiB")
    if h:
        pts = [h[0]] + h[len(h)//4:len(h)//4+1] + h[len(h)//2:len(h)//2+1] + [h[-1]]
        seen = set()
        for p in pts:
            if p["step"] in seen:
                continue
            seen.add(p["step"])
            print(f"   step {p['step']:5d} loss={p['loss']:.4f} f={p.get('f_mse',0):.3f} b={p.get('b_mse',0):.3f}")
PY
echo "=== per-pair update budget ==="
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("runs/blog_sbflow/v10/exp_*/train_summary.json")):
    d = json.loads(open(f).read())
    npairs = max(1, len(d.get("unordered_pairs", [])))
    print(f"  {f.split('/')[-2]}: {d.get('updates')} updates / {npairs} pairs = "
          f"{d.get('updates')//npairs} updates per pair")
PY
