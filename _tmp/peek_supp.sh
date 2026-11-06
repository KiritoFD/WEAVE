#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/p2/supp_bench.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines()]
print("n", len(rows))
r = rows[0]
print("keys", sorted(r.keys()))
print(json.dumps({k: r[k] for k in ("config","source_id","q_mean","coverage","argmax_style","argmax_base","bands","R_active","mask","infeasible") if k in r}, ensure_ascii=False, indent=1)[:2500])
print("meta", json.loads(Path("runs/p2/supp_bench_meta.json").read_text()))
PY
