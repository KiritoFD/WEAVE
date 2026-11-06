#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
for i in $(seq 1 110); do
  if grep -q "BENCH DONE" runs/v4/batch_bench.txt 2>/dev/null; then break; fi
  sleep 5
done
cat runs/v4/batch_bench.txt
echo "== per-cell imgs/s =="
for f in /mnt/i/Latent_Style/pseudo_gt_24h/bench_render/*/render_manifest.json; do
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d["cell"], "imgs", d["num_images"], "render_s", d["wall_sec_render"],
      "total_s", d["wall_sec_total"], "imgs/s", round(d["num_images"]/max(d["wall_sec_render"],1e-9),2))
PY
done
