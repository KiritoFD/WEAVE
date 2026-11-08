#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
echo "=== Retry V8 B3 euler16 shard (official test750) ==="
python3 -u tools/v8/evaluate.py --ckpt runs/v8/B3/model.pt --name B3_ema --tasks test750 --solvers euler16 --out-suffix _euler16 --batch 8
python3 -u tools/v8/merge_eval_shards.py B3_ema test750
echo "=== DONE ==="
