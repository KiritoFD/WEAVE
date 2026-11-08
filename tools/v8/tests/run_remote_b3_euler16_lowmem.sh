#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
echo "=== Retry V8 B3 euler16 shard with batch=4 to avoid 12GB VRAM stall ==="
python3 -u tools/v8/evaluate.py --ckpt runs/v8/B3/model.pt --name B3_ema --tasks val200 --solvers euler16 --out-suffix _euler16 --batch 4
python3 -u tools/v8/merge_eval_shards.py B3_ema val200
echo "=== V8 B3 EULER16 SHARD COMPLETE ==="
