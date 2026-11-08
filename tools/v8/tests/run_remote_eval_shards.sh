#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
TASKS=${1:-val200}

echo "=== V8 ${TASKS} VAE No-op reference (single solver) ==="
if [ ! -f runs/v8/VAE_Noop_${TASKS}_eval.json ]; then
  python3 -u tools/v8/evaluate.py --ckpt NOOP --name VAE_Noop --tasks ${TASKS} --solvers euler4 --batch 8
fi

echo "=== V8 ${TASKS} B0-B3 EMA shard evaluation (one solver per process) ==="
for B in B0 B1 B2 B3; do
  for S in euler4 euler8 euler16; do
    OUT="runs/v8/${B}_ema_${TASKS}_eval_${S}.json"
    if [ -f "$OUT" ]; then echo "skip existing $OUT"; continue; fi
    python3 -u tools/v8/evaluate.py --ckpt runs/v8/${B}/model.pt --name ${B}_ema --tasks ${TASKS} --solvers ${S} --out-suffix _${S} --batch 8
  done
  python3 -u tools/v8/merge_eval_shards.py ${B}_ema ${TASKS}
done
echo "=== V8 ${TASKS} SHARD EVAL COMPLETE ==="
