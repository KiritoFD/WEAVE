#!/usr/bin/env bash
# Cost table: 32M vs Lite, with and without Sinkhorn OT.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

echo "=== 32M, OT off, mb16 x accum4 ==="
python3 -u -m sbflow.scripts.bench_components --tag v9_32m_noOT --width 384 --depth 12 --heads 6 \
  --microbatch 16 --accum 4 --ot-ratio 0.0 --updates 100

echo "=== 32M, OT on (tol early stop), mb16 x accum4 ==="
python3 -u -m sbflow.scripts.bench_components --tag v9_32m_ot_fast --width 384 --depth 12 --heads 6 \
  --microbatch 16 --accum 4 --ot-ratio 0.5 --updates 100

echo "=== Lite, OT off, mb256 x accum1 ==="
python3 -u -m sbflow.scripts.bench_components --tag v9lite_noOT_mb256 --width 128 --depth 6 --heads 4 \
  --microbatch 256 --accum 1 --ot-ratio 0.0 --updates 100
