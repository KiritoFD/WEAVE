#!/usr/bin/env bash
# Quantify the Sinkhorn early-stop saving at the real training configuration.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

echo "=== 32M, OT on, 200 iters (reference) ==="
python3 -u -m sbflow.scripts.bench_components --tag v9_32m_ot200 --width 384 --depth 12 --heads 6 \
  --microbatch 16 --accum 4 --ot-ratio 0.5 --ot-iters 200 --updates 100

echo "=== 32M, OT on, early-stop tol=1e-6 ==="
python3 -u -m sbflow.scripts.bench_components --tag v9_32m_ot_tol --width 384 --depth 12 --heads 6 \
  --microbatch 16 --accum 4 --ot-ratio 0.5 --ot-iters 200 --ot-tol 1e-6 --updates 100

echo "=== 32M, OT off, mb256 x accum1 (batch-matched) ==="
python3 -u -m sbflow.scripts.bench_components --tag v9_32m_noOT_mb256 --width 384 --depth 12 --heads 6 \
  --microbatch 256 --accum 1 --ot-ratio 0.0 --updates 100

echo "=== 8M, OT off, mb256 x accum1 (batch-matched) ==="
python3 -u -m sbflow.scripts.bench_components --tag v9_mid_noOT_mb256 --width 256 --depth 6 --heads 4 \
  --microbatch 256 --accum 1 --ot-ratio 0.0 --updates 100
