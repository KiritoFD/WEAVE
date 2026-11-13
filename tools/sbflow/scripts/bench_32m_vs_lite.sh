#!/usr/bin/env bash
# Compare 32M vs Lite training cost with component breakdown.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

echo "=== V9 32M (width384 depth12 heads6, mb16 x accum4) ==="
python3 -u -m sbflow.scripts.bench_components --tag v9_32m --width 384 --depth 12 --heads 6 \
  --microbatch 16 --accum 4 --updates 100

echo "=== V9-Lite 2M (width128 depth6 heads4, mb16 x accum4) ==="
python3 -u -m sbflow.scripts.bench_components --tag v9lite_2m_mb16 --width 128 --depth 6 --heads 4 \
  --microbatch 16 --accum 4 --updates 100
