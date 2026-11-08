#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
echo "=== V8 P0 local math tests (remote WSL) ==="
python3 -u tools/v8/tests/run_all.py
echo "=== V8 P0 data split test ==="
python3 -u tools/v8/tests/test_data_split.py
echo "=== V8 P0 OT pairing audit ==="
python3 -u tools/v8/tests/test_ot.py
echo "=== V8 100-update benchmark ==="
python3 -u tools/v8/benchmark.py --microbatch 16 --accum 8 --updates 100
echo "=== V8 P0/BENCH COMPLETE ==="
