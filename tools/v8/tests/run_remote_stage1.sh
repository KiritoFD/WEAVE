#!/usr/bin/env bash
# Official V8 stage-1 protocol: train B0-B3, then evaluate on the frozen official
# tasks only (600 cross-style / 750 all). No val200 anywhere.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
N=800
echo "=== V8 Stage1 common-update training: N=${N} ==="
python3 -u tools/v8/train.py --name B0 --tag B0 --pairing random --sigma-mid 0.00 --steps ${N} --microbatch 16 --accum 8 --seed 42
python3 -u tools/v8/train.py --name B1 --tag B1 --pairing random --sigma-mid 0.10 --steps ${N} --microbatch 16 --accum 8 --seed 42
python3 -u tools/v8/train.py --name B2 --tag B2 --pairing ot     --sigma-mid 0.00 --steps ${N} --microbatch 16 --accum 8 --seed 42
python3 -u tools/v8/train.py --name B3 --tag B3 --pairing ot     --sigma-mid 0.10 --steps ${N} --microbatch 16 --accum 8 --seed 42
echo "=== V8 Stage1 official evaluation (600 cross + 750 all) ==="
bash tools/v8/tests/run_remote_eval_shards.sh test750
echo "=== V8 STAGE1 COMPLETE ==="
