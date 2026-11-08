#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
N=800
echo "=== V8 Stage1 common-update training: N=${N}, same seed and effective batch ==="
python3 -u tools/v8/train.py --name B0 --tag B0 --pairing random   --sigma-mid 0.00 --steps ${N} --microbatch 16 --accum 8 --seed 42
python3 -u tools/v8/train.py --name B1 --tag B1 --pairing random   --sigma-mid 0.10 --steps ${N} --microbatch 16 --accum 8 --seed 42
python3 -u tools/v8/train.py --name B2 --tag B2 --pairing ot       --sigma-mid 0.00 --steps ${N} --microbatch 16 --accum 8 --seed 42
python3 -u tools/v8/train.py --name B3 --tag B3 --pairing ot       --sigma-mid 0.10 --steps ${N} --microbatch 16 --accum 8 --seed 42
echo "=== V8 Stage1 VAE No-op val200 reference ==="
python3 -u tools/v8/evaluate.py --ckpt NOOP --name VAE_Noop --tasks val200 --solvers euler4
echo "=== V8 Stage1 EMA val200 evaluation (4/8/16 NFE) ==="
for B in B0 B1 B2 B3; do
  python3 -u tools/v8/evaluate.py --ckpt runs/v8/${B}/model.pt --name ${B}_ema --tasks val200 --solvers euler4,euler8,euler16
done
echo "=== V8 STAGE1 COMPLETE ==="
