#!/usr/bin/env bash
# P0..P3 diagnostics on the existing 04M checkpoint. No training.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
TESTS=${1:-p0,p1,p2,p3}
python3 -u -m sbflow.diagnose_p \
  --ckpt runs/blog_sbflow/v10/exp_04M/model_step6000.pt \
  --tests "$TESTS" --threads 6 \
  --out runs/blog_sbflow/v10/diagnostics_04M.json 2>&1
