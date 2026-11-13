#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u tools/v6/train_eval_v6.py --name DBG_rank --mode direct --base-dim 16 \
  --steps 2 --microbatch 80 --accum 1 --seed 42 --canon-mode c1_gentle \
  --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 --ref-top-k 1024 \
  --init-from runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt \
  --rank-critic runs/v6_next/probes/r1_r1.pt --rank-lambda 0.003 --rank-tau 0.5 2>&1 \
  | grep -aE "DBG|G1 diag|step 1/|step 2/" | head -n 10
