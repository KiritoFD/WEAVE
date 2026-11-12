#!/usr/bin/env bash
# WEAVE Style-Rank 20-class: 200-step verification run + evaluation
set -u
cd /root/Workspace/WEAVE_VAE
PY=/opt/conda/envs/cu121/bin/python
O=runs/vae_style_rank20

mkdir -p "$O/checkpoints"

echo "=== [$(date)] 200-step Style-Rank Verification Start ==="
$PY -u tools/vae_suite/train_style_rank.py \
    --steps 200 \
    --n-styles 1 --accum 0 --neg-k 4 --queue-size 4096 \
    --stale-every 100 --drift-every 100 \
    --lr-enc 3e-5 --lr-dec 2e-5 --warmup 50 \
    --save-every 200 --val-every 0 --calib-batches 8 --seed 42 > "$O/train_200.log" 2>&1

echo "=== [$(date)] 200-step training completed, running clean evaluation on ckpt_u00200.pt ==="
$PY -u tools/vae_suite/eval_style_geometry.py \
    --models "ckpt200=$O/checkpoints/ckpt_u00200.pt" > "$O/eval_200.log" 2>&1

echo "=== [$(date)] 200-step evaluation completed ==="
