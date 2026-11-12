#!/usr/bin/env bash
# WEAVE Style-Rank 20-class: 11h training + acceptance evaluation.
# microbatch 2 (1 style x 2 images) | accum 24 | effective 48
# torch.compile ON (in-place Module.compile, dynamic=False), gradient checkpointing OFF
set -u
cd /root/Workspace/WEAVE_VAE
PY=/opt/conda/envs/cu121/bin/python
O=runs/vae_style_rank20

mkdir -p "$O/checkpoints"

echo "=== [$(date)] Style-Rank 11h training start ==="
$PY -u tools/vae_suite/train_style_rank.py \
    --hours 11.0 --n-styles 1 --accum 0 --neg-k 4 --queue-size 4096 \
    --stale-every 200 --drift-every 200 \
    --lr-enc 3e-5 --lr-dec 2e-5 --warmup 50 \
    --save-every 500 --val-every 0 --calib-batches 8 \
    --resume "$O/checkpoints/ckpt_u00200.pt" \
    --seed 42 > "$O/train.log" 2>&1
echo "=== [$(date)] training finished ==="

echo "=== [$(date)] final acceptance evaluation on 20-class eval set ==="
$PY -u tools/vae_suite/eval_style_geometry.py \
    --models "FINAL=$O/final_diffusers_vae,ckpt200=$O/checkpoints/ckpt_u00200.pt" > "$O/eval.log" 2>&1
echo "=== [$(date)] evaluation finished ==="
echo "=== ALL DONE ==="
