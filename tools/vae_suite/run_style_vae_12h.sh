#!/usr/bin/env bash
# WEAVE Style-VAE: 12h serial run (S1 -> S2) then final acceptance evaluation.
set -u
cd /root/Workspace/WEAVE_VAE
PY=/opt/conda/envs/cu121/bin/python
LOG=runs_vae/style_vae

mkdir -p "$LOG"

echo "=== [$(date)] Style-VAE 12h training start (S1 0-4h, S2 4-11h) ==="
$PY -u tools/vae_suite/train_style_vae.py \
    --s1-hours 4.0 --total-hours 11.0 \
    --batch 6 --accum 8 \
    --lr-enc 3e-5 --lr-dec 2e-5 \
    --warmup 50 --save-every 500 \
    --seed 42 > "$LOG/train.log" 2>&1
echo "=== [$(date)] training finished ==="

echo "=== [$(date)] final acceptance evaluation ==="
$PY -u tools/vae_suite/eval_style_vae.py > "$LOG/eval.log" 2>&1
echo "=== [$(date)] evaluation finished ==="
echo "=== ALL DONE ==="
