#!/usr/bin/env bash
# P1: two 200-step ablations from A0, identical except lambda_anchor.
#   A: anchor x0.1    B: no anchor (0)
# Both share: same seed / sampler / queue / EMA center / Rank+CE / lr / KL / rec.
# CPU eval (500 fixed images, same rebuilt pairs) is used throughout.
set -u
cd /root/Workspace/WEAVE_VAE
PY=/opt/conda/envs/cu121/bin/python
ROOT=runs/vae_style_rank20
CALIB=$ROOT/calibration.json
A=$ROOT/p1_weak_anchor
B=$ROOT/p1_no_anchor
mkdir -p "$A" "$B"

echo "=== [$(date)] P1 arm A: anchor x0.1, 200 steps ==="
$PY -u tools/vae_suite/train_style_rank.py \
    --steps 200 --n-styles 1 --accum 0 --neg-k 4 --queue-size 4096 \
    --stale-every 0 --drift-every 0 \
    --lr-enc 3e-5 --lr-dec 2e-5 --warmup 50 \
    --save-every 200 --val-every 0 --calib-batches 8 \
    --anchor-scale 0.1 --calib-from "$CALIB" \
    --run-dir "$A" --seed 42 > "$A/train.log" 2>&1
echo "=== [$(date)] arm A done ==="

echo "=== [$(date)] P1 arm B: NO anchor, 200 steps (GPU) ==="
$PY -u tools/vae_suite/train_style_rank.py \
    --steps 200 --n-styles 1 --accum 0 --neg-k 4 --queue-size 4096 \
    --stale-every 0 --drift-every 0 \
    --lr-enc 3e-5 --lr-dec 2e-5 --warmup 50 \
    --save-every 200 --val-every 0 --calib-batches 8 \
    --anchor-scale 0 --calib-from "$CALIB" \
    --run-dir "$B" --seed 42 > "$B/train.log" 2>&1 &
PID_B=$!

# CPU-side eval of A0 + arm A while arm B trains on GPU (no GPU contention)
echo "=== [$(date)] CPU eval: A0 + arm A (parallel with arm B) ==="
$PY -u tools/vae_suite/cpu_eval_parallel.py \
    --n-images 500 --shards 32 --threads 2 --batch 2 \
    --models A0,armA_weakanchor \
    --model-paths "armA_weakanchor=/root/Workspace/WEAVE_VAE/$A/checkpoints/ckpt_u00200.pt" \
    --out "$ROOT/p1_eval_A0_armA.json" > "$ROOT/p1_eval_A0_armA.log" 2>&1
echo "=== [$(date)] A0 + arm A eval done ==="

wait $PID_B
echo "=== [$(date)] arm B done ==="

echo "=== [$(date)] CPU eval: arm B ==="
$PY -u tools/vae_suite/cpu_eval_parallel.py \
    --n-images 500 --shards 32 --threads 2 --batch 2 \
    --models armB_noanchor \
    --model-paths "armB_noanchor=/root/Workspace/WEAVE_VAE/$B/checkpoints/ckpt_u00200.pt" \
    --out "$ROOT/p1_eval_armB.json" > "$ROOT/p1_eval_armB.log" 2>&1
echo "=== [$(date)] P1 ALL DONE ==="
