#!/usr/bin/env bash
# P3-long: separate training budget from model capacity.
#
# Waits for any running GPU job to finish (single-GPU discipline), trains the pivot model
# for 40,000 equivalent steps, then samples 100 images per style at the 20k and 40k
# checkpoints so the trend is visible rather than a single endpoint.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=runs/blog_sbflow/v11
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"
mkdir -p "$ROOT/logs"

echo "[p3long] $(date +%H:%M:%S) waiting for other GPU jobs"
while pgrep -f "sbflow.(train5|gen5|traj_check|v11_run)" >/dev/null 2>&1; do
  sleep 60
done
echo "[p3long] $(date +%H:%M:%S) GPU free, starting 40000-step pivot training"

python3 -u -m sbflow.train5 --config configs/v11/p3long.json --caches $C \
  2>&1 | tee -a "$ROOT/logs/p3long_train.log"
echo "[p3long] $(date +%H:%M:%S) training done"

for EQ in 20000 40000; do
  CK="$ROOT/p3long/model_equiv${EQ}.pt"
  if [ ! -f "$CK" ]; then
    echo "[p3long] missing $CK, skip"
    continue
  fi
  echo "[p3long] $(date +%H:%M:%S) sampling at equiv${EQ}"
  python3 -u -m sbflow.gen_uncond --ckpt "$CK" --out "$ROOT/p3long_gen_${EQ}" \
    --per-style 100 --steps 4 --method heun --weights ema \
    2>&1 | tee -a "$ROOT/logs/p3long_gen_${EQ}.log"
  python3 -u -m sbflow.score_uncond --gen "$ROOT/p3long_gen_${EQ}" \
    --out "$ROOT/p3long_score_${EQ}" --threads 4 \
    2>&1 | tee -a "$ROOT/logs/p3long_score_${EQ}.log"
done

echo "[p3long] $(date +%H:%M:%S) P3-LONG DONE"
