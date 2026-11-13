#!/usr/bin/env bash
# WEAVE — VAE-swap training experiment.
# Re-encode the frozen training pool with A0, then train the SAME W_LossClean
# recipe on the A0 latents. Everything else is identical to the sd-vae-pool arms.
# This directly answers "换 VAE 重新编码再训练看看".
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_vaeswap.log

mkdir -p runs/v6_w pseudo_gt_24h/orig_mean_fp16_a0
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE VAE-SWAP (A0 RE-ENCODE) ===================="

LAT=pseudo_gt_24h/orig_mean_fp16_a0

# 1. Re-encode 15,545 training images with the A0 Encoder (idempotent).
if [ -s "$LAT/reencode_manifest.json" ] && [ -s "$LAT/00_Early_Renaissance.pt" ]; then
  qlog "===== A0 re-encode ALREADY DONE ====="
else
  run_step "A0_reencode" "$LAT/reencode_manifest.json" \
    python3 -u -m tools.v6_next.reencode_a0 --out-dir "$LAT" --batch 16
fi

# 2. Train arms on the A0 latent pool (identical recipe to W2 / plain pixel MSE).
COMMON="--mode direct --out-base runs/v6_w --steps 800 --lr 0.0003 --microbatch 80 --accum 3 --amp-dtype bf16 --canon-mode c1_gentle --w-rec 1.0 --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 --ref-top-k 1024 --seed 42 --latent-dir $LAT"
TEACH="--distill-teacher runs/v6/Q2a_embed_style/Q2a_embed_style.pt --distill-critic runs/v6_next/probes/r1_r1.pt --distill-mode oracle --distill-steps 1 --distill-rms 0.25"
Q2A=runs/v6/Q2a_embed_style/Q2a_embed_style.pt

train_swap () {
  local name=$1 lam=$2; shift 2
  local ck="runs/v6_w/${name}/${name}.pt"
  if [ -s "$ck" ]; then qlog "===== $name ALREADY TRAINED ====="; return 0; fi
  run_step "train_$name" "$ck" \
    python3 -u -m tools.v6.train_eval_v6 \
      --name "$name" --base-dim 16 --init-from "$Q2A" \
      $TEACH --distill-lambda "$lam" $COMMON "$@"
}

# V1: W_LossClean recipe on A0 latents (direct counterpart of W2 on sd-vae latents)
train_swap V1_A0_LossClean 20 \
  --w-distill --w-feat 1.0 --w-kl 1.0 --w-rep 0.5 --w-ll 1.0 \
  --w-class-rms --w-anti-attractor --w-clean-cross --w-ll-canon

# V2: plain pixel-MSE Oracle distill on A0 latents (isolates the VAE effect)
train_swap V2_A0_PixelMSE 20

qlog "==================== VAE-SWAP TRAIN DONE failures=$QFAIL ===================="
