#!/usr/bin/env bash
# ==============================================================================
# WEAVE-VAE V2: 7-Hour Co-Training Pipeline with Direct Latent SWD Ranking Loss
# ==============================================================================
set -e

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="/root/Workspace/.inductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE="1"

VAE_INIT="/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
RAW_MANIFEST="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"
SWD_CALIB="/root/Workspace/WEAVE_VAE/swd_calibration.json"
OUT_DIR="/root/Workspace/WEAVE_VAE/runs_vae/V2"
PYTHON="/opt/conda/envs/cu121/bin/python"

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/train_v2.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================================="
echo "=== [$(date)] Starting WEAVE-VAE V2 Co-Training (Direct Latent SWD Ranking) ==="
echo "=== Base Model: $VAE_INIT ==="
echo "=== Steps: 1800 | microbatch: 6 | effective_batch: 48 | Exposure: 86,400 ==="
echo "=============================================================================="

$PYTHON -u /root/Workspace/WEAVE_VAE/tools/vae_suite/train_v2.py \
    --vae-init-path "$VAE_INIT" \
    --raw-manifest "$RAW_MANIFEST" \
    --calibration-json "$SWD_CALIB" \
    --output-dir "$OUT_DIR" \
    --steps 1800 \
    --microbatch 6 \
    --effective-batch 48 \
    --num-workers 4 \
    --beta-kl 2.252e-5 \
    --lambda-style 0.0100 \
    --lr-encoder 1e-5 \
    --lr-decoder 5e-6 \
    --save-every 200 \
    --seed 42

echo "=============================================================================="
echo "=== [$(date)] WEAVE-VAE V2 Co-Training Completed Successfully! ==="
echo "=============================================================================="
