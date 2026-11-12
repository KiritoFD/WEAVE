#!/bin/bash
set -e
cd /root/Workspace/WEAVE_VAE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_FILE="/root/Workspace/WEAVE_VAE/runs/all_generation.log"
mkdir -p /root/Workspace/WEAVE_VAE/runs

echo "=== STARTING ALL IMAGE GENERATION AT $(date) ===" > "$LOG_FILE"

/opt/conda/envs/cu121/bin/python -u tools/v6_next/generate_all_swap_images.py \
  --variants sdvae,a0,s10,cedit \
  --arms Q2a,Oracle1,B4,B3,B0 \
  --bs 16 \
  --workers 8 \
  >> "$LOG_FILE" 2>&1

echo "=== ALL IMAGE GENERATION COMPLETED AT $(date) ===" >> "$LOG_FILE"
