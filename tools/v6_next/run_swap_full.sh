#!/bin/bash
cd /root/Workspace/WEAVE_VAE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/opt/conda/envs/cu121/bin/python -u tools/v6_next/vae_swap_full.py \
  --bs 8 \
  --variants sdvae,a0,s10,cedit \
  --arms Q2a,Oracle1,B4 \
  > runs/vae_swap_full.log 2>&1
echo "SWAP_DONE rc=$?" >> runs/vae_swap_full.log
