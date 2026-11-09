#!/usr/bin/env bash
# M0 plus the extrapolation diagnostic.
#
# alpha in [0,1] is the main-line interpolation knob. alpha > 1 is run ONLY to measure the
# dT / Top-1 trade-off explicitly, because the plan records that history at alpha 1.8/2.2
# raised Top-1 while lowering dT and exploding LPIPS -- and Top-1 is now the binding
# constraint, so that curve is the most informative thing left to measure.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then
  echo "GPU busy (pid $BUSY); refusing to run."
  exit 2
fi

CKPT=${1:-runs/v6/L7_highC_dim32/L7_highC_dim32.pt}
echo "=== M0 main line (alpha 0..1) on $CKPT ==="
python3 -u -m tools.v6_24h.m0_alpha --ckpt "$CKPT" \
  --alphas 0,0.25,0.5,0.75,1.0 --out runs/v6_24h/m0_main

echo
echo "=== M0 extrapolation diagnostic (alpha 1.25..2.0) ==="
python3 -u -m tools.v6_24h.m0_alpha --ckpt "$CKPT" --allow-extrapolation \
  --alphas 1.25,1.5,1.75,2.0 --out runs/v6_24h/m0_extrap
