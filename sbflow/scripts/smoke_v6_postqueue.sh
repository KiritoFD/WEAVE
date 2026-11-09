#!/usr/bin/env bash
# Post-queue GPU smoke test: exercise all three evaluation entry points on 4 tasks so that
# plumbing bugs surface in ~1 minute instead of after a long full run.
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

CKPT=runs/v6/N3_dim32/N3_dim32.pt
echo "########## 1) eval_tiers smoke (Heldout400, 4 tasks) ##########"
python3 -u -m tools.v6_24h.eval_tiers --ckpt "$CKPT" \
  --tasks runs/v6_24h/heldout400_tasks.json --out runs/v6_24h/_smoke_heldout \
  --limit 4 2>&1 | tail -n 12

echo "########## 2) eval_tiers smoke (Same150, 4 tasks) ##########"
python3 -u -m tools.v6_24h.eval_tiers --ckpt "$CKPT" \
  --tasks runs/v6_24h/same150_tasks.json --out runs/v6_24h/_smoke_same \
  --limit 4 2>&1 | tail -n 12

echo "########## 3) m0_alpha smoke (4 tasks, 2 alphas) ##########"
python3 -u -m tools.v6_24h.m0_alpha --ckpt "$CKPT" --limit 4 --alphas 0.0,1.0 \
  --out runs/v6_24h/_smoke_m0 2>&1 | tail -n 12

echo "########## 4) csd_manifold smoke (4 tasks) ##########"
python3 -u -m tools.v6_24h.csd_manifold_check --ckpt "$CKPT" --limit 4 \
  --out runs/v6_24h/_smoke_manifold.json 2>&1 | tail -n 12

echo "########## SMOKE DONE ##########"
