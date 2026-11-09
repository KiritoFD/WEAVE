#!/usr/bin/env bash
# Post-queue driver: waits for the 24h training queue to release the GPU, then runs the
# remaining GPU work in a fixed order. Intended to be launched in its own tmux session so the
# work continues unattended.
#
# Order is deliberate:
#   1. smoke test  -- all three evaluation entry points on 4 tasks each, so a plumbing bug
#                     surfaces in about a minute instead of after a long run
#   2. manifold    -- the decisive check for the Top-1 plateau: are the generated CSD features
#                     off-manifold? Real-image baseline is max_cos 0.6023 / margin 0.3550
#   3. M0 main     -- alpha in [0,1], the main-line interpolation knob
#   4. M0 extrap   -- alpha in [1.25, 2.0], DIAGNOSTIC ONLY: does pushing harder raise Top-1?
#                     The plan records that history at alpha 1.8/2.2 raised Top-1 while
#                     lowering dT and exploding LPIPS, so this curve is the most informative
#                     thing left to measure given Top-1 is the binding constraint.
#   5. P5          -- Heldout400 + Same150 on the finalists, with paired clustered CIs
#
# Usage:  wsl bash -u postqueue_v6_24h.sh [best_ckpt]
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BEST=${1:-runs/v6/L7_highC_dim32/L7_highC_dim32.pt}
LOG=runs/v6_24h/postqueue.log
mkdir -p runs/v6_24h

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for the training queue to release the GPU"
for i in $(seq 1 720); do          # up to 6 hours, polled every 30 s
  if ! pgrep -f "tools.v6_24h.queue" >/dev/null 2>&1; then
    sleep 20                        # let the last eval process exit cleanly
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
      say "GPU free after ${i} polls"
      break
    fi
  fi
  sleep 30
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  say "GPU still busy after the wait loop; aborting rather than contending"
  exit 2
fi

say "=========== 1/5 smoke test ==========="
bash sbflow/scripts/smoke_v6_postqueue.sh 2>&1 | tee -a "$LOG"
if grep -qE "Traceback|RuntimeError|KeyError" "$LOG"; then
  say "smoke test reported an error; stopping before the long runs"
  exit 3
fi

say "=========== 2/5 CSD manifold check ==========="
python3 -u -m tools.v6_24h.csd_manifold_check --ckpt "$BEST" \
  --out runs/v6_24h/csd_manifold_L7.json 2>&1 | tee -a "$LOG"

say "=========== 3/5 M0 main line (alpha 0..1) ==========="
python3 -u -m tools.v6_24h.m0_alpha --ckpt "$BEST" \
  --alphas 0,0.25,0.5,0.75,1.0 --out runs/v6_24h/m0_main 2>&1 | tee -a "$LOG"

say "=========== 4/5 M0 extrapolation (alpha 1.25..2.0, DIAGNOSTIC) ==========="
python3 -u -m tools.v6_24h.m0_alpha --ckpt "$BEST" --allow-extrapolation \
  --alphas 1.25,1.5,1.75,2.0 --out runs/v6_24h/m0_extrap 2>&1 | tee -a "$LOG"

say "=========== 5/5 P5 Heldout400 + Same150 ==========="
# Finalists are chosen per tier plus the mechanism arm:
#   L7  fidelity + balanced tier winner (dim32, lambda_c 0.75, 5/5)
#   L4  the only arm that strictly dominates A0 on all three axes (1024/style reference)
#   S6  best Top-1 (0.040) and best self-consistency (-0.0000); the only mechanism-changing arm
#   L2  style-tier winner (highest dT, but fails both content bars)
python3 -u -m tools.v6_24h.p5_final \
  --ckpts "L7=$BEST" \
          "L4=runs/v6/L4_ref_full/L4_ref_full.pt" \
          "S6=runs/v6/S6_style_keys/S6_style_keys.pt" \
          "L2=runs/v6/L2_highS/L2_highS.pt" 2>&1 | tee -a "$LOG"

say "POSTQUEUE DONE"
