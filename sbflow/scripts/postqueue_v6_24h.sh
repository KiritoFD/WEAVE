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

BEST=${1:-runs/v6/N7_dim24/N7_dim24.pt}
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
if ! bash sbflow/scripts/smoke_v6_postqueue.sh "$BEST" 2>&1 | tee -a "$LOG"; then
  say "smoke test FAILED; stopping before the long runs"
  exit 3
fi
# belt and braces: a captured traceback in the log also aborts
if grep -qaE "^Traceback" "$LOG"; then
  say "traceback found in the log; stopping before the long runs"
  exit 3
fi

say "=========== 2/6 CSD manifold check ==========="
python3 -u -m tools.v6_24h.csd_manifold_check --ckpt "$BEST" \
  --out runs/v6_24h/csd_manifold_best.json 2>&1 | tee -a "$LOG"

say "=========== 3/6 visual grid for human blind rating ==========="
python3 -u -m tools.v6_24h.make_grid --ckpt "$BEST" --n-sources 4 2>&1 | tee -a "$LOG"

say "=========== 4/6 M0 main line (alpha 0..1) ==========="
python3 -u -m tools.v6_24h.m0_alpha --ckpt "$BEST" \
  --alphas 0,0.25,0.5,0.75,1.0 --out runs/v6_24h/m0_main 2>&1 | tee -a "$LOG"

say "=========== 5/6 M0 extrapolation (alpha 1.25..2.0, DIAGNOSTIC) ==========="
python3 -u -m tools.v6_24h.m0_alpha --ckpt "$BEST" --allow-extrapolation \
  --alphas 1.25,1.5,1.75,2.0 --out runs/v6_24h/m0_extrap 2>&1 | tee -a "$LOG"

say "=========== 6/6 P5 Heldout400 + Same150 ==========="
# Finalists updated after pass3 finished. The k1024 family dominates A0 outright, so P5 now tests
# the arms that actually matter rather than the earlier tier winners:
#   L10  recommended general baseline (dim24 + k1024, 480K, tied with L8 at 43% fewer params)
#   L8   best dT and best self-consistency (dim32 + k1024)
#   L9   best content (dim32 + k1024 + lambda_c 0.75)
#   L2   style-tier winner (highest dT, fails both content bars)
python3 -u -m tools.v6_24h.p5_final \
  --ckpts "L10=runs/v6/L10_k1024_dim24/L10_k1024_dim24.pt" \
          "L8=runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt" \
          "L9=runs/v6/L9_k1024_dim32_lc075/L9_k1024_dim32_lc075.pt" \
          "L2=runs/v6/L2_highS/L2_highS.pt" 2>&1 | tee -a "$LOG"

say "POSTQUEUE DONE"
