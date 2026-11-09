#!/usr/bin/env bash
# Third queue pass: waits for the post-queue driver to finish, then trains the two combination
# arms that were added after the main queue had already been launched.
#
# L8_k1024_dim32       dim32 + 1024/style reference  (two independently validated wins)
# L9_k1024_dim32_lc075 dim32 + 1024/style + lambda_c 0.75 (the full combination)
#
# It must wait for postq because the queue's own GPU-busy pre-flight would otherwise mark both
# arms blocked_gpu and stop.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/v6_24h/pass3.log
mkdir -p runs/v6_24h

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for tmux postq to finish"
for i in $(seq 1 480); do          # up to 4 hours, polled every 30 s
  if ! tmux has-session -t postq 2>/dev/null; then
    say "postq is gone after ${i} polls"
    break
  fi
  sleep 30
done

# even after postq exits the GPU may still hold its last process briefly
for i in $(seq 1 40); do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    say "GPU free after ${i} extra polls"
    break
  fi
  sleep 15
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  say "GPU still busy; aborting pass 3 rather than contending"
  exit 2
fi

say "=========== pass 3: L8 + L9 ==========="
python3 -u -m tools.v6_24h.queue --run --max-hours 3 \
  --arms L8_k1024_dim32,L9_k1024_dim32_lc075 2>&1 | tee -a "$LOG"

say "PASS3 DONE"
