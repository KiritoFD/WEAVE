#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== chain pane ==="
tmux capture-pane -p -t v11chain 2>/dev/null | grep -v '^$' | tail -30 || echo "  no pane"
echo "=== probe rho log ==="
grep -a "rho\|grad_norm" runs/blog_sbflow/v11/logs/probe_rho.log 2>/dev/null | tail -6 || echo "  none"
echo "=== p3 train progress ==="
tail -n 4 runs/blog_sbflow/v11/logs/p3_train.log 2>/dev/null || echo "  no p3 log"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
