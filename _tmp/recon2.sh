#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "=== DATE ==="; date
echo "=== TMUX ==="; tmux ls 2>&1
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== GPU PROCS ==="; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "=== V4 RUNS ==="; ls -1 runs/v4 2>/dev/null | tail -40
echo "=== S2 EVAL FILES ==="; find ../pseudo_gt_24h/eval_render -maxdepth 2 -type d -name 'S2*' 2>/dev/null
echo "=== S2 RESULT ==="; ls -la ../pseudo_gt_24h/eval_render/S2*/ 2>/dev/null
echo "=== CACHE DIRS ==="; ls -1 ../pseudo_gt_24h/train_cache 2>/dev/null | tail -20
echo "=== CKPTS ==="; find runs -maxdepth 3 -name '*.pt' -newermt '-2 days' -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -20
