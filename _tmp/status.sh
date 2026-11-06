#!/usr/bin/env bash
echo "== date"; date
echo "== nvidia-smi"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== compute procs"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "== python procs"; ps -eo pid,etime,cmd | grep -E "python|build_trt" | grep -v grep | head -20
echo "== tmux"; tmux ls 2>&1
