#!/usr/bin/env bash
# Stop every V9 sbflow process (train, diagnose, eval, bench) and free the GPU.
set -uo pipefail
pkill -f "sbflow\." || true
pkill -f "sbflow/" || true
sleep 3
echo "--- remaining sbflow processes ---"
ps -eo pid,etime,args | grep -E "sbflow" | grep -v grep || echo "NONE"
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
