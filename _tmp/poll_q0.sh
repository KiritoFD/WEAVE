#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "=== procs ==="
ps -ef | grep -E "python3" | grep -v grep | head -10
echo "=== gpu ==="
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== q0 log tail ==="
tail -n 8 runs/p2/pgt_q0.log 2>/dev/null
echo "=== q0 json ==="
ls -la runs/p2/pgt_q0.json 2>/dev/null
