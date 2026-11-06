#!/bin/bash
pkill -f v4_chain.sh
pkill -f v4_eval750.py
sleep 2
echo "== remaining =="
ps aux | grep -E "v4_chain|v4_eval750|run.py" | grep -v grep
echo "== gpu =="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
