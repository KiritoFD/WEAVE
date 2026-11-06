#!/bin/bash
pkill -f bench_b32_d32
sleep 2
ps aux | grep -E "pgt_render|prof_render" | grep -v grep
nvidia-smi --query-gpu=memory.used --format=csv,noheader
