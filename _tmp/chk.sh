#!/bin/bash
cd /mnt/i/Latent_Style
echo "== eval_render dirs =="
ls -d pseudo_gt_24h/eval_render/* 2>/dev/null
echo "== S1_A_P5_final count =="
ls pseudo_gt_24h/eval_render/S1_A_P5_final 2>/dev/null | wc -l
echo "== procs =="
ps aux | grep -E "pgt_render|eval750|run.py" | grep -v grep
echo "== gpu =="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "== progress =="
tail -20 WEAVE/runs/gt/PROGRESS.md 2>/dev/null
