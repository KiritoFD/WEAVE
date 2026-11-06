#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
python3 _tmp/png_save_threads_inv.py
echo "== tbench =="
cat runs/v4/train_bench.txt 2>/dev/null
