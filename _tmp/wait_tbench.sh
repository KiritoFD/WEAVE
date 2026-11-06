#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
for i in $(seq 1 115); do
  if grep -q "TRAIN BENCH DONE" runs/v4/train_bench.txt 2>/dev/null; then break; fi
  sleep 5
done
cat runs/v4/train_bench.txt
