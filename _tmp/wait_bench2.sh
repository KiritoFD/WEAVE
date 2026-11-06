#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
for i in $(seq 1 115); do
  if grep -q "BENCH2 DONE" runs/v4/batch_bench2.txt 2>/dev/null; then break; fi
  sleep 5
done
cat runs/v4/batch_bench2.txt
