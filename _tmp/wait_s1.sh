#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
for i in $(seq 1 110); do
  if ! pgrep -f v4_eval750.py >/dev/null 2>&1; then break; fi
  sleep 5
done
echo "== eval log =="
grep -E "v4e|Traceback|Error|SWD" runs/v4/S1_A_P5/eval750.log | tail -6
echo "== json =="
cat /mnt/i/Latent_Style/pseudo_gt_24h/eval_render/S1_A_P5_final/v4_eval750.json 2>/dev/null
echo "== procs/gpu =="
ps aux | grep -E "v4_eval750|run.py" | grep -v grep | head -3
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
