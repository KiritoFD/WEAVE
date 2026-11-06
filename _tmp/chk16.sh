#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== epoch timing per run =="
for t in bench_mb32 bench_mb64 bench_mb128; do
  echo "-- $t"
  grep -a -E "Epoch [0-9]+/25" runs/v4/${t}.log | sed 's/ - INFO.*Epoch/ | Epoch/' | cut -c1-90
done
echo "== eval b64 =="
for i in $(seq 1 60); do
  if grep -q "== all750" runs/v4/b64_t8.eval750.log 2>/dev/null; then break; fi
  sleep 5
done
grep -a -E "v4e|all750|cross600|same150" runs/v4/b64_t8.eval750.log | tail -n 6
cat /mnt/i/Latent_Style/pseudo_gt_24h/bench_render/b64_t8/v4_eval750.json 2>/dev/null
