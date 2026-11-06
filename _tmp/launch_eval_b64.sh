#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
setsid -f bash -c 'python3 -u tools/pseudo_gt/v4_eval750.py --cell b64_t8 --render-root /mnt/i/Latent_Style/pseudo_gt_24h/bench_render --max-src 30 > runs/v4/b64_t8.eval750.log 2>&1' > /dev/null 2>&1 < /dev/null
sleep 3
echo launched
