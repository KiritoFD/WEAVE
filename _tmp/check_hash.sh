#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
md5sum tools/pseudo_gt/q0_quality.py
grep -n "批量打分\|score(z" tools/pseudo_gt/q0_quality.py | head -5
ls -la /tmp/pgt_q0_smoke.json
