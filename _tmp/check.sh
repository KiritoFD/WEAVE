#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 _tmp/check_assets.py
echo "--- stage1 ---"
tail -n 6 runs/p2/p0_audit.log 2>/dev/null
tail -n 6 runs/p2/supp_bench.log 2>/dev/null
