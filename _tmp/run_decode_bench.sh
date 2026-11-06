#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "== build log =="
tail -n 5 _tmp/build_big.log
echo "== engines =="
ls -l runs/trt/engines/ | grep 512
echo "== decode bench =="
python3 -u _tmp/decode_bench.py
