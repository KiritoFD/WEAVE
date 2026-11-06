#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 -u tools/trt/build_trt_engines.py --sizes 512 --batches 16,32,64 --precision fp16 --workspace_mb 6144
echo "== engines =="
ls -l runs/trt/engines/ | grep 512
