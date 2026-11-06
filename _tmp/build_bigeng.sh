#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
ls -l runs/trt/onnx/ 2>/dev/null | head
setsid -f bash -c 'python3 -u tools/trt/build_trt_engines.py --sizes 512 --batches 16,32,64 --precision fp16 --workspace_mb 2048 > _tmp/build_big.log 2>&1' > /dev/null 2>&1 < /dev/null
sleep 5
tail -n 3 _tmp/build_big.log
