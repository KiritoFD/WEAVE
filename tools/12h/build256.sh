#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
python3 tools/trt/build_trt_engines.py --sizes 256 --batches 8 --precision fp16 \
  > runs/trt/build_256.log 2>&1
echo "EXIT=$?" >> runs/trt/build_256.log
date >> runs/trt/build_256.log

