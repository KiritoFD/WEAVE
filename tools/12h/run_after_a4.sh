#!/usr/bin/env bash
# A4 on the final fixed path (fast b8 + 256 fp16 TRT), then the A1/A2 curves.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
bash tools/12h/eval_path_trt.sh A4_fast_trt256_b8 runs/12h/A4/step_002000.pt runs/12h/eval_A4_fast_trt 8
echo "A4_DONE"
DECODER=trt bash tools/12h/run_curves.sh
echo "CURVES_DONE"

