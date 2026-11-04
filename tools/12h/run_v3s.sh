#!/usr/bin/env bash
# V3-SWD 一键训练 + 四个评测点（512 口径）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SB_STRICT_CKPT=1
export OMP_NUM_THREADS=4

STEPS="${STEPS:-4000}"
python3 -m v2.train --config configs/v2/V3S.json --variant V3S --steps "$STEPS"

for s in 500 1000 2000 4000; do
  ckpt="runs/v2/V3S/step_$(printf '%06d' "$s").pt"
  if [ ! -f "$ckpt" ]; then
    echo "[v3s] missing $ckpt, skip"
    continue
  fi
  bash tools/12h/eval_path_trt.sh "v3s_step${s}_512" "$ckpt" "runs/12h/v3s/step_$(printf '%06d' "$s")_750_512" 8
done
