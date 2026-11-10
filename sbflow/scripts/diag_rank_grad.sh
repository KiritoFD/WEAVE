#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi

echo "=== G1a 2 steps, FULL output, looking for the step-1 gradient ratio ==="
python3 -u tools/v6/train_eval_v6.py --name G1a_diag --mode direct --base-dim 16 \
  --steps 2 --microbatch 80 --accum 1 --seed 42 --canon-mode c1_gentle \
  --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 --ref-top-k 1024 \
  --init-from runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt \
  --rank-critic runs/v6_next/probes/r1_r1.pt --rank-lambda 0.1 --rank-tau 0.5 2>&1 \
  | grep -aE "init_from|rank_critic|G1 diag|step 1/|step 2/|Conditioning|Eval200" | head -20

echo
echo "=== report keys available ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/v6/G1a_smoke/report.json")
if p.exists():
    d = json.loads(p.read_text())
    print("  top:", list(d.keys()))
    e = d.get("eval200") or {}
    print("  eval200 keys:", list(e.keys()))
PY
