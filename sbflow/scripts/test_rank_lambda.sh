#!/usr/bin/env bash
# Does lambda_r=0.001 change the weights at all? Use the EXACT settings of the real arms
# (microbatch 80, accum 3) but only 4 steps, and compare against a no-rank twin.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
C="--mode direct --base-dim 16 --steps 4 --microbatch 80 --accum 3 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --init-from $INIT"

python3 -u tools/v6/train_eval_v6.py --name A4_noRank $C 2>&1 | tail -n 2
python3 -u tools/v6/train_eval_v6.py --name A4_r0001 $C \
  --rank-critic "$CRITIC" --rank-lambda 0.001 --rank-tau 0.5 2>&1 | tail -n 2
python3 -u tools/v6/train_eval_v6.py --name A4_r05 $C \
  --rank-critic "$CRITIC" --rank-lambda 0.05 --rank-tau 0.5 2>&1 | tail -n 2

python3 - <<'PY'
import torch
from pathlib import Path
def load(n):
    return torch.load(f"runs/v6/{n}/{n}.pt", map_location="cpu", weights_only=False)["model"]
a = load("A4_noRank")
for name in ("A4_r0001", "A4_r05"):
    p = Path(f"runs/v6/{name}/{name}.pt")
    if not p.exists():
        print(f"  {name}: missing"); continue
    b = load(name)
    mx = max(float((a[k].float()-b[k].float()).abs().max()) for k in a)
    nd = sum(1 for k in a if float((a[k].float()-b[k].float()).abs().max()) > 0)
    print(f"  {name:12s} max|dW|={mx:.3e} tensors_differing={nd}/{len(a)}")
PY
