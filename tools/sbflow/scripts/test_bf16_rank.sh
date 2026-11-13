#!/usr/bin/env bash
# Verify the fp16-underflow diagnosis: with bf16 the small lambda should now reach the weights.
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

echo "########## bf16 control (no rank) ##########"
python3 -u tools/v6/train_eval_v6.py --name B4_noRank $C --amp-dtype bf16 2>&1 | grep -aE "Eval200|Error" | head -2
for LAM in 0.001 0.003 0.01; do
  echo "########## bf16 lambda=$LAM ##########"
  python3 -u tools/v6/train_eval_v6.py --name "B4_lam${LAM}" $C --amp-dtype bf16 \
    --rank-critic "$CRITIC" --rank-lambda "$LAM" --rank-tau 0.5 2>&1 \
    | grep -aE "RANKLOG|Eval200|Error" | head -3
done

python3 - <<'PY'
import torch
from pathlib import Path
base = torch.load("runs/v6/B4_noRank/B4_noRank.pt", map_location="cpu",
                  weights_only=False)["model"]
print(f"\n{'amp':>6s}{'lambda':>9s}{'max|dW|':>14s}{'tensors':>12s}")
for L in ("0.001", "0.003", "0.01"):
    p = Path(f"runs/v6/B4_lam{L}/B4_lam{L}.pt")
    if not p.exists():
        print(f"{'bf16':>6s}{L:>9s}   missing"); continue
    b = torch.load(p, map_location="cpu", weights_only=False)["model"]
    mx = max(float((base[k].float()-b[k].float()).abs().max()) for k in base)
    nd = sum(1 for k in base if float((base[k].float()-b[k].float()).abs().max()) > 0)
    print(f"{'bf16':>6s}{L:>9s}{mx:14.3e}{nd:6d}/{len(base)}")
PY
