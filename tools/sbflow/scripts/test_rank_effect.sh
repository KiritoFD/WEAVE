#!/usr/bin/env bash
# Decisive minimal test: 3 steps with and without the ranking term, NO diagnostic probes at all.
# If the weights differ, the mechanism works and only the earlier diagnostic was broken.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
C="--mode direct --base-dim 16 --steps 3 --microbatch 40 --accum 1 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --init-from $INIT"

python3 -u tools/v6/train_eval_v6.py --name T_noRank $C 2>&1 | tail -n 3
python3 -u tools/v6/train_eval_v6.py --name T_withRank $C \
  --rank-critic "$CRITIC" --rank-lambda 0.05 --rank-tau 0.5 2>&1 | tail -n 3

python3 - <<'PY'
import torch
from pathlib import Path
a = torch.load("runs/v6/T_noRank/T_noRank.pt", map_location="cpu", weights_only=False)["model"]
b = torch.load("runs/v6/T_withRank/T_withRank.pt", map_location="cpu", weights_only=False)["model"]
mx = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
diff = sum(1 for k in a if float((a[k].float() - b[k].float()).abs().max()) > 0)
print(f"\nmax|W_norank - W_withrank| = {mx:.6e}   tensors_differing={diff}/{len(a)}")
print("  -> RANKING TERM HAS NO EFFECT" if mx == 0.0 else "  -> RANKING TERM DOES AFFECT WEIGHTS")
PY
