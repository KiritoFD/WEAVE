#!/usr/bin/env bash
# Recover rho for a distribution-supervision arm with a 2-update probe.
#
# The probe resumes the SAME prefix with the SAME seed, so the restored RNG reproduces
# the exact first training batch. The step-1 gradient norms are therefore identical to
# what the full run would have recorded, at a cost of a few seconds of GPU.
#
# Used for d1, whose diagnostic was dropped by the microbatch overwrite bug.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=${1:-d1}
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"

for a in ${ARMS//,/ }; do
  OUT=runs/blog_sbflow/v11/probe_rho_$a
  mkdir -p "$OUT"
  echo "=== probe $a ==="
  python3 -u -m sbflow.train5 --config "configs/v11/$a.json" --caches $C \
    --steps 2 --output "$OUT" --log-interval 1 --save-interval 100000 \
    2>&1 | tail -4
  python3 - <<PY
import json
from pathlib import Path
p = Path("$OUT/train_summary.json")
if not p.exists():
    print("  no summary")
else:
    s = json.loads(p.read_text())
    for h in s["loss_history"]:
        if h.get("rho_swd_over_bridge") is not None:
            print(f"  {h['step']}: dist_grad_norm={h['dist_grad_norm']:.6f} "
                  f"bridge_grad_norm={h['bridge_grad_norm']:.6f} "
                  f"rho={h['rho_swd_over_bridge']:.6f}")
            break
    else:
        print("  rho still missing")
PY
done
