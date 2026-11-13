#!/usr/bin/env bash
# P0: zero-training exact compression of the A0 direct operator, with per-image
# output verification in fp32 and fp16 (plan section 4).
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 -u -m tools.v6_24h.fold_direct \
  --ckpt runs/v6/D1_c1_gentle/D1_c1_gentle.pt \
  --out runs/v6_24h/folded/A0_folded.pt
