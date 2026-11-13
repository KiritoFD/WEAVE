#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import torch
# what dtype does the cached eval latent use? that is what canon expects
lat = torch.load("runs/v5/cache/eval_src_latents.pt", map_location="cpu")
k = next(iter(lat))
print("eval_src_latents dtype:", lat[k].dtype, "shape:", tuple(lat[k].shape))

st = torch.load("runs/v6/cache/canonicalizer_stats.pt", map_location="cpu", weights_only=False)
print("canonicalizer stats dtype:", next(iter(st.values())).dtype)
PY
