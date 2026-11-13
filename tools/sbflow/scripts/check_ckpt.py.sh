#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import torch
p = "runs/v6/N3_dim32/N3_dim32.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
print("top keys:", sorted(ck.keys()))
a = ck.get("args", {})
print("args keys:", sorted(a.keys()))
for k in ("mode", "canon_mode", "base_dim", "no_style", "content_mode",
          "no_inject_64", "no_inject_32", "no_inject_16", "style_keys"):
    print(f"  {k} = {a.get(k)}")
print("n params:", sum(v.numel() for v in ck["model"].values()))
PY
