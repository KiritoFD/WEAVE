#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
echo "=== CLI check ==="
python3 tools/v6/train_eval_v6.py --help 2>&1 | sed -n '1,40p'
echo "=== ablation switch smoke (tiny model, no data) ==="
python3 - <<'PY'
import sys, torch
sys.path.insert(0, ".")
from tools.v6.models_v6 import C3UNet
from tools.v6.canonicalize import ContentCanonicalizer
import torch
for nc, ns in ((False, False), (True, False), (False, True)):
    m = C3UNet(base_dim=32, num_styles=5, mode="direct", no_content=nc, no_style=ns)
    z = torch.randn(2, 4, 64, 64); c = torch.randn(2, 4, 64, 64)
    s = torch.tensor([1, 3])
    o = m(z, 0.0, c, s)
    print(f"  no_content={nc} no_style={ns} "
          f"params={sum(p.numel() for p in m.parameters())} "
          f"finite={bool(torch.isfinite(o).all())}")
st = torch.load("runs/v6/cache/canonicalizer_stats.pt", map_location="cpu",
                weights_only=False)
canon = ContentCanonicalizer(st)
z = torch.randn(2, 4, 64, 64)
for mode in ("c1", "c1_gentle", "c1_soft", "none"):
    out = canon(z, mode=mode, augment=False)
    same = bool(torch.allclose(out, z, atol=1e-6))
    print(f"  canon mode={mode:10s} shape={tuple(out.shape)} identical_to_input={same}")
PY
