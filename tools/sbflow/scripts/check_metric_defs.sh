#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import numpy as np, torch, sys
sys.path.insert(0, ".")
from tools.v5.route_c import csd_prototypes
p = np.asarray(csd_prototypes(), dtype=np.float32)
print("proto shape", p.shape)
n = np.linalg.norm(p, axis=1)
print("proto row norms:", np.round(n, 6).tolist())
print("=> unit-norm:", bool(np.allclose(n, 1.0, atol=1e-4)))
print("=> csd @ proto.T is a COSINE SIMILARITY" if np.allclose(n, 1.0, atol=1e-4)
      else "=> NOT unit-norm, dot product is unnormalised")
print()
print("class counts from the five latent caches:")
import glob
for f in sorted(glob.glob("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    print(f"  {f.split('/')[-1]:32s} count={d['count']}")
PY
