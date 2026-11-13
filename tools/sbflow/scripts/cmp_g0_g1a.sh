#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import torch, hashlib
from pathlib import Path

def digest(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    m = d["model"]
    h = hashlib.sha256()
    for k in sorted(m):
        h.update(k.encode())
        h.update(m[k].float().numpy().tobytes())
    return h.hexdigest()[:16], len(m)

for a in ("G0_ctrl", "G1a_rank_weak"):
    p = Path(f"runs/v6/{a}/{a}.pt")
    if not p.exists():
        print(f"  {a}: missing"); continue
    d, n = digest(p)
    print(f"  {a:16s} tensors={n} sha={d}")

p0 = Path("runs/v6/G0_ctrl/G0_ctrl.pt")
p1 = Path("runs/v6/G1a_rank_weak/G1a_rank_weak.pt")
if p0.exists() and p1.exists():
    m0 = torch.load(p0, map_location="cpu", weights_only=False)["model"]
    m1 = torch.load(p1, map_location="cpu", weights_only=False)["model"]
    mx = max(float((m0[k].float() - m1[k].float()).abs().max()) for k in m0)
    print(f"\n  max|W_G0 - W_G1a| = {mx:.3e}")
    print("  -> identical weights" if mx == 0.0 else "  -> weights differ")
PY
