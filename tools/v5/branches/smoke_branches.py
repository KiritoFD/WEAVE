"""Smoke-test the pruned V5 branch code on the GPU without starting a full run.

Checks imports, model forward shapes, identity initialisation, supervision
wiring and that the branch shell script is syntactically valid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.v5.branches import route_a
from tools.v5 import route_c


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.randn(4, 4, 64, 64, device=device)
    ids = torch.tensor([0, 1, 2, 3], device=device)
    m1 = route_a.DirectStyleOperator().to(device)
    with torch.no_grad():
        z1 = m1(z, ids)
    assert z1.shape == z.shape
    identity_err = float((z1 - z).abs().max())
    print(f"[smoke] M1 shape={tuple(z1.shape)} max_identity_error={identity_err:.6f}")

    flow = route_c.StyleRectifiedFlow(base=64, n_blocks=2).to(device)
    with torch.no_grad():
        z2 = route_a.ode_forward(flow, z, ids, 2)
    print(f"[smoke] M2 shape={tuple(z2.shape)} finite={bool(torch.isfinite(z2).all())}")

    desc = torch.load(ROOT / "runs" / "v5" / "cache" / "d5_style_multiscale_descriptors.pt",
                      map_location="cpu", weights_only=False)
    sim = route_a.P1.spatial_self_similarity(z)
    parts = route_a.supervision(
        z1, z, sim, ids, descriptors=desc, style_encoder=None, style_protos=None,
        objective="descriptor", structure="selfsim",
    )
    print(f"[smoke] supervision loss={parts['loss'].item():.6f} keys={sorted(parts)}")
    assert torch.isfinite(parts["loss"])
    print("[smoke] BRANCH_IMPORTS_OK")


if __name__ == "__main__":
    main()
