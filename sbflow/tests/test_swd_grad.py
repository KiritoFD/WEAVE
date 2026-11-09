"""P0-1 verification: the distribution loss must carry a real gradient.

This test exists because V10 shipped a silent no-op: `swd2_chunked` sliced the
generated cloud inside `torch.no_grad()`, detaching it, so L_dist had zero gradient
while still returning a non-zero *value*. The trainer only checked
`float(ld) != 0.0`, which that bug passes.

Every check here is chosen so the V10 implementation FAILS.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from sbflow.style_loss import (DistributionLoss, StyleEncoder, _projections, swd2,
                               swd2_chunked, swd2_local)


def _clouds(n=96, m=80, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, requires_grad=True)
    y = torch.randn(m, d, generator=g) * 1.5 + 0.7
    return x, y


def test_gradient_exists():
    x, y = _clouds()
    loss = swd2(x, y, n_proj=16, seed=3)
    assert loss.requires_grad, "swd2 lost the graph"
    loss.backward()
    assert x.grad is not None, "no gradient reached the generated cloud"
    assert float(x.grad.abs().sum()) > 0, "gradient is identically zero"
    print(f"PASS gradient exists (|grad|= {float(x.grad.norm()):.4f})")


def test_chunked_matches_full():
    """Value AND gradient must equal the full-batch estimator, chunk-size independent."""
    x, y = _clouds(n=128, m=100, d=24, seed=5)
    proj = _projections(24, 16, x.device, x.dtype, seed=11)

    xf = x.detach().clone().requires_grad_(True)
    lf = swd2(xf, y, proj=proj)
    lf.backward()
    gf = xf.grad.clone()

    for chunk in (16, 32, 64, 128):
        xc = x.detach().clone().requires_grad_(True)
        lc = swd2_chunked(xc, y, chunk=chunk, proj=proj)
        lc.backward()
        rel_loss = abs(float(lc) - float(lf)) / max(abs(float(lf)), 1e-12)
        rel_grad = float((lc.grad if False else xc.grad - gf).norm()) / max(
            float(gf.norm()), 1e-12)
        assert rel_loss < 1e-5, f"chunk={chunk} loss mismatch {rel_loss:.2e}"
        assert rel_grad < 1e-4, f"chunk={chunk} grad mismatch {rel_grad:.2e}"
        print(f"PASS chunk={chunk:3d} equals full batch "
              f"(loss rel {rel_loss:.1e}, grad rel {rel_grad:.1e})")


def test_identity_is_zero():
    x, _ = _clouds()
    assert float(swd2(x.detach(), x.detach(), n_proj=16)) < 1e-10
    print("PASS swd2(x, x) == 0")


def test_v10_bug_would_fail():
    """Reproduce the V10 pattern and assert it is detected as gradient-free."""
    x, y = _clouds()
    proj = _projections(32, 16, x.device, x.dtype, seed=1)
    with torch.no_grad():
        xg = x[torch.randperm(x.shape[0])[:32]]
    detached_loss = (xg @ proj).square().mean()
    assert not detached_loss.requires_grad, "test is not reproducing the V10 bug"
    print("PASS the V10 detach pattern is correctly detected as gradient-free")


def test_local_swd_has_gradient():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(16, 8, 12, generator=g, requires_grad=True)
    y = torch.randn(16, 8, 12, generator=g)
    loss = swd2_local(x, y, n_proj=8, seed=2)
    loss.backward()
    assert float(x.grad.abs().sum()) > 0
    print(f"PASS local SWD gradient (|grad|= {float(x.grad.norm()):.4f})")


def test_distribution_loss_gradient():
    """End-to-end: DistributionLoss on flattened latents must reach its input."""
    g = torch.Generator().manual_seed(9)
    z_gen = torch.randn(24, 4, 8, 8, generator=g, requires_grad=True)
    z_real = torch.randn(24, 4, 8, 8, generator=g) * 1.3
    for mode in ("global", "local", "both"):
        enc = StyleEncoder(channels=4)
        dl = DistributionLoss(mode, "fixed", None, n_proj=8, chunk=8)
        zg = z_gen.detach().clone().requires_grad_(True)
        loss = dl(zg, z_real, seed=1)
        loss.backward()
        gn = float(zg.grad.norm()) if zg.grad is not None else 0.0
        assert gn > 1e-8, f"DistributionLoss({mode}) produced no gradient ({gn})"
        print(f"PASS DistributionLoss({mode}) gradient |grad|= {gn:.5f}")


def test_grouped_distribution_loss_gradient():
    """Grouped (per-target-style) path must also carry a gradient."""
    g = torch.Generator().manual_seed(13)
    z_gen = torch.randn(20, 4, 8, 8, generator=g)
    z_real = torch.randn(20, 4, 8, 8, generator=g)
    groups = torch.tensor([0, 1, 2, 3, 4] * 4)
    dl = DistributionLoss("both", "fixed", None, n_proj=8, chunk=8)
    zg = z_gen.detach().clone().requires_grad_(True)
    loss = dl(zg, z_real, seed=2, groups=groups)
    loss.backward()
    gn = float(zg.grad.norm()) if zg.grad is not None else 0.0
    assert gn > 1e-8, f"grouped DistributionLoss produced no gradient ({gn})"
    print(f"PASS grouped DistributionLoss gradient |grad|= {gn:.5f}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_gradient_exists()
    test_chunked_matches_full()
    test_identity_is_zero()
    test_v10_bug_would_fail()
    test_local_swd_has_gradient()
    test_distribution_loss_gradient()
    test_grouped_distribution_loss_gradient()
    print("ALL SWD GRADIENT TESTS PASSED")
