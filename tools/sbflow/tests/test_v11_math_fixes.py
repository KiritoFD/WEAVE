"""P0-5 / P0-6 verification: reverse-time sign and Monge mean collapse."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from tools.sbflow.train5 import score_current_targets


def test_score_current_reverse_sign():
    """v* must flip sign between forward and reverse; s* must NOT."""
    g = torch.Generator().manual_seed(0)
    B, C, H, W = 8, 4, 8, 8
    z0 = torch.randn(B, C, H, W, generator=g)
    z1 = torch.randn(B, C, H, W, generator=g) + 1.5
    t = torch.full((B,), 0.4)
    xi = torch.randn(B, C, H, W, generator=g)
    eps = 0.04

    xt, v_f, s_f = score_current_targets(z0, z1, t, xi, eps,
                                         torch.ones(B, dtype=torch.long))
    _, v_r, s_r = score_current_targets(z0, z1, t, xi, eps,
                                        torch.zeros(B, dtype=torch.long))

    assert torch.allclose(v_f, -v_r, atol=1e-6), "velocity did not flip sign in reverse"
    assert torch.allclose(s_f, s_r, atol=1e-6), "score incorrectly flipped sign"
    assert float(v_f.abs().sum()) > 0, "velocity target is identically zero"
    print(f"PASS score_current: v flips sign (max|v|= {float(v_f.abs().max()):.4f}), "
          f"score unchanged (max|s|= {float(s_f.abs().max()):.2f})")


def test_score_current_reduces_to_velocity_at_eps0():
    """At eps -> 0 the velocity target must equal z1 - z0 (the CFM velocity)."""
    g = torch.Generator().manual_seed(1)
    B, C, H, W = 4, 2, 4, 4
    z0 = torch.randn(B, C, H, W, generator=g)
    z1 = torch.randn(B, C, H, W, generator=g) + 1.0
    t = torch.full((B,), 0.3)
    xi = torch.zeros(B, C, H, W)
    _, v, _ = score_current_targets(z0, z1, t, xi, 1e-7,
                                    torch.ones(B, dtype=torch.long))
    err = float((v - (z1 - z0)).abs().max())
    assert err < 1e-4, f"eps->0 velocity mismatch {err:.2e}"
    print(f"PASS score_current -> CFM velocity at eps->0 (max err {err:.1e})")


def test_monge_requires_ot():
    """objective=monge without OT coupling must refuse to train (mean collapse)."""
    from tools.sbflow.train5 import objective_mixed

    class Dummy(torch.nn.Module):
        def forward(self, x, tau, d, a, b, **kw):
            return torch.zeros_like(x)

    g = torch.Generator().manual_seed(2)
    B = 4
    z0 = torch.randn(B, 4, 4, 4, generator=g)
    z1 = torch.randn(B, 4, 4, 4, generator=g)
    pa = torch.zeros(B, dtype=torch.long)
    pb = torch.ones(B, dtype=torch.long)
    dirs = torch.ones(B, dtype=torch.long)

    class Args:
        objective = "monge"
        time_sampling = "uniform"
        strata = 8
        cycle_weight = 0.0
        monge_ot = False

    try:
        objective_mixed(Dummy(), z0, z1, pa, pb, dirs, None, 0.04, 0.002, g, Args())
    except ValueError as e:
        assert "monge" in str(e).lower()
        print("PASS objective=monge without --monge-ot is rejected")
        return
    raise AssertionError("monge with independent endpoints was silently accepted")


def test_independent_pairing_collapses():
    """Numerical demonstration of theory 6.2: independent pairs -> T = E[z1]."""
    torch.manual_seed(0)
    n = 4000
    z0 = torch.randn(n, 1)
    z1 = torch.randn(n, 1) * 0.5 + 2.0          # target mean 2.0

    # Fit a constant map (extreme version of a collapsed network) both ways.
    indep_const = float(z1.mean())

    # OT/sorted coupling: pair by rank, then the conditional mean is ~ z0*0.5 + 2.0
    z0s, idx = torch.sort(z0, dim=0)
    z1s, _ = torch.sort(z1, dim=0)
    slope = float(((z0s - z0s.mean()) * (z1s - z1s.mean())).sum() / ((z0s - z0s.mean()) ** 2).sum())

    assert abs(indep_const - 2.0) < 0.1, f"independent optimum should be mu_j, got {indep_const}"
    assert abs(slope - 0.5) < 0.15, f"rank-coupled slope should recover 0.5, got {slope}"
    print(f"PASS independent pairing collapses to mu_j={indep_const:.3f}; "
          f"rank coupling recovers slope={slope:.3f} (true 0.5)")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_score_current_reverse_sign()
    test_score_current_reduces_to_velocity_at_eps0()
    test_monge_requires_ot()
    test_independent_pairing_collapses()
    print("ALL V11 MATH FIX TESTS PASSED")
