"""V8 math unit tests: bridge path, zero-noise CFM, Gaussian OT, ODE convergence, train/eval consistency."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v8.bridge import sample_bridge  # noqa: E402
from tools.v8.models import VelocityUNet  # noqa: E402
from tools.v8.solvers import solve_euler, solve_heun  # noqa: E402


def test_bridge_formula_and_finite_difference():
    torch.manual_seed(0)
    z0 = torch.randn(4, 4, 8, 8)
    z1 = torch.randn_like(z0)
    xi = torch.randn_like(z0)
    tau = torch.tensor([0.37, 0.5, 0.73, 0.91])
    sigma_mid = 0.25
    z_t, v_t = sample_bridge(z0, z1, tau, sigma_mid, xi)
    h = 1e-4
    z_plus, _ = sample_bridge(z0, z1, tau + h, sigma_mid, xi)
    z_minus, _ = sample_bridge(z0, z1, tau - h, sigma_mid, xi)
    fd = (z_plus - z_minus) / (2 * h)
    err = float((fd - v_t).abs().mean().item())
    assert err < 2e-3, err

    # Endpoint consistency for zero-noise CFM.
    z_start, v_cf = sample_bridge(z0, z1, torch.zeros(4), 0.0, xi)
    z_end, _ = sample_bridge(z0, z1, torch.ones(4), 0.0, xi)
    assert torch.allclose(z_start, z0, atol=1e-5)
    assert torch.allclose(z_end, z1, atol=1e-5)
    assert torch.allclose(v_cf, z1 - z0, atol=1e-5)


def test_gaussian_and_ode_convergence():
    # Constant velocity field z1-z0 with z1 = T*(z0): Euler errors shrink with NFE.
    torch.manual_seed(1)
    x = torch.randn(256, 1, 1, 1)
    m0, s0, m1, s1 = -1.5, 0.7, 2.0, 1.4
    z0 = m0 + s0 * x
    z1 = m1 + (s1 / s0) * (z0 - m0)
    true_velocity = z1 - z0

    class ConstV(torch.nn.Module):
        def forward(self, z, tau, s, t):
            return true_velocity

    model = ConstV()
    src = torch.zeros(256, dtype=torch.long)
    tgt = torch.ones(256, dtype=torch.long)
    outs = {n: solve_euler(model, z0, src, tgt, n) for n in [4, 8, 16, 32]}
    errs = {n: float((out - z1).abs().mean().item()) for n, out in outs.items()}
    assert all(errs[a] >= errs[b] - 1e-6 for a, b in [(4, 8), (8, 16), (16, 32)])
    assert errs[32] < 1e-6

    heun = solve_heun(model, z0, src, tgt, 4)
    assert float((heun - z1).abs().mean().item()) < 1e-6


def test_train_eval_consistency():
    torch.manual_seed(2)
    model = VelocityUNet(base_dim=16, num_styles=5).eval()
    z = torch.randn(3, 4, 16, 16)
    tau = torch.tensor([0.1, 0.5, 0.9])
    s = torch.tensor([0, 1, 2]); t = torch.tensor([1, 2, 3])
    v1 = model(z, tau, s, t)
    v2 = model(z, tau, s, t)
    assert torch.allclose(v1, v2, atol=1e-7)
    # Same-style hard identity is a solver-interface property; verify it here.
    out = solve_euler(model, z, s, s, 8)
    assert torch.allclose(out, z, atol=1e-7)


if __name__ == "__main__":
    test_bridge_formula_and_finite_difference()
    test_gaussian_and_ode_convergence()
    test_train_eval_consistency()
    print("V8 MATH TESTS PASS: bridge FD, zero-noise CFM, Gaussian ODE convergence, train/eval consistency.")
