"""V8 ODE convergence test: constant velocity Euler 4/8/16/32."""
from __future__ import annotations

import torch

from tools.v8.solvers import solve_euler


class ConstV(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, z, tau, s, t):
        return self.v


def main():
    torch.manual_seed(0)
    z0 = torch.randn(64, 4, 8, 8)
    z1 = torch.randn_like(z0)
    model = ConstV(z1 - z0)
    s = torch.zeros(64, dtype=torch.long)
    t = torch.ones(64, dtype=torch.long)
    errs = {}
    errs_base = []
    for n in [4, 8, 16, 32]:
        z = solve_euler(model, z0, s, t, n)
        errs[n] = float((z - z1).abs().mean().item())
        errs_base.append(errs[n])
    # For a constant velocity field all NFEs are exact; only require numerical
    # convergence up to floating point noise, not strict monotonicity.
    assert max(errs_base) < 1e-6
    assert errs[32] < 1e-6
    print("V8 ODE PASS (constant-v exact):", errs)


if __name__ == "__main__":
    main()
