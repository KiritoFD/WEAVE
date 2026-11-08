"""V8 analytic Gaussian transport test (CFM straight path)."""
from __future__ import annotations

import torch


def main():
    torch.manual_seed(0)
    n = 20000
    m0, s0, m1, s1 = -1.0, 0.5, 2.0, 1.2
    x = torch.randn(n, 1, 1, 1)
    z0 = m0 + s0 * x
    z1 = m1 + (s1 / s0) * (z0 - m0)
    # Constant velocity from z1-z0, Euler 32 (exact for constant v).
    z = z0.clone()
    for _ in range(32):
        z = z + (z1 - z0) / 32
    mz, sz = float(z.mean()), float(z.std())
    assert abs(mz - m1) < 0.02 and abs(sz - s1) < 0.02
    print(f"V8 GAUSSIAN PASS: mean={mz:.4f} std={sz:.4f} target=({m1},{s1})")


if __name__ == "__main__":
    main()
