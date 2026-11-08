"""V8 time-reverse / bidirectional-drift symbol test (analytical only)."""
from __future__ import annotations

import torch


def main():
    z0 = torch.tensor([0.0])
    z1 = torch.tensor([2.0])
    tau = torch.tensor([0.25])
    zt = (1 - tau) * z0 + tau * z1
    b_plus_target = (z1 - zt) / (1 - tau)
    b_minus_target = (z0 - zt) / tau
    u_pf = 0.5 * (b_plus_target - b_minus_target)
    assert torch.allclose(u_pf, z1 - z0, atol=1e-5)
    assert float(b_plus_target.item()) > 0 and float(b_minus_target.item()) < 0
    print("V8 TIME REVERSE PASS: u=0.5*(b+ - b-) has correct sign.")


if __name__ == "__main__":
    main()
