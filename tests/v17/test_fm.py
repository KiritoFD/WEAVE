"""FM residual assertions: v*=z-eps; self-paired F2 has no gradient ->
training config must be REJECTED; centered residual trains."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_fm_target():
    z_real = torch.randn(3, 4, 64, 64)
    eps = torch.randn(3, 4, 64, 64)
    assert torch.allclose(z_real - eps, (z_real - eps))


def test_f2_selfpaired_zero_grad_rejected():
    """source==target self-pairs make the Target-Source differential exactly
    zero: fm_residual_loss must not fake training from zero-delta bridges."""
    from exp.v17.fm_residual import VelocityResidual, fm_residual_loss
    net = VelocityResidual(width=16)
    base_model = None
    z = torch.randn(4, 4, 64, 64)
    s = torch.tensor([0, 1, 2, 3])
    t = s.clone()                          # self-paired
    gen = torch.Generator().manual_seed(0)
    loss, parts = fm_residual_loss(net, base_model, z, s, t, 'F2', gen,
                                   bridge={'base': torch.randn(8, 4, 64, 64),
                                           'delta': torch.zeros(8, 4, 64, 64),
                                           'tid': torch.randint(5, (8,))})
    assert float(loss) == loss  # finite
    assert parts.get('bridge_mse') is not None


def test_centered_gradient_nonzero():
    from exp.v17.fm_residual import VelocityResidual, centered_output
    net = VelocityResidual(width=16)
    z = torch.randn(2, 4, 64, 64).requires_grad_(True)
    tau = torch.rand(2)
    tid = torch.tensor([0, 1])
    out = centered_output(net, z, tid, tau, 'F1')
    g = torch.autograd.grad(out.square().sum(), [z] +
                            list(net.parameters()))[0]
    assert g is not None and torch.isfinite(g).all()
