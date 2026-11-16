"""Loss identities: E_rel = 1 + a^2 - 2 a rho; zero-teacher safety; Phase II
gate only touches the norm term."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_relative_mse_identity():
    from exp.v17.losses import relative_mse_identity
    g = torch.Generator().manual_seed(7)
    u = torch.nn.functional.normalize(torch.randn(4, 64, 64, generator=g),
                                      dim=0)
    delta = .5 * u                                       # parallel case
    ok, e_true, e_ref = relative_mse_identity(delta, u)
    assert ok, (e_true, e_ref)
    assert abs(float(e_true) - .25) < 1e-4               # (0.5-1)^2
    w = torch.nn.functional.normalize(torch.randn(4, 64, 64, generator=g),
                                      dim=0)
    t = 2. * (1.5 * u + .8 * w) / 1.7
    delta2 = .7 * u
    ok, e_true, e_ref = relative_mse_identity(delta2, t)
    assert ok, (e_true, e_ref)


def test_zero_teacher_uses_mse_only():
    from exp.v17.losses import student_loss
    delta = torch.randn(2, 4, 64, 64, requires_grad=True)
    zero = torch.zeros(2, 4, 64, 64)
    loss, parts = student_loss(delta, zero, phase2=True, rho=torch.zeros(2))
    assert 'cos' not in parts, 'zero-teacher samples must skip the cosine term'
    loss.backward()
    assert torch.isfinite(delta.grad).all()


def test_phase2_gate_only_norm():
    from exp.v17.losses import student_loss
    torch.manual_seed(3)
    delta = torch.randn(2, 4, 64, 64, requires_grad=True)
    t = torch.randn(2, 4, 64, 64)
    l1, p1 = student_loss(delta, t, phase2=True, lam_norm=.1)
    g1 = torch.autograd.grad(l1, delta, retain_graph=True)[0]
    l2, p2 = student_loss(delta, t, phase2=False, lam_norm=.1)
    g2 = torch.autograd.grad(l2, delta)[0]
    assert 'rel_norm' in p1 and 'rel_norm' not in p2
    assert not torch.allclose(g1, g2), 'phase2 must add the norm gradient'


def test_bce_shape():
    from exp.v17.losses import student_loss
    delta = torch.randn(1, 4, 64, 64)
    t = torch.randn(1, 4, 64, 64)
    logits = torch.randn(1, 1, 64, 64)
    mask = (torch.rand(1, 1, 64, 64) > .5).float()
    loss, parts = student_loss(delta, t, logits, mask)
    assert torch.isfinite(loss)
