"""H2 assertions: DWT<->iDWT < 1e-5; gamma=0 identity; negative gamma lowers
band energy; analytic signed gain respects the clamp."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_dwt_idwt_roundtrip():
    from exp.v11b.bands import dwt2, idwt2
    x = torch.rand(2, 3, 512, 512)
    b = dwt2(x)
    y = idwt2(b['LL2'], b['H2'], b['H1'])
    assert (y - x).abs().max() < 1e-5, (y - x).abs().max()


def test_zero_gamma_identity():
    from exp.v17.h2 import H2Head
    net = H2Head(bias=True).cpu().eval()
    h2 = torch.randn(2, 9, 128, 128)
    edge = torch.rand(2, 1, 128, 128)
    tid = torch.tensor([0, 3])
    out, gamma = net(h2, edge, tid)
    assert gamma.abs().max() == 0 and (out - h2).abs().max() == 0


def test_negative_gamma_reduces_band_energy():
    from exp.v17.h2 import H2Head
    net = H2Head(bias=False).cpu()
    with torch.no_grad():
        last = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)][-1]
        last.weight.fill_(0.)
        last.bias.fill_(0.)
        last.bias[:3] = -2.        # tanh(.5*-2) ~ -.76 -> strong negative gain
    h2 = torch.randn(1, 9, 128, 128)
    out, gamma = net(h2, torch.rand(1, 1, 128, 128), torch.tensor([0]))
    assert out[:, :3].square().mean() < h2[:, :3].square().mean()


def test_signed_gain_clamped():
    from exp.v17.h2 import signed_gain
    rgb = torch.rand(2, 3, 512, 512)
    tid = torch.tensor([0, 1])
    tgt = torch.full((5, 3), 1e6)          # absurd target -> gain clamps
    out = signed_gain(rgb, tid, tgt, gmax=.5)
    assert torch.isfinite(out).all() and out.max() <= 1.
