"""Unit tests for V29 big.LITTLE PCoreSLFUNet, gamma upgrade, and trajectory schedule."""
from __future__ import annotations

import torch

from exp.v29_biglittle.pcore_model import PCoreSLFUNet, match_raw_k8_submode
from exp.v29_biglittle.phase1_train import _blur_or_drop_y16, _build_style_counts, _pseudo_huber_velocity
from exp.v29_biglittle.phase2_tune import default_knob_schedule


def test_pcore_shapes_and_gamma_upgrade():
    m = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=False).eval()
    assert 20_000_000 <= m.n_params() <= 24_000_000
    m4 = m.upgrade_with_gamma().eval()
    assert m4.use_gamma is True

    z = torch.randn(2, 4, 64, 64)
    y16 = torch.randn(2, 1, 16, 16)
    q32 = torch.randn(2, 32, 16, 16)
    tid = torch.tensor([0, 4], dtype=torch.long)
    u = torch.randn(2, 417)
    tau = torch.tensor([0.2, 0.8])

    with torch.no_grad():
        v0 = m(z, y16, q32, tid, u, tau)
        v1 = m4(z, y16, q32, tid, u, tau, gamma=torch.ones(2))
        v_low = m4(z, y16, q32, tid, u, tau, gamma=torch.zeros(2))
        m_norm = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=False, normalize_u=True).eval()
        m_norm.load_state_dict(m.state_dict(), strict=True)
        v_norm = m_norm(z, y16, q32, tid, u * 10.0, tau)
    assert v0.shape == (2, 4, 64, 64)
    assert torch.allclose(v0, v1, atol=1e-5)
    assert v_low.shape == (2, 4, 64, 64)
    assert v_norm.shape == (2, 4, 64, 64) and torch.isfinite(v_norm).all()
    assert torch.allclose(v_norm, m_norm(z, y16, q32, tid, u, tau), atol=1e-5)


def test_raw_k8_matcher_and_losses():
    bank = {
        "sub_scene": torch.randn(5, 8, 6144),
        "sub_raw417": torch.randn(5, 8, 417),
    }
    d16 = torch.randn(3, 384, 16, 16)
    tid = torch.tensor([0, 2, 4], dtype=torch.long)
    u = match_raw_k8_submode(d16, tid, bank)
    assert u.shape == (3, 417)

    y16 = torch.randn(16, 1, 16, 16)
    y_aug, keep_m = _blur_or_drop_y16(y16, drop_prob=0.40)
    assert y_aug.shape == y16.shape and keep_m.dtype == torch.bool

    ph = _pseudo_huber_velocity(torch.ones(4, 4, 64, 64), torch.zeros(4, 4, 64, 64), delta=0.05)
    assert ph.ndim == 0 and float(ph) > 0.9

    counts = _build_style_counts(128)
    assert sum(counts) == 128 and counts[0] > counts[1] and counts[4] > counts[2]


def test_knob_schedule_monotonic():
    pts = [default_knob_schedule(s, use_gamma=True) for s in (0.65, 0.75, 0.85, 1.00, 1.15, 1.25, 1.35, 1.50)]
    assert all(pts[i]["tau0"] >= pts[i + 1]["tau0"] for i in range(len(pts) - 1))
    assert all(pts[i]["draft_lambda"] <= pts[i + 1]["draft_lambda"] for i in range(len(pts) - 1))
    assert all(p["rescale_phi"] == 0.0 for p in pts[:5])
    assert all(p["rescale_phi"] > 0.0 for p in pts[5:])
    assert pts[1]["draft_lambda"] == 0.6 and pts[2]["draft_lambda"] == 0.75 and pts[3]["draft_lambda"] == 1.0
    assert pts[-1]["tau0"] == 0.0 and pts[-1]["gamma"] < 0.30 and pts[-1]["ws"] >= 5.0
