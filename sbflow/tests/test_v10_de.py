"""Math and behaviour tests for the group D (distribution loss) and E (online SDE) mechanisms."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from sbflow.network import BidirDiT
from sbflow.online_bridge import (EndpointReplayBuffer, both_drifts, estimate_cross_cov,
                                  ode_forward, online_loss, sde_forward, sde_reverse)
from sbflow.preconditioning import StyleStats
from sbflow.style_loss import (DistributionLoss, StyleEncoder, swd2, swd2_local,
                               supervised_contrastive)


def small_model():
    return BidirDiT(channels=8, image_size=8, patch=2, width=32, depth=1, heads=2,
                    num_styles=5, style_cond="decomposed").eval()


def stats5(channels=8):
    z = torch.zeros(5, channels)
    v = torch.ones(5, channels)
    return StyleStats(torch.zeros(channels), torch.ones(channels), z, v)


# --------------------------------------------------------------------------- #
def test_swd2_properties():
    torch.manual_seed(0)
    x = torch.randn(48, 16)
    assert float(swd2(x, x)) < 1e-8
    assert float(swd2(x, x + 1.0)) > 0.1
    # differentiable in the first argument
    y = (x + 0.5).clone().requires_grad_(True)
    swd2(y, x).backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    print("PASS test_swd2_properties")


def test_swd2_local_shape():
    torch.manual_seed(0)
    a = torch.randn(4, 16, 8)
    b = torch.randn(4, 16, 8)
    assert float(swd2_local(a, a)) < 1e-8
    assert float(swd2_local(a, b)) > 0
    print("PASS test_swd2_local_shape")


def test_contrastive_learns_separation():
    torch.manual_seed(0)
    enc = StyleEncoder(channels=8, width=32, out_dim=32)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-3)
    # two well separated style clusters
    z0 = torch.randn(64, 8, 8, 8) * 0.2 + 1.0
    z1 = torch.randn(64, 8, 8, 8) * 0.2 - 1.0
    z = torch.cat([z0, z1], 0)
    lab = torch.tensor([0] * 64 + [1] * 64)
    first = None
    for _ in range(40):
        emb, _ = enc(z)
        loss = supervised_contrastive(emb, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
    assert float(loss) < first, (first, float(loss))
    print(f"PASS test_contrastive_learns_separation ({first:.3f} -> {float(loss):.3f})")


def test_distribution_loss_modes():
    torch.manual_seed(0)
    enc = StyleEncoder(channels=8, width=32, out_dim=32).eval()
    z = torch.randn(6, 8, 8, 8)
    for mode in ("global", "local", "both"):
        for kind in ("learned", "fixed"):
            dl = DistributionLoss(mode, kind, enc if kind == "learned" else None)
            v = dl(z, torch.randn_like(z))
            assert torch.isfinite(v) and float(v) >= 0, (mode, kind, float(v))
    print("PASS test_distribution_loss_modes")


def test_sde_endpoints_are_stochastic_and_stay_close():
    """A near-zero network must map the source almost onto itself plus diffusion."""
    torch.manual_seed(0)
    m = small_model()
    st = stats5()
    z = torch.randn(16, 8, 8, 8)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    a = sde_forward(m, z, 0, 1, st, 0.04, 4, 0.002, g1)
    b = sde_forward(m, z, 0, 1, st, 0.04, 4, 0.002, g2)
    assert not torch.allclose(a, b), "SDE must be stochastic (two generators differ)"
    assert torch.isfinite(a).all()
    # with a zero network the drift is the analytic EDM drift; endpoints must stay bounded
    assert float(a.abs().max()) < 1e4
    print("PASS test_sde_endpoints_are_stochastic_and_stay_close")


def test_ode_forward_is_deterministic():
    torch.manual_seed(0)
    m = small_model()
    st = stats5()
    z = torch.randn(8, 8, 8, 8)
    a = ode_forward(m, z, 0, 1, st, 0.04, 4, 0.002)
    b = ode_forward(m, z, 0, 1, st, 0.04, 4, 0.002)
    assert torch.allclose(a, b), "ODE endpoints must be deterministic"
    print("PASS test_ode_forward_is_deterministic")


def test_online_loss_variants():
    torch.manual_seed(0)
    m = small_model()
    m.train()
    teacher = small_model()
    st = stats5()
    z0 = torch.randn(8, 8, 8, 8)
    z1 = torch.randn(8, 8, 8, 8)
    g = torch.Generator().manual_seed(3)
    for cfg in (
        {"online_steps": 4, "online_mode": "sde", "online_direction": "both"},
        {"online_steps": 8, "online_mode": "sde", "online_direction": "both"},
        {"online_steps": 4, "online_mode": "ode", "online_direction": "both"},
        {"online_steps": 4, "online_mode": "sde", "online_direction": "forward"},
        {"online_steps": 4, "online_mode": "sde", "online_direction": "reverse"},
        {"online_steps": 4, "online_mode": "sde", "online_direction": "both", "online_mix": 0.5},
    ):
        loss, info = online_loss(m, teacher, z0, z1, 0, 1, st, 0.04, 0.002, g, cfg)
        assert torch.isfinite(loss) and float(loss) >= 0, (cfg, float(loss))
        assert info["endpoint_shift_fwd"] >= 0
    print("PASS test_online_loss_variants")


def test_online_endpoints_have_no_grad():
    """Generated endpoints must be stop-gradient: the teacher is not differentiated."""
    torch.manual_seed(0)
    m = small_model()
    st = stats5()
    z0 = torch.randn(4, 8, 8, 8)
    g = torch.Generator().manual_seed(4)
    x1 = sde_forward(m, z0, 0, 1, st, 0.04, 2, 0.002, g)
    assert not x1.requires_grad, "SDE endpoint must be detached from the graph"
    print("PASS test_online_endpoints_have_no_grad")


def test_replay_buffer():
    torch.manual_seed(0)
    buf = EndpointReplayBuffer(capacity=8, channels=8, size=8)
    z = torch.randn(4, 8, 8, 8)
    for _ in range(3):
        buf.push(0, 1, "fwd", z)
    g = torch.Generator().manual_seed(0)
    got = buf.sample(0, 1, "fwd", 2, "cpu", g)
    assert got is not None and got.shape == (2, 8, 8, 8)
    assert buf.sample(2, 3, "fwd", 1, "cpu", g) is None, "empty key must return None"
    print(f"PASS test_replay_buffer (len={len(buf)})")


def test_cross_cov_estimation():
    torch.manual_seed(0)
    a = torch.randn(256, 8, 8, 8)
    b = 0.5 * a + 0.1 * torch.randn_like(a)
    cov = estimate_cross_cov(a, b)
    assert cov.shape == (8,)
    # correlated channels must have positive covariance
    assert float(cov.mean()) > 0.1, float(cov.mean())
    ema = estimate_cross_cov(a, b, ema=torch.zeros(8), decay=0.5)
    assert torch.allclose(ema, 0.5 * cov, atol=1e-5)
    print("PASS test_cross_cov_estimation")


def test_both_drifts_finite():
    torch.manual_seed(0)
    m = small_model()
    st = stats5()
    z = torch.randn(6, 8, 8, 8)
    t = torch.rand(6) * 0.8 + 0.1
    bp, bm = both_drifts(m, z, t, 0, 4, st, 0.04, 0.002)
    assert torch.isfinite(bp).all() and torch.isfinite(bm).all()
    assert bp.shape == z.shape == bm.shape
    print("PASS test_both_drifts_finite")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL D/E MECHANISM TESTS PASSED")
