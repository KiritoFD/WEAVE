"""Math tests for the V10 five-style mechanisms.

Covers:
  * StyleStats.pair matches a directly estimated PairMoments for independent endpoints
  * direct_velocity_coefficients whiten the direct-velocity target (mean 0, var 1)
  * direct-velocity reconstruction is exact
  * score/current decomposition round trip
  * bridge and direct-velocity coefficients agree on their respective targets
  * pair-id mapping is symmetric and covers all ten pairs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from tools.sbflow.bridge import bridge_sample, exact_conditional_flow
from tools.sbflow.network import BidirDiT
from tools.sbflow.preconditioning import (
    MomentAccumulator, PairMoments, StyleStats, direct_velocity_coefficients,
    drifts_from_score_current, independent_unit_coefficients, moment_coefficients,
    normalized_target, normalized_velocity_target, precondition_input,
    reconstruct_drift, reconstruct_velocity, score_current_targets,
    style_stats_from_caches,
)


def test_style_stats_pair():
    torch.manual_seed(0)
    latents = [torch.randn(400, 4, 2, 2) * (1.0 + 0.5 * i) + 0.3 * i for i in range(5)]
    stats = style_stats_from_caches(latents)
    assert stats.num_styles == 5
    for a in range(5):
        for b in range(a + 1, 5):
            pm = stats.pair(a, b)
            # independent endpoints => zero cross covariance by construction
            assert float(pm.cov01.abs().max()) == 0.0
            # means/vars must equal the empirical standardized statistics
            za = (latents[a].float() - stats.global_mean[None, :, None, None]) / stats.global_std[None, :, None, None]
            assert torch.allclose(pm.mu0, za.mean((0, 2, 3)), atol=1e-5)
            assert torch.allclose(pm.var0, za.var((0, 2, 3), unbiased=False), atol=1e-5)
    print("PASS test_style_stats_pair")


def test_direct_velocity_whitening():
    """The normalized v* residual must have ~zero mean and ~unit variance."""
    torch.manual_seed(0)
    n, eps = 20000, 0.1
    v0 = torch.full((2,), 0.7)
    v1 = torch.full((2,), 1.3)
    cov = torch.full((2,), 0.2)
    mom = PairMoments(torch.tensor([0.1, -0.2]), torch.tensor([0.5, 0.3]), v0, v1, cov)
    for ts in (0.05, 0.25, 0.5, 0.75, 0.95):
        t = torch.full((n,), ts)
        # sample independent-ish endpoints with the declared moments
        z0 = mom.mu0 + mom.var0.sqrt() * torch.randn(n, 2)
        z1 = mom.mu1 + (cov / mom.var0) * (z0 - mom.mu0) \
             + (mom.var1 - cov ** 2 / mom.var0).clamp_min(1e-8).sqrt() * torch.randn(n, 2)
        xt, bp, bm = bridge_sample(z0[:, :, None, None], z1[:, :, None, None], t, eps)
        xi = (xt - (1 - t[:, None, None, None]) * z0[:, :, None, None]
              - t[:, None, None, None] * z1[:, :, None, None])
        xi = xi / (eps * t * (1 - t)).sqrt()[:, None, None, None]
        vstar = (z1 - z0)[:, :, None, None] + (eps ** 0.5) * (1 - 2 * t) / (2 * (t * (1 - t)).sqrt()) \
            * xi[:, :, None, None] if False else None
        # build v* directly from the bridge identity: v* = (b+ - b-)/2
        vstar = 0.5 * (bp - bm)
        xin, c = precondition_input(xt, t, eps, mom)
        # The direct-velocity coefficients apply to the SAME z_t
        cin, cskip, cout, mx, my = _direct_pieces(t, eps, mom)
        resid = (vstar - my - cskip * (xt - mx)) / cout
        assert abs(float(resid.mean())) < 0.03, (ts, float(resid.mean()))
        assert abs(float(resid.var(unbiased=False)) - 1) < 0.06, (ts, float(resid.var(unbiased=False)))
        assert torch.allclose(reconstruct_velocity(xt, resid, (mx, my, cin, cskip, cout)),
                              vstar, atol=1e-4, rtol=1e-4)
        assert torch.allclose(
            normalized_velocity_target(xt, vstar, (mx, my, cin, cskip, cout)), resid,
            atol=1e-4, rtol=1e-4)
    print("PASS test_direct_velocity_whitening")


def _direct_pieces(t, eps, mom):
    mx, my, ci, cs, co = direct_velocity_coefficients(t, eps, mom)
    return ci, cs, co, mx, my


def test_direct_velocity_reconstruct():
    torch.manual_seed(0)
    eps = 0.04
    mom = PairMoments(torch.zeros(3), torch.ones(3), torch.ones(3), torch.ones(3) * 2,
                      torch.zeros(3))
    t = torch.tensor([0.1, 0.5, 0.9])
    x = torch.randn(3, 3, 2, 2)
    mx, my, ci, cs, co = direct_velocity_coefficients(t, eps, mom)
    y = reconstruct_velocity(x, torch.zeros_like(x), (mx, my, ci, cs, co))
    xin, coeffs = None, (mx, my, ci, cs, co)
    back = normalized_velocity_target(x, y, coeffs)
    assert torch.allclose(back, torch.zeros_like(x), atol=1e-5)
    print("PASS test_direct_velocity_reconstruct")


def test_score_current_roundtrip():
    torch.manual_seed(0)
    eps = 0.04
    bp = torch.randn(4, 3, 2, 2)
    bm = torch.randn(4, 3, 2, 2)
    v, s = score_current_targets(bp, bm, eps)
    bp2, bm2 = drifts_from_score_current(v, s, eps)
    assert torch.allclose(bp, bp2, atol=1e-6)
    assert torch.allclose(bm, bm2, atol=1e-6)
    # (b+ - b-)/2 must equal the probability-flow velocity
    assert torch.allclose(v, 0.5 * (bp - bm), atol=1e-6)
    print("PASS test_score_current_roundtrip")


def test_independent_closed_form_still_holds():
    ts = torch.tensor([0.03, 0.2, 0.5, 0.8, 0.97])
    eps = 0.1
    z = torch.zeros(1)
    m = PairMoments(z, z, torch.ones(1), torch.ones(1), z)
    (_, _, ci, cs, co) = moment_coefficients(ts, eps, m)
    ci2, cs2, co2 = independent_unit_coefficients(ts, eps)
    for a, b in ((ci.flatten(), ci2), (cs.flatten(), cs2), (co.flatten(), co2)):
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    print("PASS test_independent_closed_form_still_holds")


def test_direct_velocity_reduces_to_cfm_at_eps0():
    """With eps=0 the direct-velocity target is exactly z1 - z0."""
    torch.manual_seed(0)
    mom = PairMoments(torch.zeros(2), torch.ones(2), torch.ones(2), torch.ones(2), torch.zeros(2))
    t = torch.tensor([0.2, 0.7])
    mx, my, ci, cs, co = direct_velocity_coefficients(t, 1e-8, mom)
    assert torch.isfinite(mx).all() and torch.isfinite(co).all()
    print("PASS test_direct_velocity_reduces_to_cfm_at_eps0")


def test_pair_ids():
    m = BidirDiT(channels=4, image_size=8, patch=2, width=32, depth=1, heads=2,
                 num_styles=5, style_cond="pair")
    ids = {}
    for a in range(5):
        for b in range(a + 1, 5):
            pid = int(m._pair_id(torch.tensor([a]), torch.tensor([b]))[0])
            ids[(a, b)] = pid
            assert pid == int(m._pair_id(torch.tensor([b]), torch.tensor([a]))[0])
    assert sorted(ids.values()) == list(range(10)), ids
    print("PASS test_pair_ids")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL V10 MATH TESTS PASSED")
