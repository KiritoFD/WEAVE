"""A03 unit tests: pure statistics helpers in tools/v25_cpu/a03_teacher_quality.py.

Covers only the pure functions (numpy in / float out); artifact loading and
the main report are exercised by running the module against the local packs.
"""
import numpy as np
import pytest

from tools.v25_cpu.a03_teacher_quality import (
    attribution_chain,
    composition_stats,
    family_of,
    family_pair_matrix,
    kept_cos_stats,
    pearson,
    rejected_mean_cos,
    retention_estimates,
    same_family_fraction,
    selection_share_of_excess,
    spearman,
)


def test_rejected_mean_cos_is_exact_partition_mean():
    # candidates = 900 kept (mean 0.96) + 300 rejected (mean 0.50)
    kept = np.full(900, 0.96)
    rej = np.full(300, 0.50)
    mean_all = float(np.concatenate([kept, rej]).mean())
    got = rejected_mean_cos(mean_all, 0.96, 1200, 900)
    assert abs(got - 0.50) < 1e-12
    with pytest.raises(ValueError):
        rejected_mean_cos(0.9, 0.9, 100, 100)  # n_kept == n_all


def test_selection_share_of_excess_decomposes_objective_vs_selection():
    # selection moved the mean from 0.94 to 0.96 above a band ceiling of 0.90:
    # the shift owns 0.02 of the 0.06 excess above the band
    assert selection_share_of_excess(0.94, 0.96, band_hi=0.90) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError):
        selection_share_of_excess(0.85, 0.89, band_hi=0.90)


def test_family_layout_and_pair_matrix():
    idx_c = np.array([0, 879, 880, 4399, 900])
    idx_r = np.array([10, 870, 2000, 300, 890])
    assert family_of(idx_c).tolist() == [0, 0, 1, 4, 1]
    mat = family_pair_matrix(idx_c, idx_r)
    assert mat.shape == (5, 5)
    assert mat.sum() == 5
    assert mat[0, 0] == 2  # (0,10) and (879,870)
    assert mat[1, 2] == 1  # (880, 2000)
    assert mat[4, 0] == 1  # (4399, 300)
    assert mat[1, 1] == 1  # (900, 890)
    assert same_family_fraction(idx_c, idx_r) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        family_of(np.array([4400]))  # outside the 5x880 pool


def test_retention_estimates_binomial_band():
    kept = np.array([431, 388, 352, 394, 435])
    r = retention_estimates(kept, 2400)
    assert r["expected_candidates_per_family"] == pytest.approx(480.0)
    assert r["rate"][0] == pytest.approx(431 / 480)
    assert all(lo <= rate <= hi for rate, lo, hi in zip(r["rate"], r["rate_ci_lo"], r["rate_ci_hi"]))
    with pytest.raises(ValueError):
        retention_estimates(np.array([1.0, 2.0]), 2400)  # wrong family count


def test_corr_helpers():
    x = np.arange(50.0)
    assert pearson(x, 2.0 * x + 1.0) == pytest.approx(1.0)
    assert spearman(x, np.log(x + 1.0)) == pytest.approx(1.0, abs=1e-12)  # monotone
    y = -x
    assert spearman(x, y) == pytest.approx(-1.0)
    # ties share average ranks: ranks(z)=[1.5,1.5,3] vs ranks=[1,2,3] -> r=0.866
    z = np.array([1.0, 1.0, 2.0])
    assert spearman(z, np.array([0.0, 1.0, 2.0])) == pytest.approx(0.8660254, abs=1e-6)
    # zero-variance input is defined as 0.0, not an error
    assert pearson(np.zeros(4), x[:4]) == 0.0
    with pytest.raises(ValueError):
        pearson(x[:4], x[:3])  # shape mismatch


def test_kept_cos_stats_quantiles_and_threshold():
    rng = np.random.default_rng(0)
    x = 0.95 + 0.03 * rng.random(1000)
    s = kept_cos_stats(x)
    assert s["min_threshold"] == pytest.approx(float(x.min()))
    assert s["q25"] <= s["q50"] <= s["q75"]
    assert s["frac_in_c1_band"] == 0.0  # everything above 0.90
    assert s["frac_gt_095"] == pytest.approx(float((x > 0.95).mean()))
    with pytest.raises(ValueError):
        kept_cos_stats(np.zeros((2, 2)))


def test_attribution_chain_shares_and_efficiency():
    o3 = {"dino_c": 0.58, "dino_s": 0.52, "csd_top1": 0.36}
    lf = {"dino_c": 0.68, "dino_s": 0.47, "csd_top1": 0.11}
    se = {"dino_c": 0.72, "dino_s": 0.473, "csd_top1": 0.12}
    c = attribution_chain(o3, lf, se)
    # teacher marginal over identity is negative on style axes; the identity
    # control already covers 24/25 of the Top-1 gap
    assert c["csd_top1"]["teacher_marginal_over_identity"] < 0
    assert c["csd_top1"]["share_at_or_below_identity"] == pytest.approx(0.24 / 0.25)
    assert c["dino_s"]["share_at_or_below_identity"] == pytest.approx(
        (0.52 - 0.473) / (0.52 - 0.47), abs=1e-9
    )
    # O3 buys positive style per unit content sacrifice; label_free does not
    assert c["style_per_content"]["o3_vs_self"] > 0
    assert c["style_per_content"]["label_free_vs_self"] < 0


def test_composition_stats_groups_partition_the_kept_set():
    rng = np.random.default_rng(1)
    idx_c = rng.integers(0, 4400, 500)
    idx_r = rng.integers(0, 4400, 500)
    mc = 0.96 + 0.01 * rng.random(500)
    s = composition_stats(mc, idx_c, idx_r)
    assert s["n_same"] + s["n_cross"] == 500
    assert 0.0 <= s["point_biserial_r_same_family"] <= 1.0 or s["point_biserial_r_same_family"] < 0.0
    assert len(s["mean_cos_by_content_family"]) == 5
