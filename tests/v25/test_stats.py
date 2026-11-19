"""A01 CPU tests: paired cluster bootstrap + non-inferiority verdicts.

Synthetic small samples with known answers only; no GPU, no real run data
required (one optional smoke test skips when the measured arms are absent).
Statistical foundation: ``src/eval/metrics.py`` (never reinvented).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pytest

from src.eval.metrics import bootstrap_source_ci, paired_cluster_ci
from tools.v25_cpu.a01_noninferiority import (
    BAND_METRICS,
    N_BOOT,
    REPO_ROOT,
    SEED,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
    VERDICT_PASS,
    compare_arms,
    noninferiority_verdict,
    pair_cross_rows,
    shared_bootstrap_draws,
)

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")


# --------------------------------------------------------------------------- #
# synthetic arm builders (known-answer fixtures)
# --------------------------------------------------------------------------- #
def make_rows(
    n_sources: int,
    *,
    dino_c: float = 0.70,
    dino_s: float = 0.50,
    lpips: float = 0.53,
    top1: float = 1.0,
    tasks_per_source: int = 4,
    seed: int = 7,
) -> list[dict]:
    """Build protocol-valid cross rows: `tasks_per_source` cross targets per source."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sources):
        stem = f"src_{i:03d}"
        for j in range(tasks_per_source):
            tgt = STYLES[(j + 1) % 5]
            ds = float(dino_s) + 0.001 * float(rng.random())
            style_scores = [0.30] * 5
            style_scores[STYLES.index(tgt)] = ds
            rows.append(
                {
                    "task_id": f"{stem}/{tgt}",
                    "src_stem": stem,
                    "src_style": STYLES[0],
                    "tgt_style": tgt,
                    "src_style_id": 0,
                    "tgt_style_id": STYLES.index(tgt),
                    "is_cross": True,
                    "style_scores_5": style_scores,
                    "clip_scores_5": [0.60] * 5,
                    "csd_scores_5": [0.20] * 5,
                    "dino_s": ds,
                    "dino_c": float(dino_c) + 0.001 * float(rng.random()),
                    "lpips": float(lpips) + 0.001 * float(rng.random()),
                    "target_top1": float(top1),
                }
            )
    return rows


def shifted(rows, delta_fn, metrics=("dino_c", "dino_s", "lpips", "target_top1")):
    """Copy rows with a per-source delta added to the given metrics (vectors kept consistent)."""
    out = []
    for r in rows:
        r2 = dict(r)
        d = delta_fn(r["src_stem"])
        if d and "dino_s" in metrics:
            sv = list(r2["style_scores_5"])
            k = STYLES.index(r2["tgt_style"])
            sv[k] = sv[k] + d
            r2["style_scores_5"] = sv
        for m in metrics:
            r2[m] = float(r2[m]) + d
        out.append(r2)
    return out


def ci_map_of(res: dict, metrics=BAND_METRICS) -> dict:
    return {m: (res["ci"][m]["mean"], res["ci"][m]["lo"], res["ci"][m]["hi"]) for m in metrics}


# --------------------------------------------------------------------------- #
# verdict three-state logic and band boundaries
# --------------------------------------------------------------------------- #
def test_verdict_pass_at_exact_band_edges():
    """Band edges are inclusive for PASS."""
    ci_map = {
        "dino_c": (-0.005, -0.010, -0.001),   # lo == floor
        "dino_s": (-0.001, -0.005, 0.002),    # lo == floor
        "lpips": (0.005, 0.002, 0.010),       # hi == ceiling (lower is better)
    }
    verdict, reasons = noninferiority_verdict(ci_map)
    assert verdict == VERDICT_PASS
    assert reasons


def test_verdict_fail_when_complex_significantly_better():
    dino_c = (-0.02, -0.03, -0.011)   # hi < floor
    dino_s = (-0.02, -0.03, -0.009)
    lpips = (-0.02, -0.03, -0.01)
    assert noninferiority_verdict({"dino_c": dino_c, "dino_s": dino_s, "lpips": lpips})[0] == VERDICT_FAIL
    # fail driven by dino_s alone
    assert noninferiority_verdict(
        {"dino_c": (0.0, -0.001, 0.001), "dino_s": (-0.01, -0.02, -0.0051),
         "lpips": (0.0, -0.001, 0.001)}
    )[0] == VERDICT_FAIL
    # fail driven by lpips alone (simple arm has higher = worse LPIPS)
    verdict, reasons = noninferiority_verdict(
        {"dino_c": (0.0, -0.001, 0.001), "dino_s": (0.0, -0.001, 0.001),
         "lpips": (0.02, 0.011, 0.03)}
    )
    assert verdict == VERDICT_FAIL
    assert any("lpips" in r for r in reasons)


def test_verdict_inconclusive_when_ci_overlaps_band_edge():
    # lo below floor but hi above floor: neither PASS (CI not fully inside) nor FAIL
    ci_map = {
        "dino_c": (-0.006, -0.012, -0.001),
        "dino_s": (-0.008, -0.009, -0.004),
        "lpips": (0.0, -0.001, 0.001),
    }
    verdict, _ = noninferiority_verdict(ci_map)
    assert verdict == VERDICT_INCONCLUSIVE


def test_lpips_direction_lower_is_better():
    # simple arm with LOWER LPIPS (better) passes even far below zero
    ci_map = {
        "dino_c": (0.0, -0.001, 0.001),
        "dino_s": (0.0, -0.001, 0.001),
        "lpips": (-0.02, -0.03, -0.01),   # hi <= +0.010 -> inside band
    }
    assert noninferiority_verdict(ci_map)[0] == VERDICT_PASS
    # simple arm with higher LPIPS (worse) can never PASS, fails when clearly beyond
    worse = {
        "dino_c": (0.0, -0.001, 0.001),
        "dino_s": (0.0, -0.001, 0.001),
        "lpips": (0.012, 0.008, 0.016),   # overlaps the ceiling -> INCONCLUSIVE
    }
    assert noninferiority_verdict(worse)[0] == VERDICT_INCONCLUSIVE
    much_worse = dict(worse, lpips=(0.02, 0.015, 0.03))
    assert noninferiority_verdict(much_worse)[0] == VERDICT_FAIL


def test_verdict_rejects_metric_set_mismatch():
    with pytest.raises(ValueError):
        noninferiority_verdict({"dino_c": (0, 0, 0), "dino_s": (0, 0, 0)})
    with pytest.raises(ValueError):
        noninferiority_verdict(dict.fromkeys((*BAND_METRICS, "dino_x"), (0.0, 0.0, 0.0)))


# --------------------------------------------------------------------------- #
# strict pairing validation
# --------------------------------------------------------------------------- #
def test_pairing_happy_path_cluster_integrity():
    rows_a = make_rows(2, seed=1)
    rows_b = shifted(rows_a, lambda s: 0.01)
    a, b, clusters = pair_cross_rows(
        rows_a, rows_b, expect_cross=8, expect_sources=2, tasks_per_source=4
    )
    assert len(a) == len(b) == 8
    assert len(set(clusters)) == 2
    assert all(x["task_id"] == y["task_id"] for x, y in zip(a, b))
    # deterministic task_id order
    assert [x["task_id"] for x in a] == sorted(x["task_id"] for x in a)


def test_pairing_duplicate_task_id_raises():
    rows_a = make_rows(2, seed=1)
    rows_b = shifted(rows_a, lambda s: 0.0)
    rows_a.append(dict(rows_a[0]))  # duplicate
    with pytest.raises(ValueError, match="duplicate"):
        pair_cross_rows(rows_a, rows_b, expect_cross=8, expect_sources=2)


def test_pairing_mismatched_task_sets_raise():
    rows_a = make_rows(2, seed=1)
    rows_b = shifted(rows_a, lambda s: 0.0)
    # same count, different id set -> true task_id-set mismatch
    rows_b_alt = [dict(r) for r in rows_b]
    rows_b_alt[0]["task_id"] = "ghost/task"
    with pytest.raises(ValueError, match="task_id sets differ"):
        pair_cross_rows(rows_a, rows_b_alt, expect_cross=8, expect_sources=2)
    # a genuinely missing task trips the count guard first
    with pytest.raises(ValueError, match="expected 8 cross tasks, got 7"):
        pair_cross_rows(rows_a, rows_b[:-1], expect_cross=8, expect_sources=2)


def test_pairing_nan_metric_raises():
    rows_a = make_rows(2, seed=1)
    rows_b = shifted(rows_a, lambda s: 0.0)
    rows_b[3] = dict(rows_b[3], lpips=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        pair_cross_rows(rows_a, rows_b, expect_cross=8, expect_sources=2)


def test_pairing_cluster_completeness_violations_raise():
    rows_a = make_rows(2, seed=1)
    rows_b = shifted(rows_a, lambda s: 0.0)
    # drop the same task from both arms: counts still consistent, but one
    # source no longer carries 4 cross tasks -> cluster completeness violated
    def drop_one(rows):
        return [r for r in rows if r["task_id"] != "src_001/" + STYLES[4]]

    with pytest.raises(ValueError, match="cluster completeness"):
        pair_cross_rows(drop_one(rows_a), drop_one(rows_b),
                        expect_cross=7, expect_sources=2, tasks_per_source=4)
    # wrong cluster count
    with pytest.raises(ValueError, match="source clusters"):
        pair_cross_rows(rows_a, rows_b, expect_cross=8, expect_sources=3)


# --------------------------------------------------------------------------- #
# known-answer end-to-end comparisons on synthetic arms
# --------------------------------------------------------------------------- #
def test_compare_arms_identical_arms_pass():
    rows = make_rows(3, seed=2)
    # direct pairing path: identical rows -> all diffs exactly 0 -> PASS
    a, b, clusters = pair_cross_rows(rows, rows, simple_name="s", complex_name="c",
                                     expect_cross=12, expect_sources=3)
    draws = shared_bootstrap_draws(len(set(clusters)))
    ci = {}
    for m in BAND_METRICS:
        diffs = [float(x[m]) - float(y[m]) for x, y in zip(a, b)]
        r = bootstrap_source_ci(diffs, clusters, n_boot=N_BOOT, seed=SEED, draws=draws)
        ci[m] = (r["mean"], r["ci95"][0], r["ci95"][1])
    verdict, _ = noninferiority_verdict(ci)
    assert verdict == VERDICT_PASS
    assert all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in ci.values())


def test_compare_arms_known_answer_three_states():
    rows_complex = make_rows(2, seed=3)
    # simple strictly better/equal everywhere -> PASS
    rows_better = shifted(rows_complex, lambda s: 0.0, metrics=("dino_c", "dino_s", "lpips", "target_top1"))
    a, b, clusters = pair_cross_rows(rows_better, rows_complex, simple_name="s",
                                     complex_name="c", expect_cross=8, expect_sources=2)
    draws = shared_bootstrap_draws(2)
    ci = {}
    for m in BAND_METRICS:
        diffs = [float(x[m]) - float(y[m]) for x, y in zip(a, b)]
        r = bootstrap_source_ci(diffs, clusters, n_boot=N_BOOT, seed=SEED, draws=draws)
        ci[m] = (r["mean"], r["ci95"][0], r["ci95"][1])
    assert noninferiority_verdict(ci)[0] == VERDICT_PASS

    # constant -0.02 on every metric -> CI degenerate at -0.02 -> FAIL
    rows_worse = shifted(rows_complex, lambda s: -0.02)
    a, b, clusters = pair_cross_rows(rows_worse, rows_complex, simple_name="s",
                                     complex_name="c", expect_cross=8, expect_sources=2)
    ci = {}
    for m in BAND_METRICS:
        diffs = [float(x[m]) - float(y[m]) for x, y in zip(a, b)]
        r = bootstrap_source_ci(diffs, clusters, n_boot=N_BOOT, seed=SEED, draws=draws)
        ci[m] = (r["mean"], r["ci95"][0], r["ci95"][1])
    verdict, reasons = noninferiority_verdict(ci)
    assert verdict == VERDICT_FAIL
    assert reasons

    # cluster-opposed deltas (+0.02 vs -0.02) -> wide CI overlapping the band -> INCONCLUSIVE
    rows_mixed = shifted(rows_complex, lambda s: 0.02 if s == "src_000" else -0.02)
    a, b, clusters = pair_cross_rows(rows_mixed, rows_complex, simple_name="s",
                                     complex_name="c", expect_cross=8, expect_sources=2)
    ci = {}
    for m in BAND_METRICS:
        diffs = [float(x[m]) - float(y[m]) for x, y in zip(a, b)]
        r = bootstrap_source_ci(diffs, clusters, n_boot=N_BOOT, seed=SEED, draws=draws)
        ci[m] = (r["mean"], r["ci95"][0], r["ci95"][1])
    assert ci["dino_s"][1] < -0.005 and ci["dino_s"][2] > -0.005  # truly overlapping
    assert noninferiority_verdict(ci)[0] == VERDICT_INCONCLUSIVE


def test_compare_arms_pipeline_and_foundation_equivalence():
    """compare_arms CI must equal both the foundation bootstrap and paired_cluster_ci."""
    rows_complex = make_rows(4, seed=4)
    rows_simple = shifted(rows_complex, lambda s: 0.004 if s.endswith("1") else 0.002)
    res = compare_arms(
        "syn_simple", "syn_complex",
        extra_metrics=("target_top1",),
        expect_cross=16, expect_sources=4,
        rows_simple=rows_simple, rows_complex=rows_complex,
    )
    assert res["n_tasks"] == 16 and res["n_clusters"] == 4
    assert res["verdict"] == VERDICT_PASS
    # same draws as the foundation's internal RNG (joint-index reuse)
    expected_draws = np.random.default_rng(SEED).integers(0, 4, size=(N_BOOT, 4))
    assert np.array_equal(shared_bootstrap_draws(4), expected_draws)
    # per-metric CI equals bootstrap_source_ci with draws=None, same seed
    a, b, clusters = pair_cross_rows(rows_simple, rows_complex, simple_name="syn_simple",
                                     complex_name="syn_complex", expect_cross=16, expect_sources=4)
    for m in BAND_METRICS:
        diffs = [float(x[m]) - float(y[m]) for x, y in zip(a, b)]
        r = bootstrap_source_ci(diffs, clusters, n_boot=N_BOOT, seed=SEED)
        got = res["ci"][m if m != "dino_s" else "dino_s"]
        assert got["lo"] == pytest.approx(r["ci95"][0], abs=1e-12)
        assert got["hi"] == pytest.approx(r["ci95"][1], abs=1e-12)
    # and equals paired_cluster_ci (clustering logic + RNG reuse, row order independent)
    pc = paired_cluster_ci(
        rows_simple, rows_complex, metrics=BAND_METRICS, n_boot=N_BOOT, seed=SEED
    )
    for m in BAND_METRICS:
        assert res["ci"][m]["lo"] == pytest.approx(pc[m]["ci95"][0], abs=1e-12)
        assert res["ci"][m]["hi"] == pytest.approx(pc[m]["ci95"][1], abs=1e-12)
        assert res["ci"][m]["mean"] == pytest.approx(pc[m]["mean"], abs=1e-12)


def test_compare_arms_guard_downgrades_pass_to_inconclusive():
    rows_complex = make_rows(3, seed=5)
    rows_simple = shifted(rows_complex, lambda s: 0.0, metrics=("dino_c", "dino_s", "lpips"))
    for r in rows_simple:
        if r["tgt_style"] == "Minimalism":
            r["target_top1"] = 0.5  # 50pp worst-class drop
    res = compare_arms("syn_simple", "syn_complex", extra_metrics=(),
                       expect_cross=12, expect_sources=3,
                       rows_simple=rows_simple, rows_complex=rows_complex)
    assert res["verdict"] == VERDICT_PASS           # CI-based verdict alone
    assert res["final"] == VERDICT_INCONCLUSIVE     # guard downgrades the decision
    assert res["guards"]["worst_class_ok"] is False
    assert res["guards"]["worst_class_style"] == "Minimalism"


def test_same_blind_guard_uses_non_cross_rows():
    rows_complex = make_rows(3, seed=6)
    rows_complex.append(dict(rows_complex[0], is_cross=False, tgt_style=STYLES[0],
                             task_id="same_x", target_top1=1.0))
    rows_simple = shifted(rows_complex, lambda s: 0.0, metrics=("dino_c", "dino_s", "lpips"))
    for r in rows_simple:
        if not r["is_cross"]:
            r["target_top1"] = 0.9  # 10pp same-blind drop
    res = compare_arms("syn_simple", "syn_complex", extra_metrics=(),
                       expect_cross=12, expect_sources=3,
                       rows_simple=rows_simple, rows_complex=rows_complex)
    assert res["guards"]["same_blind_ok"] is False
    assert res["final"] == VERDICT_INCONCLUSIVE


@pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "v25" / "measured" / "P02_d6h_csj" / "eval.json").is_file(),
    reason="Round 1 measured arms not present on this machine",
)
def test_real_round1_primary_comparison_smoke():
    res = compare_arms("P02_d6h_csj", "P02_d6h_full")
    assert res["n_tasks"] == 600 and res["n_clusters"] == 150
    assert res["verdict"] in {VERDICT_PASS, VERDICT_FAIL, VERDICT_INCONCLUSIVE}
    assert res["ci"]["dino_c"]["mean"] == pytest.approx(-0.0038, abs=5e-4)
