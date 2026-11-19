"""A02 CPU tests: lineage records, solver hashing, consistency verdicts.

No GPU, no real caches required: heavy/large artifacts stay UNKNOWN by design
(legacy-probe contract: no sha -> "unknown", never raises).  Two guarded
drift tests cross-check the in-file literals against the live experiment
modules when those import cleanly on CPU; they skip otherwise.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json

import pytest

from tools.v25_cpu.a02_teacher_lineage import (
    ANCHOR_EQUIVALENCE_CLAIMS,
    B4_CKPT,
    D5_SOLVER_LIT,
    D6H_SOLVER_LIT,
    D6_SOLVER_LIT,
    EVAL_SEED_PROTOCOL,
    LineageRecord,
    REPO_ROOT,
    R2_EVAL_SEED,
    T4_CONFIG_LIT,
    T4_HASH_LOCAL,
    UNKNOWN,
    VERDICT_IDENTICAL,
    VERDICT_INCONSISTENT,
    VERDICT_PENDING_4090,
    VERDICT_DISTINCT,
    b4_t4_stats,
    build_registry,
    compare,
    find_inconsistencies,
    probe_pt,
    registry_by_name,
    run_pair_plan,
    sha256_file,
    solver_hash,
)

# --------------------------------------------------------------------------- #
# guarded live-module imports (drift checks skip when unavailable)
# --------------------------------------------------------------------------- #
try:
    import exp.v22.core as _v22_core
    import exp.v22 as _v22
    import exp.v22.t4 as _v22_t4

    V22_IMPORTABLE = True
except Exception:  # pragma: no cover - environment dependent
    V22_IMPORTABLE = False


# --------------------------------------------------------------------------- #
# solver_hash: deterministic, key-order independent, repo-convention aligned
# --------------------------------------------------------------------------- #
def test_solver_hash_deterministic_and_order_independent() -> None:
    h1 = solver_hash(D6_SOLVER_LIT)
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)
    assert solver_hash(D6_SOLVER_LIT) == h1
    shuffled = {k: D6_SOLVER_LIT[k] for k in reversed(list(D6_SOLVER_LIT))}
    assert solver_hash(shuffled) == h1
    assert solver_hash(D6_SOLVER_LIT, n=12) == h1[:12]


def test_solver_hash_distinguishes_solvers() -> None:
    hashes = {solver_hash(s) for s in (D5_SOLVER_LIT, D6_SOLVER_LIT, D6H_SOLVER_LIT)}
    assert len(hashes) == 3
    # D6H is D6 with only tau_min/tau_hi replaced
    assert solver_hash(D6_SOLVER_LIT) != solver_hash(D6H_SOLVER_LIT)


@pytest.mark.skipif(not V22_IMPORTABLE, reason="exp.v22 not importable on CPU")
def test_solver_hash_matches_repo_config_hash_convention() -> None:
    assert solver_hash(D5_SOLVER_LIT, n=12) == _v22.config_hash(D5_SOLVER_LIT)
    assert solver_hash(D6_SOLVER_LIT, n=12) == _v22.config_hash(D6_SOLVER_LIT)


@pytest.mark.skipif(not V22_IMPORTABLE, reason="exp.v22 not importable on CPU")
def test_solver_literals_match_live_modules() -> None:
    assert D5_SOLVER_LIT == dict(_v22_core.D5_SOLVER)
    assert D6_SOLVER_LIT == dict(_v22_core.D6_SOLVER)
    assert D6H_SOLVER_LIT == dict(D6_SOLVER_LIT, tau_min=0.48, tau_hi=0.78)


@pytest.mark.skipif(not V22_IMPORTABLE, reason="exp.v22 not importable on CPU")
def test_t4_config_literal_matches_live_t4_config() -> None:
    assert T4_CONFIG_LIT == dict(_v22_t4.T4_CONFIG)
    assert T4_HASH_LOCAL == _v22_t4.T4_HASH


# --------------------------------------------------------------------------- #
# LineageRecord: serialization roundtrip + unknown-sha tolerance
# --------------------------------------------------------------------------- #
def _rec(**over) -> LineageRecord:
    base = dict(
        name="R", side="eval", stage="V25",
        model_path="runs/x/model.pt", model_sha=UNKNOWN,
        anchor_kind="t4_opt", anchor_def="T4", anchor_sha=T4_HASH_LOCAL,
        solver=dict(D6H_SOLVER_LIT), branches="cs_j", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/x.py:1",
    )
    base.update(over)
    return LineageRecord(**base)


def test_lineage_record_roundtrip() -> None:
    rec = _rec()
    d = rec.to_dict()
    assert json.loads(json.dumps(d)) == d  # JSON-safe payload
    rec2 = LineageRecord.from_dict(d)
    assert rec2.identity_key() == rec.identity_key()
    assert rec2.solver_h == rec.solver_h == solver_hash(D6H_SOLVER_LIT)
    # extra keys in the payload are tolerated (forward compatibility)
    rec3 = LineageRecord.from_dict({**d, "future_field": 1})
    assert rec3.name == rec.name


def test_lineage_record_identity_key_shape() -> None:
    rec = _rec()
    key = rec.identity_key()
    assert key == (rec.model_sha, rec.anchor_sha, rec.solver_h, rec.branches, rec.gate)
    assert key[2] == solver_hash(D6H_SOLVER_LIT)


def test_lineage_record_without_shas_marks_unknown_without_raising() -> None:
    rec = LineageRecord(name="LEG", side="legacy", stage="old", model_path="gone.pt")
    assert rec.model_sha == UNKNOWN
    assert rec.anchor_sha == UNKNOWN
    assert rec.solver_h == UNKNOWN
    assert rec.identity_key() == (UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN)
    d = rec.to_dict()
    assert LineageRecord.from_dict(d).identity_key() == rec.identity_key()


def test_sha256_file_missing_returns_unknown() -> None:
    assert sha256_file(REPO_ROOT / "runs/__definitely_missing__.pt") == UNKNOWN


# --------------------------------------------------------------------------- #
# compare(): verdict matrix
# --------------------------------------------------------------------------- #
B4_SHA = "b" * 64


def test_compare_identical() -> None:
    a = _rec(name="A")
    b = _rec(name="B")
    res = compare(a, b)
    assert res["verdict"] == VERDICT_IDENTICAL
    assert res["attribution_allowed"].startswith("yes")


def test_compare_pending_on_declared_equivalent_anchors() -> None:
    a = _rec(name="A", anchor_kind="t4_opt", anchor_sha=T4_HASH_LOCAL)
    b = _rec(name="B", anchor_kind="b4_canon", anchor_sha=B4_SHA)
    res = compare(a, b)
    assert res["verdict"] == VERDICT_PENDING_4090
    assert frozenset(("t4_opt", "b4_canon")) in ANCHOR_EQUIVALENCE_CLAIMS


def test_compare_inconsistent_on_semantic_anchor_mismatch() -> None:
    a = _rec(name="A", anchor_kind="t4_opt", anchor_sha=T4_HASH_LOCAL)
    b = _rec(name="B", anchor_kind="z_src_clean", anchor_sha=UNKNOWN)
    res = compare(a, b)
    assert res["verdict"] == VERDICT_INCONSISTENT
    assert not res["attribution_allowed"].startswith("yes")


def test_compare_inconsistent_on_same_family_version_drift() -> None:
    a = _rec(name="A", anchor_sha="1" * 64)
    b = _rec(name="B", anchor_sha="2" * 64)
    assert compare(a, b)["verdict"] == VERDICT_INCONSISTENT


def test_compare_distinct_on_solver_difference() -> None:
    a = _rec(name="A", solver=dict(D6_SOLVER_LIT))
    b = _rec(name="B", solver=dict(D6H_SOLVER_LIT))
    res = compare(a, b)
    assert res["verdict"] == VERDICT_DISTINCT
    assert any("solver_hash differs" in r for r in res["reasons"])


def test_compare_unknown_sha_does_not_fake_identity_across_paths() -> None:
    a = _rec(name="A", model_path="runs/x/one.pt", model_sha=UNKNOWN)
    b = _rec(name="B", model_path="runs/y/two.pt", model_sha=UNKNOWN)
    res = compare(a, b)
    assert res["verdict"] == VERDICT_PENDING_4090


def test_compare_never_raises_with_all_unknown() -> None:
    a = LineageRecord(name="A", side="legacy", stage="s", model_path="p")
    b = LineageRecord(name="B", side="legacy", stage="s", model_path="p")
    res = compare(a, b)
    assert res["verdict"] in (VERDICT_IDENTICAL, VERDICT_PENDING_4090)


# --------------------------------------------------------------------------- #
# registry + pair plan: real lineage consistency audit
# --------------------------------------------------------------------------- #
def test_registry_names_unique_and_provenance_cited() -> None:
    reg = build_registry()
    names = [r.name for r in reg]
    assert len(names) == len(set(names))
    for r in reg:
        assert r.code_ref, f"{r.name} lacks code provenance"
        assert r.side in ("eval", "train", "legacy", "anchor")
        assert r.anchor_kind != UNKNOWN, f"{r.name} lacks anchor_kind"


def test_pair_plan_names_resolve() -> None:
    reg = registry_by_name()
    for a, b, table in (
        (aa, bb, tt)
        for aa, bb, tt in [
            (c["a"], c["b"], c["table"]) for c in run_pair_plan()
        ]
    ):
        assert a in reg and b in reg
        assert table in ("a", "b", "c", "d")


def test_inconsistency_detection_triggers() -> None:
    comps = run_pair_plan()
    bad = find_inconsistencies(comps)
    assert bad, "the known inconsistent pairs must be detected"
    pairs = {(c["a"], c["b"]) for c in bad}
    assert ("TRAIN_V25_R2_SELFPAIRED", "EVAL_V25_R2_TRAINED") in pairs, \
        "Round-2 z_src-clean training anchor vs T4 eval anchor must be INCONSISTENT"
    assert ("EVAL_V25_P13_ANCSRC", "EVAL_V25_P02_D6H_CSJ") in pairs
    assert ("LEGACY_V24_M3_M4_POOL", "TRAIN_V25_TEACHER_CACHE") in pairs


def test_b4_t4_pair_is_pending_not_identical() -> None:
    comps = run_pair_plan()
    by_pair = {(c["a"], c["b"]): c for c in comps}
    res = by_pair[("ANCHOR_T4_TEST750", "ANCHOR_B4_TRAIN4400")]
    assert res["verdict"] == VERDICT_PENDING_4090
    res2 = by_pair[("TRAIN_V25_TEACHER_CACHE", "EVAL_V25_P02_D6H_CSJ")]
    assert res2["verdict"] == VERDICT_PENDING_4090


def test_v20_to_v24_d6_lineage_is_identical() -> None:
    comps = run_pair_plan()
    by_pair = {(c["a"], c["b"]): c for c in comps}
    res = by_pair[("EVAL_V20_D6_ORIG", "EVAL_V24_M0_V20_D6")]
    assert res["verdict"] == VERDICT_IDENTICAL


def test_teacher_cache_matches_eval_solver_and_branches() -> None:
    reg = registry_by_name()
    tr = reg["TRAIN_V25_TEACHER_CACHE"]
    ev = reg["EVAL_V25_P02_D6H_CSJ"]
    assert tr.solver_h == ev.solver_h
    assert tr.branches == ev.branches == "cs_j"
    assert tr.gate == ev.gate == "original"
    assert tr.anchor_kind == "b4_canon" and ev.anchor_kind == "t4_opt"


def test_round2_eval_seed_mismatch_is_registered() -> None:
    reg = registry_by_name()
    assert reg["EVAL_V25_R2_TRAINED"].noise_seed == R2_EVAL_SEED
    assert reg["EVAL_V25_P02_D6H_CSJ"].noise_seed == EVAL_SEED_PROTOCOL
    assert R2_EVAL_SEED != EVAL_SEED_PROTOCOL


# --------------------------------------------------------------------------- #
# b4_t4_stats: pure math on synthetic tensors (4090 check logic)
# --------------------------------------------------------------------------- #
def _synth(n: int = 40, ch: int = 4, hw: int = 8, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    z_src = rng.normal(size=(n, ch, hw, hw))
    z_t4 = z_src + 0.10 * rng.normal(size=z_src.shape)      # T4-like small edit
    z_b4_same = z_t4 + 0.01 * rng.normal(size=z_src.shape)  # B4 ~ T4
    z_b4_diff = z_src + 1.00 * rng.normal(size=z_src.shape)  # B4 far from T4
    return z_src, z_t4, z_b4_same, z_b4_diff


def test_b4_t4_stats_equivalent_scale() -> None:
    z_src, z_t4, z_b4, _ = _synth(seed=1)
    stats = b4_t4_stats(z_b4, z_t4, z_ref=z_src)
    assert stats["n"] == 40
    assert stats["reading"].startswith("equivalent-scale")
    assert stats["channel_quantile_absdiff_max"] < 0.5


def test_b4_t4_stats_distinct_scale() -> None:
    z_src, z_t4, _, z_b4 = _synth(seed=2)
    stats = b4_t4_stats(z_b4, z_t4, z_ref=z_src)
    assert stats["reading"].startswith("distinct-scale")


def test_b4_t4_stats_inconclusive_band() -> None:
    z_src, z_t4, _, _ = _synth(seed=3)
    import numpy as np

    rng = np.random.default_rng(33)
    z_b4 = z_t4 + 0.05 * rng.normal(size=z_t4.shape)  # halfway between T4 and src scale
    stats = b4_t4_stats(z_b4, z_t4, z_ref=z_src)
    assert stats["reading"].startswith("inconclusive")


def test_b4_t4_stats_shape_mismatch_raises() -> None:
    import numpy as np

    with pytest.raises(ValueError):
        b4_t4_stats(np.zeros((2, 4, 8, 8)), np.zeros((3, 4, 8, 8)))


# --------------------------------------------------------------------------- #
# probe_pt: legacy/tolerant .pt probing
# --------------------------------------------------------------------------- #
def test_probe_pt_missing_is_unknown_not_raise(tmp_path) -> None:
    info = probe_pt(tmp_path / "nope.pt")
    assert info["status"] == "missing"


def test_probe_pt_non_torch_file_is_unknown_not_raise(tmp_path) -> None:
    p = tmp_path / "plain.txt"
    p.write_text("not a torch file", encoding="utf-8")
    info = probe_pt(p)
    assert info["status"] == "unknown"
    assert "error" in info


def test_probe_pt_reads_meta_without_tensors(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    p = tmp_path / "cache.pt"
    torch.save(
        {
            "teacher_spec": {"anchor": "B4_distill50 + canonicalizer(c1_gentle)"},
            "updated_utc": "2026-11-19T00:00:00",
            "z": torch.zeros(4, 4, 8, 8),
        },
        p,
    )
    info = probe_pt(p)
    assert info["status"] == "ok"
    assert info["meta"]["teacher_spec"]["anchor"].startswith("B4_distill50")
    assert info["tensor_shapes"]["z"] == [4, 4, 8, 8]
    assert len(info["hashes"]["file_sha256"]) == 64


def test_probe_pt_local_b4_checkpoint_if_present() -> None:
    info = probe_pt(REPO_ROOT / B4_CKPT)
    if info["status"] != "ok":
        pytest.skip("B4 checkpoint not present on this machine")
    assert info["size_mb"] < 100  # small ckpt: fully readable
