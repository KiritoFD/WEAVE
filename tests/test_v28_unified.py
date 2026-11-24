"""Unit and isolation tests for exp.v28_unified."""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from exp.v28_unified import atomic_save, recipe_hash
from exp.v28_unified.data_map import resolve_train_rgb_paths
from exp.v28_unified.student_models import UnifiedHybridUNet
from exp.v28_unified.train_student import _compute_targets, _sample_strengths_and_targets

ROOT = Path(__file__).resolve().parents[1]


def test_unet_params_and_lambda_zero_exact_identity() -> None:
    model = UnifiedHybridUNet(stats_dim=417).eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert 10_000_000 <= n_params <= 13_000_000, f"Unexpected param count: {n_params:,}"

    B = 4
    source = torch.randn(B, 4, 64, 64)
    c64 = torch.randn(B, 4, 64, 64)
    q32 = torch.randn(B, 32, 16, 16)
    tid = torch.tensor([0, 1, 2, 4])
    noise = torch.randn(B, 4, 64, 64)
    raw_stats = torch.randn(5, 417)
    priors = {
        "P_Y": torch.eye(4)[:1].repeat(4, 1) * 0.25,
        "P_C": torch.eye(4) - 0.25,
        "mu_c": torch.randn(5, 4, 1, 1),
        "std_c": torch.ones(5, 4, 1, 1),
    }

    with torch.no_grad():
        out_zero = model(source, c64, q32, tid, torch.zeros(B), raw_stats, priors, noise=noise)
    assert float((out_zero - source).abs().max()) == 0.0


def test_three_tier_no_midrange_linear_blend() -> None:
    gen = torch.Generator().manual_seed(42)
    modes = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    strengths, tc = _sample_strengths_and_targets(modes, len(modes), gen, "3tier")
    src = torch.zeros(len(modes), 4, 4, 4)
    hic = torch.ones(len(modes), 4, 4, 4) * 1.0
    mid = torch.ones(len(modes), 4, 4, 4) * 5.0
    his = torch.ones(len(modes), 4, 4, 4) * 9.0
    tgt = _compute_targets(src, hic, mid, his, strengths, tc, "3tier")
    assert torch.allclose(tgt[modes == 0], hic[modes == 0])
    assert torch.allclose(tgt[modes == 1], mid[modes == 1])
    assert torch.allclose(tgt[modes == 2], his[modes == 2])
    assert (tgt[modes == 3] > 0.0).all() and (tgt[modes == 3] <= 1.0).all()


def test_atomic_save_and_recipe_hash(tmp_path: Path) -> None:
    p = tmp_path / "sub/shard.pt"
    atomic_save({"x": torch.tensor([1, 2, 3])}, p)
    assert p.exists() and not p.with_suffix(".pt.tmp").exists()
    h1 = recipe_hash({"a": 1, "b": [2, 3]})
    h2 = recipe_hash({"b": [2, 3], "a": 1})
    assert h1 == h2 and len(h1) == 16


def test_referee_isolation_in_teacher_and_selector() -> None:
    """Strictly forbid CSD / DINO / CLIP / LPIPS imports in teacher candidate selection & T2/T4."""
    banned = ("csd_model", "csd_prototypes", "load_dino_model", "encode_dino", "FeatureStack", "lpips")
    for rel in ("exp/v28_unified/referee_free.py", "exp/v28_unified/teacher_routes.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module)
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
        for bad in banned:
            assert bad not in names and bad not in src, f"Forbidden referee symbol {bad!r} found in {rel}"


def test_resolve_train_rgb_paths_fails_on_missing() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="Incomplete train RGB mapping"):
        resolve_train_rgb_paths(["nonexistent_sample_xyz_12345"], [0], require_full=True)


def test_continuous_two_tier_and_hiswd_latent_damping() -> None:
    from exp.v28_unified.train_student import _latent_loss_terms

    gen = torch.Generator().manual_seed(7)
    modes = torch.tensor([0, 1, 2, 3, 2, 3])
    strengths, tc = _sample_strengths_and_targets(modes, len(modes), gen, "2tier")
    src = torch.zeros(len(modes), 4, 8, 8)
    hic = torch.ones(len(modes), 4, 8, 8) * 2.0
    his = torch.ones(len(modes), 4, 8, 8) * 6.0
    tgt = _compute_targets(src, hic, None, his, strengths, tc, "2tier")
    assert torch.allclose(tgt[modes == 0], hic[modes == 0])
    assert torch.allclose(tgt[modes == 1], his[modes == 1])
    assert (tgt[modes == 2] > 0.0).all() and (tgt[modes == 2] <= 2.0).all()
    assert (tgt[modes == 3] > 2.0).all() and (tgt[modes == 3] <= 6.0).all()

    priors = {"P_Y": torch.eye(4)[:1].repeat(4, 1) * 0.25}
    pred = torch.zeros(2, 4, 8, 8)
    target = torch.ones(2, 4, 8, 8)
    d_cal, _ = _latent_loss_terms(pred, target, torch.tensor([0.3, 0.9]), priors, "calibrated")
    d_hi, _ = _latent_loss_terms(pred, target, torch.tensor([0.3, 0.9]), priors, "hiswd")
    assert float(d_hi) < float(d_cal)


def test_unified_knob_schedule_two_stage_continuity() -> None:
    from exp.v28_unified.phase3_4_5 import unified_knob_schedule

    s0 = unified_knob_schedule(0.0)
    assert s0["base_lambda"] == 0.0 and s0["prox_steps"] == 0
    s1 = unified_knob_schedule(1.0)
    assert s1["base_lambda"] == 1.0 and s1["prox_steps"] == 0

    prev_eta = 0.0
    prev_pull = float("inf")
    prev_swd = 0.0
    for s in (1.25, 1.50, 1.75, 2.00):
        cfg = unified_knob_schedule(s)
        assert cfg["base_lambda"] == 1.0, "Must forbid >1.0 residual extrapolation"
        assert 1 <= cfg["prox_steps"] <= 3
        assert cfg["eta"] > prev_eta
        assert cfg["pullback_w"] < prev_pull
        assert cfg["swd_w"] > prev_swd
        prev_eta, prev_pull, prev_swd = cfg["eta"], cfg["pullback_w"], cfg["swd_w"]


def test_unet_latent_opt_exact_at_zero_and_tto_isolation() -> None:
    from exp.v28_unified.student_models import UnifiedHybridUNet

    m = UnifiedHybridUNet(stats_dim=27).eval()
    src = torch.randn(2, 4, 64, 64)
    c64 = torch.randn(2, 4, 64, 64)
    q32 = torch.randn(2, 32, 16, 16)
    tid = torch.tensor([0, 4])
    st = torch.ones(2)
    stats = torch.randn(5, 27)
    priors = {
        "P_Y": torch.eye(4)[:1].repeat(4, 1) * 0.25,
        "P_C": torch.eye(4) - 0.25,
        "mu_c": torch.randn(5, 4, 1, 1),
        "std_c": torch.ones(5, 4, 1, 1),
    }
    eps = torch.randn_like(src)
    out_std = m(src, c64, q32, tid, st, stats, priors, noise=eps)
    out_opt0 = m.forward_latent_opt(
        src, c64, q32, tid, st, stats, priors, noise=eps,
        dz_in=torch.zeros_like(src), dq32=torch.zeros_like(q32), dcond=torch.zeros(2, 192),
    )
    assert torch.allclose(out_std, out_opt0, atol=1e-6)

    tto_src = (ROOT / "exp/v28_unified/tto_ablation.py").read_text(encoding="utf-8")
    tto_tree = ast.parse(tto_src)
    imports = set()
    for node in ast.walk(tto_tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
    for bad in ("csd_model", "csd_prototypes", "load_dino_model", "encode_dino", "FeatureStack", "lpips", "dino_16"):
        assert bad not in imports and bad not in tto_src.lower(), f"Referee leakage in tto_ablation: {bad}"
