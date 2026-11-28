"""CPU regressions for the actual uint8 reference bug and strict archived CI gates."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.stage1_recipe.evaluation import audit

ROOT = Path(__file__).resolve().parents[1]


def feature_stack_without_models():
    spec = importlib.util.spec_from_file_location("isolated_features", ROOT / "src/utils/features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stack = module.FeatureStack.__new__(module.FeatureStack)
    stack.device = "cpu"
    stack._ref = SimpleNamespace(to_lpips_input=lambda rgb01: rgb01 * 2 - 1)
    stack.lpips = lambda left, right: (left - right).square().flatten(1).mean(1)
    return stack


def test_actual_f1_f6_calls_normalize_reference_and_preserve_generated_uint8():
    # Extract real call expressions without importing the GPU-oriented audit scripts.
    stack = feature_stack_without_models()
    raw = torch.randint(0, 256, (3, 3, 8, 8), dtype=torch.uint8)
    scope = {"stack": stack, "u8": raw, "raw_rgb": raw, "u8_all": raw,
             "src_u8": raw, "p_idx_a": torch.arange(3), "u8_v": raw,
             "src_v": raw, "mb": 2}
    count = 0
    for path in ("exp/paper_audit/f1_artist_holdout.py", "exp/paper_audit/f6_photo_to_art.py"):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "lpips_vs":
                result = eval(compile(ast.Expression(node), path, "eval"), scope)
                assert torch.equal(result, torch.zeros_like(result))
                assert eval(compile(ast.Expression(node.args[0]), path, "eval"), scope).dtype == torch.uint8
                count += 1
    assert count == 3
    # This is the original defect: uint8 reference passes into the RGB01 branch.
    assert stack.lpips_vs(raw, raw, batch=2).max() > 1


def test_fast5_s100_is_not_proven_noninferior_by_a_passing_mean():
    report = audit.audit_fast5(ROOT)
    p100 = report["profiles"]["fast"]["p_s100"]
    assert p100["values"]["same150"]["delta"] == pytest.approx(-4 / 3)
    assert p100["values"]["same150"]["ci95"][0] == -4
    assert p100["status"] == "inconclusive"
    assert p100["differs_from_archive"]
    assert not p100["non_inferiority_pass"]
    assert all(v["status"] == "pass" for k, v in report["profiles"]["fast"].items() if k != "p_s100")


def test_missing_ci_never_passes():
    assert audit.strict_ci_gate({})["status"] == "inconclusive"
    assert not audit.strict_ci_gate({})["non_inferiority_pass"]


def test_repair_identity_uses_same_normalization_both_sides():
    u8 = torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8)
    fake = lambda x, y: (x - y).square().flatten(1).mean(1)
    assert audit.score_lpips_u8(fake, u8, u8, batch=2) == [0.] * 4
    with pytest.raises(ValueError, match="uint8"):
        audit.score_lpips_u8(fake, u8.float(), u8)


def test_source_cluster_bootstrap_and_missing_assets_stop_before_model(tmp_path, monkeypatch):
    ci = audit.source_cluster_ci([1., 1., 3., 3.], ["a", "a", "b", "b"], n_boot=200)
    assert ci["n_sources"] == 2
    assert ci["mean"] == 2
    assert np.allclose(ci["ci95"], [1, 3])
    monkeypatch.setattr(audit, "_offline_lpips", lambda *_: pytest.fail("missing outputs must not load a model"))
    with pytest.raises(FileNotFoundError, match="generation is disabled"):
        audit.repair_lpips(tmp_path, tmp_path / "repair")
