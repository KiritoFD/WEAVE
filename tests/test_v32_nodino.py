"""Unit tests for V32 Zero-DINO ConvNeXt-Stage3 Q32, K=8 bank, and AST zero-DINO hygiene."""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor, extract_cnx_q32, match_cnx_k8_submode
from exp.v32_nodino.diag_q32_6arm import perturb_q32


def test_v32_nodino_modules_and_hygiene():
    # Ensure training/extraction modules in exp/v32_nodino (except 6-arm diag on legacy Q32) never touch DINOv2
    v32_dir = Path(__file__).resolve().parents[1] / "exp/v32_nodino"
    for py_file in ("cnx_layout.py", "build_cnx_assets.py", "train_ecore_nodino.py", "train_pcore_nodino.py"):
        src = (v32_dir / py_file).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = (getattr(node, "module", "") or "").replace("nodino", "")
                assert "dino" not in mod.lower(), f"Forbidden DINO import {mod} in {py_file}"
                names = [a.name for a in getattr(node, "names", [])]
                for nm in names:
                    assert "dino" not in nm.replace("NoDINO", "").replace("nodino", "").lower(), f"Forbidden symbol {nm} in {py_file}"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                sval = node.value.replace("nodino", "").replace("NoDINO", "").lower()
                assert "v21_stiefel_ref_cache" not in sval and "k8_raw417_bank" not in sval and "dino_16" != sval, py_file

    # ConvNeXt Stage-3 shape + Stiefel orthogonality
    ext = ConvNeXtTinyStage3Extractor(device="cpu").eval()
    assert 10_000_000 <= ext.n_params() <= 14_000_000
    rgb = torch.rand(2, 3, 256, 256)
    g16 = ext.extract_gray_layout(rgb)
    mu = ext.extract_rgb_style_mean(rgb)
    assert g16.shape == (2, 384, 16, 16) and torch.isfinite(g16).all()
    assert mu.shape == (2, 384) and torch.isfinite(mu).all()

    w, _ = torch.linalg.qr(torch.randn(384, 32), mode="reduced")
    q32 = extract_cnx_q32(g16, w)
    assert q32.shape == (2, 32, 16, 16)
    gram = torch.bmm(q32.flatten(2), q32.flatten(2).transpose(1, 2)) / 256.0
    assert torch.allclose(gram, torch.eye(32).unsqueeze(0).expand(2, -1, -1), atol=1e-4)

    # K=8 ConvNeXt submode router
    bank = {
        "sub_scene": torch.randn(5, 8, 6144),
        "sub_raw417": torch.randn(5, 8, 417),
    }
    u = match_cnx_k8_submode(g16, torch.tensor([0, 4]), bank)
    assert u.shape == (2, 417) and torch.isfinite(u).all()

    # 6-arm Q32 perturbations
    assert torch.allclose(perturb_q32(q32, "correct"), q32)
    assert torch.allclose(perturb_q32(q32, "zero"), torch.zeros_like(q32))
    q_shuf = perturb_q32(q32, "shuffled")
    assert q_shuf.shape == q32.shape and not torch.allclose(q_shuf, q32)
