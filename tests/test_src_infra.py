"""Unit tests for consolidated src/ infrastructure & 3060-safe helpers."""
from __future__ import annotations

import torch

from src.eval.metrics import enrich_rows, paired_cluster_ci, summarize_rows
from src.model.prism import PRISMFlashNet, extract_latent_multiscale_stats
from src.utils.protocol import STYLES, canonical_pid, pick_wrong_cross_tid
from src.utils.runtime import sanitize_cuda_env, tier_micro_accum
from src.utils.structure import (
    extract_color_fiber_27d,
    latent_to_structure_64,
    local_contrast_norm,
    rgb_to_oklab,
)


def test_tier_micro_accum_3060_and_4090():
    assert tier_micro_accum(128, total_gb_override=24.0) == (128, 1)
    assert tier_micro_accum(128, total_gb_override=12.0) == (32, 4)
    assert tier_micro_accum(128, total_gb_override=8.0) == (16, 8)


def test_sanitize_cuda_env_strips_expandable_segments():
    env = sanitize_cuda_env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
    assert "expandable_segments" not in env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    assert "PYTORCH_ALLOC_CONF" not in env
    assert env["PYTHONUTF8"] == "1"


def test_pick_wrong_cross_tid_never_collides_with_src_or_tgt():
    for s in range(5):
        for t in range(5):
            sid = torch.tensor([s])
            tid = torch.tensor([t])
            w = int(pick_wrong_cross_tid(sid, tid).item())
            if s == t:
                assert w == t
            else:
                assert w != t and w != s


def test_enrich_and_summarize_populates_confusion_matrix_and_paired_ci():
    rows_a, rows_b = [], []
    for src_i in range(4):
        s_id = src_i % 5
        for t_id, t_name in enumerate(STYLES):
            cs_a = [-0.2] * 5
            cs_a[t_id] = 0.5
            cs_b = [-0.2] * 5
            cs_b[s_id] = 0.5
            base = {
                "task_id": f"{STYLES[s_id]}/stem_{src_i}/{t_name}",
                "src_stem": f"stem_{src_i}",
                "src_style": STYLES[s_id],
                "tgt_style": t_name,
                "src_style_id": s_id,
                "tgt_style_id": t_id,
                "is_cross": s_id != t_id,
                "style_scores_5": [0.4] * 5,
                "clip_scores_5": [0.65] * 5,
                "dino_s": 0.4,
                "dino_c": 0.75,
                "lpips": 0.48,
                "dT": 0.08,
            }
            rows_a.append({**base, "csd_scores_5": cs_a, "target_top1": 1, "target_margin": 0.7})
            rows_b.append({**base, "csd_scores_5": cs_b, "target_top1": int(s_id == t_id), "target_margin": -0.7})

    summary = summarize_rows(rows_a, require_full=False, n_boot=64)
    cm = summary["csd_confusion_matrix"]
    assert cm is not None and sum(sum(r) for r in cm) == 16
    assert summary["cross600"]["means"]["pred_eq_src"] == 0.0

    pci = paired_cluster_ci(rows_a, rows_b, metrics=("target_top1",), n_boot=128)
    assert pci["target_top1"]["mean"] == 1.0
    assert canonical_pid("wikiart__Rococo__Antoine-Watteau") == "antoine-watteau"


def test_structure_and_color_fiber_differentiability():
    rgb = torch.rand(2, 3, 64, 64, requires_grad=True)
    lab = rgb_to_oklab(rgb)
    fib27 = extract_color_fiber_27d(rgb)
    assert lab.shape == (2, 3, 64, 64)
    assert fib27.shape == (2, 27)
    fib27.sum().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()

    z = torch.randn(2, 4, 64, 64) * 0.5
    c64 = latent_to_structure_64(z, torch.tensor([0.25] * 4), torch.tensor(0.5))
    assert c64.shape == (2, 4, 64, 64)
    assert float(c64.abs().max().item()) <= 1.0 + 1e-5
    lcn = local_contrast_norm(torch.rand(2, 1, 32, 32))
    assert float(lcn.abs().max().item()) <= 1.0 + 1e-5


def test_prism_flash_net_zero_chord_identity_cpu():
    import src.infer as infer_pkg
    import src.infer.variants  # noqa: F401
    assert "prism_flash" in infer_pkg.MODELS
    net = PRISMFlashNet(width=48, chord_dim=64).eval()
    with torch.no_grad():
        net.out_conv.weight.normal_(0, 0.02)
        z = torch.randn(2, 4, 64, 64) * 0.4
        st = extract_latent_multiscale_stats(z)
        assert st.shape == (2, 86)
        z_self = net(z, z_ref=z)
        assert float((z_self - z).abs().max().item()) < 1e-6
        z_cross = net(z, tid=torch.tensor([0, 4]))
        assert z_cross.shape == (2, 4, 64, 64)
        assert float((z_cross - z).abs().max().item()) > 0.0

