"""Unit tests for CLEAVE V35-Finalization pre-flight verification:
1. Zero-DINO AST hygiene across exp/v35_final/.
2. Continuous A3 startpoint constructor (build_pcore_startpoint):
   - Exact equivalence with V34 Stage A A0/A1/A2/A3 for (0,0), (1,0), (0,1), (1,1).
   - Continuous linear interpolation monotonicity for A4=(0.5, 1.0) and A5=(1.0, 0.5).
3. CLEAVE-XL (~68M, base_dim=256, cond_dim=512) architecture shapes, finiteness, and param budget.
4. Single-knob schedule parameterization (make_v35_schedule) & train_calib_300 score logic.
"""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v34_frontier.stage_a_startpoint import _py_unit_vec, reconstruct_analytic_draft
from exp.v35_final import RawWriterV35, build_pcore_startpoint, summarize_matched_c
from exp.v35_final.common import compute_pca32_raw, compute_whiten32, make_v35_schedule
from src.utils.runtime import auto_decode_batch, auto_infer_batch, auto_referee_batch, auto_vram_budget_gb, tier_micro_accum


def test_v35_zero_dino_ast_hygiene() -> None:
    v35_dir = Path(__file__).resolve().parents[1] / "exp/v35_final"
    for py_file in sorted(v35_dir.glob("*.py")):
        src = py_file.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = (getattr(node, "module", "") or "").replace("nodino", "")
                assert "dino" not in mod.lower(), f"Forbidden DINO import {mod} in {py_file.name}"
                for a in getattr(node, "names", []):
                    cleaned = a.name.replace("NoDINO", "").replace("nodino", "").replace("compute_dino_top5_mean", "")
                    assert "dino" not in cleaned.lower(), (py_file.name, a.name)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                sval = node.value.replace("nodino", "").replace("NoDINO", "").lower()
                assert "v21_stiefel_ref_cache" not in sval and "k8_raw417_bank" not in sval and sval != "dino_16", py_file.name


def test_build_pcore_startpoint_exact_equivalence_and_continuous_linearity() -> None:
    torch.manual_seed(17)
    u_y = torch.tensor([0.5574, 0.4953, 0.1774, -0.6423])
    u_y = u_y / u_y.norm()
    p_y = torch.outer(u_y, u_y)
    p_c = torch.eye(4) - p_y

    priors_a = {
        "P_Y": p_y,
        "P_C": p_c,
        "u_y": _py_unit_vec(p_y),
        "mu_low_1d": torch.linspace(-0.4, 0.4, 5).view(5, 1, 1, 1),
        "std_low_1d": torch.linspace(0.6, 1.2, 5).view(5, 1, 1, 1),
        "mu_c": torch.randn(5, 4, 1, 1) * 0.2,
        "std_c": torch.full((5, 4, 1, 1), 0.7),
    }

    z_e = torch.randn(3, 4, 64, 64)
    tid = torch.tensor([0, 2, 4])

    for mode, (ay, ac) in (("A0", (0.0, 0.0)), ("A1", (1.0, 0.0)), ("A2", (0.0, 1.0)), ("A3", (1.0, 1.0))):
        z_ref = reconstruct_analytic_draft(z_e, tid, priors_a, mode=mode)
        z_new = build_pcore_startpoint(z_e, tid, priors_a, alpha_y=ay, alpha_c=ac)
        assert torch.allclose(z_new, z_ref, atol=1e-5), (mode, (z_new - z_ref).abs().max().item())

    # Continuous linearity: A4(0.5, 1.0) == A2(0,1) + 0.5*(A1(1,0) - A0(0,0))
    z_a0 = build_pcore_startpoint(z_e, tid, priors_a, alpha_y=0.0, alpha_c=0.0)
    z_a1 = build_pcore_startpoint(z_e, tid, priors_a, alpha_y=1.0, alpha_c=0.0)
    z_a2 = build_pcore_startpoint(z_e, tid, priors_a, alpha_y=0.0, alpha_c=1.0)
    z_a4 = build_pcore_startpoint(z_e, tid, priors_a, alpha_y=0.5, alpha_c=1.0)
    z_a5 = build_pcore_startpoint(z_e, tid, priors_a, alpha_y=1.0, alpha_c=0.5)
    assert torch.allclose(z_a4, z_a2 + 0.5 * (z_a1 - z_a0), atol=1e-5)
    assert torch.allclose(z_a5, z_a1 + 0.5 * (z_a2 - z_a0), atol=1e-5)


def test_cleave_xl_architecture_and_v35_schedule() -> None:
    torch.manual_seed(5)
    m_xl = PCoreSLFUNet(base_dim=256, u_dim=417, cond_dim=512, use_gamma=True, normalize_u=True).eval()
    n_p = m_xl.n_params()
    assert 65_000_000 <= n_p <= 72_000_000, n_p
    with torch.no_grad():
        out = m_xl(
            torch.randn(2, 4, 64, 64),
            torch.randn(2, 1, 16, 16),
            torch.randn(2, 32, 16, 16),
            torch.tensor([1, 3]),
            torch.randn(2, 417),
            torch.tensor([0.4, 0.8]),
        )
    assert out.shape == (2, 4, 64, 64) and torch.isfinite(out).all()

    kw0 = make_v35_schedule(0.85, ws_mul=1.2, wc_mul=1.0, dtau0=-0.05)
    kw1 = make_v35_schedule(1.25, ws_mul=1.2, wc_mul=1.0, dtau0=-0.05)
    assert kw0["tau0"] > kw1["tau0"] and abs(kw0["tau0"] - 0.74) < 1e-5

    fake_levels = {
        "s075": {"dino_c": 0.71, "dino_s": 0.495, "csd_top1": 0.53},
        "s085": {"dino_c": 0.67, "dino_s": 0.515, "csd_top1": 0.55},
        "s100": {"dino_c": 0.56, "dino_s": 0.557, "csd_top1": 0.68},
        "s125": {"dino_c": 0.46, "dino_s": 0.578, "csd_top1": 0.81},
    }
    mc = summarize_matched_c(fake_levels)
    assert len(mc["dino_s_at_c"]) == 3 and mc["dino_s_at_c"][0] > mc["dino_s_at_c"][1] > mc["dino_s_at_c"][2]


def test_gpu_saturation_tier_helpers() -> None:
    # 24GB 4090 tier should run 44M at micro=128x1, eval batch 128, S10 decode 64, referee batch 64, budget >=21GB
    assert tier_micro_accum(128, default_micro_24gb=128, max_micro_12gb=32, max_micro_8gb=16, total_gb_override=23.6) == (128, 1)
    assert auto_vram_budget_gb(total_gb_override=23.6) >= 21.0
    assert auto_infer_batch(total_gb_override=23.6) == 128
    assert auto_decode_batch(total_gb_override=23.6) == 64
    assert auto_referee_batch(total_gb_override=23.6) == 64

    # 12GB 3060 / 8GB 4070LP tiers remain bounded cleanly
    assert auto_vram_budget_gb(total_gb_override=12.0) == 10.5
    assert auto_infer_batch(total_gb_override=12.0) == 32
    assert auto_decode_batch(total_gb_override=12.0) == 32
    assert auto_referee_batch(total_gb_override=12.0) == 16
    assert auto_vram_budget_gb(total_gb_override=8.0) == 7.2
    assert auto_infer_batch(total_gb_override=8.0) == 16
    assert auto_decode_batch(total_gb_override=8.0) == 16
    assert auto_referee_batch(total_gb_override=8.0) == 8


def test_content_ablation_reps_and_s0_identity() -> None:
    torch.manual_seed(11)
    p16 = torch.randn(4, 384, 16, 16)
    w, _, _ = torch.linalg.svd(torch.randn(384, 32), full_matrices=False)
    q_st = stiefel_polar_quotient(p16, w)
    q_wh = compute_whiten32(p16, w)
    q_pc = compute_pca32_raw(p16, w, rms_scale=1.0)
    assert q_st.shape == q_wh.shape == q_pc.shape == (4, 32, 16, 16)

    # Stiefel has exact identity Gram matrix; plain channel whitening has diag==1 but nonzero off-diagonal covariance
    g_st = torch.bmm(q_st.flatten(2), q_st.flatten(2).transpose(1, 2)) / 256.0
    g_wh = torch.bmm(q_wh.flatten(2), q_wh.flatten(2).transpose(1, 2)) / 256.0
    assert (g_st - torch.eye(32)).abs().max().item() < 1e-4
    assert (g_wh.diagonal(dim1=1, dim2=2) - 1.0).abs().max().item() < 1e-4
    off_wh = (g_wh - torch.diag_embed(g_wh.diagonal(dim1=1, dim2=2))).abs().mean().item()
    assert off_wh > 0.01

    kw_zero = make_v35_schedule(0.0)
    assert kw_zero["tau0"] == 1.0 and kw_zero["draft_lambda"] == 0.0 and kw_zero["wsp"] == 0.0

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        wr = RawWriterV35("unit_calib_tmp", dec=None, device="cpu", raw_root=Path(td), expected_tasks=300)
        assert wr.expected_tasks == 300


def test_p9_artist_key_schedule_and_bracketing() -> None:
    from exp.v35_final.p0_p2_evals import _artist_key

    assert _artist_key("d5__Early_Renaissance__Early_Renaissance__benozzo-gozzoli_foo") == "benozzo-gozzoli"
    assert _artist_key("Early_Renaissance__andrea-mantegna_bar") == "andrea-mantegna"

    # Sub-0.75 smooth schedule preserves >=0.75 and =0.65 exactly while bridging 0.68..0.72 smoothly
    k65 = make_v35_schedule(0.65)
    k70 = make_v35_schedule(0.70)
    k75 = make_v35_schedule(0.75)
    k85 = make_v35_schedule(0.85)
    assert k65["tau0"] == 0.87 and k75["tau0"] == 0.835 and k85["tau0"] == 0.79
    assert k65["tau0"] > k70["tau0"] > k75["tau0"]
    assert k65["draft_lambda"] < k70["draft_lambda"] < k75["draft_lambda"]

    # Explicit bracketing metadata detects boundary clamping when C is outside [c_lo, c_hi]
    unbr_lv = {"s075": {"dino_c": 0.68, "dino_s": 0.51, "csd_top1": 0.55}, "s100": {"dino_c": 0.55, "dino_s": 0.56, "csd_top1": 0.69}}
    mc_un = summarize_matched_c(unbr_lv, c_grid=(0.61, 0.70))
    assert mc_un["bracketed"] == [True, False] and not mc_un["all_bracketed"]
    br_lv = {"s065": {"dino_c": 0.74, "dino_s": 0.49, "csd_top1": 0.50}, **unbr_lv}
    mc_br = summarize_matched_c(br_lv, c_grid=(0.61, 0.70))
    assert mc_br["bracketed"] == [True, True] and mc_br["all_bracketed"]
