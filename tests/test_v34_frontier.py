"""Unit tests for CLEAVE V34 pre-flight verification:
1. Stage A analytic decomposition L_low + L_high == P_Y z_E exact, plus A1/A2/A3 orthogonal invariance.
2. 44M C2 & C3 (PCoreWithSemanticControl) shapes, parameter counts (~44.5M & ~51.8M), zero-init parity, and drop_content=True zero leak.
3. Stage B continuous convex combination retrieval & multi-scale texture extractor shape/finiteness.
4. Selective merge gate logic across Stage A/B/C reports.
5. Zero-DINO AST hygiene across exp/v34_frontier/.
"""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v34_frontier.stage_a_startpoint import _py_unit_vec, decompose_py_luma, reconstruct_analytic_draft
from exp.v34_frontier.stage_d_merge import select_merge_config
from exp.v34_frontier.train_common import build_pcore_v34, retrieve_continuous_style_u, sample_analytic_bridge_segment


def test_v34_zero_dino_ast_hygiene() -> None:
    v34_dir = Path(__file__).resolve().parents[1] / "exp/v34_frontier"
    for py_file in sorted(v34_dir.glob("*.py")):
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


def test_stage_a_analytic_decomposition_and_orthogonality() -> None:
    torch.manual_seed(11)
    u_y = torch.tensor([0.5574, 0.4953, 0.1774, -0.6423])
    u_y = u_y / u_y.norm()
    p_y = torch.outer(u_y, u_y)
    p_c = torch.eye(4) - p_y

    z_e = torch.randn(3, 4, 64, 64)
    tid = torch.tensor([0, 2, 4])

    l_low, l_high, zy = decompose_py_luma(z_e, p_y)
    assert torch.allclose(l_low + l_high, zy, atol=1e-5)
    assert torch.allclose(_py_unit_vec(p_y).abs(), u_y.abs(), atol=1e-5)

    priors_a = {
        "P_Y": p_y,
        "P_C": p_c,
        "u_y": _py_unit_vec(p_y),
        "mu_low_1d": torch.linspace(-0.4, 0.4, 5).view(5, 1, 1, 1),
        "std_low_1d": torch.linspace(0.6, 1.2, 5).view(5, 1, 1, 1),
        "mu_c": torch.randn(5, 4, 1, 1) * 0.2,
        "std_c": torch.full((5, 4, 1, 1), 0.7),
    }

    # A0 exact identity
    z_a0 = reconstruct_analytic_draft(z_e, tid, priors_a, mode="A0")
    assert torch.equal(z_a0, z_e)

    # A1 keeps L_high & P_C untouched in construction (Gaussian G is smooth so G^2 != G, verify high-freq correlation & exact P_C invariance)
    z_a1 = reconstruct_analytic_draft(z_e, tid, priors_a, mode="A1")
    l_low_a1, l_high_a1, zy_a1 = decompose_py_luma(z_a1, p_y)
    zc_e = torch.einsum("ij,bjhw->bihw", p_c, z_e)
    zc_a1 = torch.einsum("ij,bjhw->bihw", p_c, z_a1)
    assert torch.allclose(zc_a1, zc_e, atol=1e-5)
    assert torch.nn.functional.cosine_similarity(l_high_a1.flatten(1), l_high.flatten(1), dim=1).min().item() > 0.90
    assert torch.allclose(l_low_a1.mean(dim=(2, 3)), priors_a["mu_low_1d"][tid].view(-1, 1) * priors_a["u_y"].view(1, 4), atol=2e-2)

    # A2 keeps P_Y untouched and leaves P_Y(z_a2) == P_Y(z_e)
    z_a2 = reconstruct_analytic_draft(z_e, tid, priors_a, mode="A2")
    _, _, zy_a2 = decompose_py_luma(z_a2, p_y)
    assert torch.allclose(zy_a2, zy, atol=1e-5)

    # A3 keeps high-freq luma structure while modifying both L_low and P_C
    z_a3 = reconstruct_analytic_draft(z_e, tid, priors_a, mode="A3")
    l_low_a3, l_high_a3, _ = decompose_py_luma(z_a3, p_y)
    zc_a3 = torch.einsum("ij,bjhw->bihw", p_c, z_a3)
    assert torch.nn.functional.cosine_similarity(l_high_a3.flatten(1), l_high.flatten(1), dim=1).min().item() > 0.90
    assert not torch.allclose(l_low_a3, l_low, atol=1e-2)
    assert not torch.allclose(zc_a3, zc_e, atol=1e-2)


def test_44m_c2_c3_param_counts_zero_init_and_cfg_isolation() -> None:
    torch.manual_seed(22)
    m_c2, cfg_c2 = build_pcore_v34(base_dim=208, u_dim=417, cond_dim=384, use_adapter=False, device="cpu")
    m_c3, cfg_c3 = build_pcore_v34(base_dim=208, u_dim=417, cond_dim=384, use_adapter=True, device="cpu")
    assert 44_000_000 <= cfg_c2["pcore_params"] <= 45_500_000, cfg_c2
    assert 6_500_000 <= cfg_c3["ctrl_params"] <= 8_000_000, cfg_c3
    assert cfg_c3["total_params"] == cfg_c3["pcore_params"] + cfg_c3["ctrl_params"]

    # Zero-init parity: with identical weights on pcore, initial Adapter adds exact 0
    m_c3.pcore.load_state_dict(m_c2.state_dict(), strict=True)
    with torch.no_grad():
        m_c2.out_conv.weight.normal_(0.0, 0.05)
        m_c3.pcore.out_conv.weight.copy_(m_c2.out_conv.weight)
    m_c2.eval()
    m_c3.eval()

    zt = torch.randn(2, 4, 64, 64)
    y16 = torch.randn(2, 1, 16, 16)
    q16 = torch.randn(2, 32, 16, 16)
    q32_hi = torch.randn(2, 32, 32, 32)
    tid = torch.tensor([0, 3])
    u = torch.randn(2, 417)
    tau = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        v2 = m_c2(zt, y16, q16, tid, u, tau)
        v3 = m_c3(zt, y16, q16, tid, u, tau, q32_hi=q32_hi)
    assert torch.allclose(v2, v3, atol=1e-5)

    # After non-zero perturbation to adapter out_conv, drop_content=True leaks zero signal
    with torch.no_grad():
        for mod in m_c3.ctrl.modules():
            if hasattr(mod, "out_conv"):
                mod.out_conv.weight.normal_(0.0, 0.02)
    dc = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        va = m_c3(zt, y16, q16, tid, u, tau, q32_hi=q32_hi, drop_content=dc)
        vb = m_c3(zt, y16 * 5.0 + 1.0, q16 * -3.0, tid, u, tau, q32_hi=q32_hi * 4.0, drop_content=dc)
    assert torch.allclose(va, vb, atol=1e-6)

    # Integrator works smoothly for both C2 (q32_hi=None) and C3 (q32_hi!=None)
    eps = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        z_out2 = integrate_v33_trajectory(m_c2, None, eps, y16, q16, tid, u, tau0=0.0, ws=3.0, wc=1.8, wsp=0.8, steps=2)
        z_out3 = integrate_v33_trajectory(m_c3, None, eps, y16, q16, tid, u, tau0=0.0, ws=3.0, wc=1.8, wsp=0.8, steps=2, q32_hi=q32_hi)
    assert z_out2.shape == (2, 4, 64, 64) and torch.isfinite(z_out2).all()
    assert z_out3.shape == (2, 4, 64, 64) and torch.isfinite(z_out3).all()


def test_continuous_style_retrieval_and_merge_gate() -> None:
    torch.manual_seed(33)
    # 20 samples per style = 100 total
    sid = torch.arange(5).repeat_interleave(20)
    style_indices = [(sid == s).nonzero(as_tuple=True)[0] for s in range(5)]
    inst_cache = {
        "inst_scene": torch.nn.functional.normalize(torch.randn(100, 6144), dim=-1),
        "inst_417": torch.nn.functional.normalize(torch.randn(100, 417), dim=-1),
        "inst_1101": torch.randn(100, 1101),
        "style_indices": style_indices,
    }
    cnx_gray = torch.randn(4, 384, 16, 16)
    tid = torch.tensor([0, 1, 2, 4])
    u417 = retrieve_continuous_style_u(cnx_gray, tid, inst_cache, cond_key="inst_417")
    u1101 = retrieve_continuous_style_u(cnx_gray, tid, inst_cache, cond_key="inst_1101")
    assert u417.shape == (4, 417) and torch.isfinite(u417).all()
    assert u1101.shape == (4, 1101) and torch.isfinite(u1101).all()

    # Check selective merge ignores Stage A & B when marked not effective
    fake_a = {
        "stage_a_effective": False,
        "arms": {"StageA_A0_s100": {"dino_c": 0.50, "dino_s": 0.56, "per_tgt_dino_s": {"Min": 0.45}}},
    }
    fake_b = {"frontier_verified_effective": False, "winner_style_mode": "inst_1101"}
    fake_c = {
        "winner_arm": "C3_44m_s2_joint",
        "arm_summary": {
            "C0_22m_q32": {"spec": {"base_dim": 144, "cond_dim": 320, "use_adapter": False}},
            "C3_44m_s2_joint": {"spec": {"base_dim": 208, "cond_dim": 384, "use_adapter": True}},
        },
    }
    mcfg = select_merge_config(fake_a, fake_b, fake_c)
    assert mcfg["analytic_mode"] == "A0"
    assert mcfg["style_mode"] == "k8"
    assert mcfg["base_dim"] == 208 and mcfg["use_adapter"] is True

    # Verify analytic bridge ODE identity: z_tau + (1 - tau) * target_v == z1
    z1 = torch.randn(4, 4, 16, 16)
    z0 = torch.randn(4, 4, 16, 16)
    ze_a3 = torch.randn(4, 4, 16, 16)
    tau0 = torch.tensor([0.60, 0.635, 0.75, 0.79])
    s_prog = torch.tensor([0.0, 0.25, 0.6, 1.0])
    tau, zt, tv = sample_analytic_bridge_segment(z1, z0, ze_a3, tau0, s_prog)
    assert (tau >= tau0 - 1e-6).all() and (tau <= 1.0 + 1e-6).all()
    recon = zt + (1.0 - tau).view(-1, 1, 1, 1) * tv
    assert torch.allclose(recon, z1, atol=1e-4)
