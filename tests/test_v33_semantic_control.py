"""Unit tests for CLEAVE V33 pre-flight verification:
1. Q64+ orthogonality, original Q32 invariance, and rank-degeneracy handling.
2. Zero-init Adapter parity with V32 PCoreSLFUNet (both Q64ResidualAdapter and SemanticControlAdapter).
3. CFG drop_content=True zero information leak.
4. s=0 / lambda=0 latent exact identity.
5. Flow time direction (tau0 -> 1.0) & S10B 0.18215 scaling interface.
6. Zero-DINO AST hygiene across exp/v33_semantic_control/.
"""
from __future__ import annotations

import ast
from pathlib import Path

import torch
import torch.nn as nn

from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v28_unified.student_models import UnifiedHybridUNet
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v33_semantic_control.models import (
    PCoreWithQ64Adapter,
    PCoreWithSemanticControl,
    Q64ResidualAdapter,
    SemanticControlAdapter,
    integrate_v33_trajectory,
    partial_whitening_residual,
    stiefel_q64_plus,
)
from src.utils.vae import StudentDecoder, decode_s10, s10_scaling_factor


def test_v33_zero_dino_ast_hygiene() -> None:
    v33_dir = Path(__file__).resolve().parents[1] / "exp/v33_semantic_control"
    for py_file in sorted(v33_dir.glob("*.py")):
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


def test_q64_plus_orthogonality_invariance_and_rank_degeneracy() -> None:
    torch.manual_seed(42)
    q_full, _ = torch.linalg.qr(torch.randn(384, 64), mode="reduced")
    w32, wb32 = q_full[:, :32], q_full[:, 32:64]

    # Full-rank random input
    p16 = torch.randn(3, 384, 16, 16)
    p16 = p16 - p16.mean(dim=(2, 3), keepdim=True)
    q32 = stiefel_polar_quotient(p16, w32)
    q64 = stiefel_q64_plus(p16, w32, wb32)
    assert q64.shape == (3, 64, 16, 16) and torch.isfinite(q64).all()
    assert torch.allclose(q64[:, :32], q32, atol=1e-6)
    gram64 = torch.bmm(q64.flatten(2), q64.flatten(2).transpose(1, 2)) / 256.0
    assert torch.allclose(gram64, torch.eye(64).unsqueeze(0).expand(3, -1, -1), atol=1e-4)

    # Rank-deficient input (spanning only 20 dims in wb32 subspace, or identical to w32 only)
    p16_deg = torch.einsum("dk,bkp->bdp", w32, torch.randn(2, 32, 256)).reshape(2, 384, 16, 16)
    p16_deg = p16_deg - p16_deg.mean(dim=(2, 3), keepdim=True)
    q32_deg = stiefel_polar_quotient(p16_deg, w32)
    q64_deg = stiefel_q64_plus(p16_deg, w32, wb32)
    assert torch.isfinite(q64_deg).all()
    assert torch.allclose(q64_deg[:, :32], q32_deg, atol=1e-6)
    gram64_deg = torch.bmm(q64_deg.flatten(2), q64_deg.flatten(2).transpose(1, 2)) / 256.0
    assert torch.allclose(gram64_deg, torch.eye(64).unsqueeze(0).expand(2, -1, -1), atol=1e-4)

    # Partial whitening residual: zero-mean & unit RMS calibrated
    pw = partial_whitening_residual(p16, w32, alpha=0.5)
    assert pw.shape == (3, 32, 16, 16) and torch.isfinite(pw).all()
    assert pw.mean(dim=(2, 3)).abs().max().item() < 1e-5
    rms = pw.flatten(1).square().mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.ones(3), atol=1e-4)


def test_adapters_zero_init_parity_and_cfg_drop_no_leak() -> None:
    torch.manual_seed(7)
    pcore = PCoreSLFUNet(base_dim=64, u_dim=417, cond_dim=128, use_gamma=True, normalize_u=True).eval()
    # Non-zero out_conv so base model output is non-trivial
    with torch.no_grad():
        pcore.out_conv.weight.normal_(0.0, 0.05)

    w_mid16, w_dec32 = 64 * 3, 64 * 2
    q64_adp = Q64ResidualAdapter(in_q_ch=64, w_mid16=w_mid16, cond_dim=128, inner_ch=96).eval()
    sem_ctrl = SemanticControlAdapter(d_q16=32, d_q32=32, w32=64, w16=96, pcore_w32=w_dec32, pcore_w16=w_mid16, cond_dim=128).eval()

    m_q64 = PCoreWithQ64Adapter(pcore, q64_adp).eval()
    m_sem = PCoreWithSemanticControl(pcore, sem_ctrl).eval()

    B = 2
    zt = torch.randn(B, 4, 64, 64)
    y16 = torch.randn(B, 1, 16, 16)
    q32 = torch.randn(B, 32, 16, 16)
    qe = torch.randn(B, 32, 16, 16)
    q32_hi = torch.randn(B, 32, 32, 32)
    tid = torch.tensor([0, 3])
    u = torch.randn(B, 417)
    tau = torch.tensor([0.3, 0.7])
    gamma = torch.tensor([0.8, 1.0])

    # 1. Exact zero-init parity with V32 PCoreSLFUNet
    v_base = pcore(zt, y16, q32, tid, u, tau, gamma=gamma)
    v_q64 = m_q64(zt, y16, q32, tid, u, tau, q_extra=qe, gamma=gamma)
    v_sem = m_sem(zt, y16, q32, tid, u, tau, q32_hi=q32_hi, gamma=gamma)
    assert torch.allclose(v_base, v_q64, atol=1e-6)
    assert torch.allclose(v_base, v_sem, atol=1e-6)

    # 2. Simulate trained (non-zero) adapters and verify drop_content=True has ZERO leak!
    with torch.no_grad():
        q64_adp.out_conv.weight.normal_(0.0, 0.05)
        sem_ctrl.zero_16.weight.normal_(0.0, 0.05)
        sem_ctrl.zero_32.weight.normal_(0.0, 0.05)

    dc = torch.ones(B, dtype=torch.bool)
    v_base_dc = pcore(zt, y16, q32, tid, u, tau, drop_content=dc, gamma=gamma)
    v_q64_dc_a = m_q64(zt, y16, q32, tid, u, tau, q_extra=qe, drop_content=dc, gamma=gamma)
    v_q64_dc_b = m_q64(zt, y16 * 5.0, q32 * 5.0, tid, u, tau, q_extra=qe * 9.0, drop_content=dc, gamma=gamma)
    v_sem_dc_a = m_sem(zt, y16, q32, tid, u, tau, q32_hi=q32_hi, drop_content=dc, gamma=gamma)
    v_sem_dc_b = m_sem(zt, y16 * 5.0, q32 * 5.0, tid, u, tau, q32_hi=q32_hi * 9.0, drop_content=dc, gamma=gamma)
    assert torch.allclose(v_base_dc, v_q64_dc_a, atol=1e-6)
    assert torch.allclose(v_q64_dc_a, v_q64_dc_b, atol=1e-6)
    assert torch.allclose(v_base_dc, v_sem_dc_a, atol=1e-6)
    assert torch.allclose(v_sem_dc_a, v_sem_dc_b, atol=1e-6)


def test_s0_identity_flow_direction_and_s10_scaling() -> None:
    torch.manual_seed(11)
    # 1. E-Core lambda=0 exact latent identity
    ec = UnifiedHybridUNet(stats_dim=16, w0=32, w1=48, w2=64, cond_dim=64).eval()
    z_src = torch.randn(2, 4, 64, 64)
    c64 = torch.randn(2, 4, 64, 64)
    q32 = torch.randn(2, 32, 16, 16)
    tid = torch.tensor([1, 2])
    lam0 = torch.zeros(2)
    raw_stats = torch.randn(5, 16)
    w_y = torch.tensor([1.0, 0.0, 0.0, 0.0])
    P_Y = torch.outer(w_y, w_y)
    P_C = torch.eye(4) - P_Y
    priors = {"P_Y": P_Y, "P_C": P_C, "mu_c": torch.zeros(5, 4, 1, 1), "std_c": torch.ones(5, 4, 1, 1)}
    z_out_s0 = ec(z_src, c64, q32, tid, lam0, raw_stats, priors, noise=torch.randn_like(z_src))
    assert torch.allclose(z_out_s0, z_src, atol=1e-6)

    # 2. Flow time direction: constant velocity v(z_t) = c -> z_1 = z_tau0 + (1 - tau0) * c
    class _ConstVel(nn.Module):
        use_gamma = True

        def __init__(self, target_z1: torch.Tensor):
            super().__init__()
            self.target = target_z1

        def forward(self, z_tau, y16, q32, tid, u, tau, **kwargs):
            # Exact noise-to-data velocity field toward self.target from tau0=0 eps=0
            return self.target.repeat(z_tau.shape[0] // self.target.shape[0], 1, 1, 1)

    target = torch.randn(2, 4, 64, 64)
    eps = torch.zeros_like(target)
    ze = torch.zeros_like(target)
    z_int = integrate_v33_trajectory(
        _ConstVel(target), ze, eps,
        torch.zeros(2, 1, 16, 16), torch.zeros(2, 32, 16, 16), tid, torch.zeros(2, 417),
        tau0=0.25, ws=1.0, wc=1.0, wsp=0.0, rescale_phi=0.0, steps=6, solver="dpm2m",
    )
    # From z(0.25) = 0 with constant v = target over dt = 0.75 -> z(1.0) = 0.75 * target
    assert torch.allclose(z_int, 0.75 * target, atol=1e-4)

    # 3. S10B / S10 scaling interface divides by s10_scaling_factor (0.18215) exactly once
    class _IdentityDec(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, z):
            return z[:, :3]

    dec = StudentDecoder(_IdentityDec(), nn.Identity()).eval()
    z_scaled = torch.full((1, 4, 8, 8), 0.18215 * 0.5)
    rgb01 = decode_s10(dec, z_scaled, dtype=torch.float32, batch=1, output_cpu=False)
    # decode_s10 maps [-1, 1] -> [0, 1] via (out + 1)/2, so 0.5 -> (0.5 + 1)/2 = 0.75
    assert abs(float(s10_scaling_factor()) - 0.18215) < 1e-6
    assert torch.allclose(rgb01, torch.full((1, 3, 8, 8), 0.75), atol=1e-4)


def test_paired_ci_vs_baseline_helper(tmp_path: Path) -> None:
    import json
    from exp.v33_semantic_control import paired_ci_vs_baseline

    rows_a, rows_b = [], []
    styles = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
    for s_idx in range(5):
        stem = f"src_{s_idx}"
        for t_idx in range(5):
            base = {
                "task_id": f"{stem}/{t_idx}",
                "src_stem": stem,
                "src_style": styles[s_idx],
                "tgt_style": styles[t_idx],
                "src_style_id": s_idx,
                "tgt_style_id": t_idx,
                "is_cross": s_idx != t_idx,
                "style_scores_5": [0.5] * 5,
                "clip_scores_5": [0.7] * 5,
                "csd_scores_5": [0.2] * 5,
                "dino_s": 0.5,
                "dino_c": 0.6,
                "lpips": 0.5,
                "target_top1": 1,
                "target_margin": 0.1,
                "dT": 0.1,
            }
            ra = dict(base)
            ra["dino_s"] = 0.55
            ra["style_scores_5"] = [0.55 if j == t_idx else 0.5 for j in range(5)]
            rows_a.append(ra)
            rows_b.append(base)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps({"rows": rows_a}), encoding="utf-8")
    pb.write_text(json.dumps({"rows": rows_b}), encoding="utf-8")
    res = paired_ci_vs_baseline(pa, pb)
    assert res["n_tasks"] == 20 and abs(res["dino_s"]["diff"] - 0.05) < 1e-6 and res["dino_s"]["excludes_zero"] is True


def test_compute_style_pixel_stats_rgb_u8(tmp_path: Path) -> None:
    from exp.v33_semantic_control import compute_style_pixel_stats
    from src.utils.raw_io import RawWriter
    from src.utils.runtime import read_json

    tasks = read_json(Path(__file__).resolve().parents[1] / "eval_protocol/tasks_750.json")["tasks"]
    min_cross = [t for t in tasks if t["target_style_id"] == 2 and t["source_style_id"] != 2][:4]
    w = RawWriter("pix_test", raw_root=tmp_path, expected_tasks=4)
    rgb = torch.zeros(4, 3, 16, 16)
    rgb[:, 0] = 0.8
    rgb[:, 1] = 0.2
    rgb[:, 2] = 0.2
    w.append([t["task_id"] for t in min_cross], rgb)
    w.finish()
    stats = compute_style_pixel_stats(tmp_path / "pix_test", target_style_id=2, cross_only=True)
    assert stats["n"] == 4 and stats["sat_hsv"] > 0.7 and stats["chroma_oklab"] > 0.1
