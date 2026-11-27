"""Unit tests for CLEAVE V35 deploy bundle, E-only isolation, invariants, and zero-sync ODE."""
from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from exp.deploy_v35.bundle import build_deploy_bundle, verify_bundle_manifest
from exp.deploy_v35.eval_gate import interp_matched_c, paired_cluster_bootstrap
from exp.deploy_v35.pipeline import _precompute_dpm2m_schedule
from src.utils.runtime import save_json


def test_precompute_dpm2m_schedule_grid():
    rows = _precompute_dpm2m_schedule(6, 0.785, ws=3.5, wc=2.4, wsp=1.35, startup="midpoint")
    assert len(rows) == 6
    assert rows[0]["need_midpoint"] is True
    assert all(not r["need_midpoint"] for r in rows[1:])
    assert abs(rows[0]["t0"] - 0.785) < 1e-6
    assert abs(sum(r["dp"] for r in rows) - 1.0) < 1e-6

    rows_euler = _precompute_dpm2m_schedule(6, 0.785, ws=3.5, wc=2.4, wsp=1.35, startup="euler")
    assert all(not r["need_midpoint"] for r in rows_euler)


def test_paired_bootstrap_and_matched_c():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rows_ref, rows_cand = [], []
        for s_idx in range(150):
            stem = f"src_{s_idx:03d}"
            for t_id in range(5):
                is_c = (t_id != (s_idx % 5))
                rows_ref.append({
                    "task_id": f"S/{stem}/T{t_id}",
                    "src_stem": stem,
                    "tgt_style_id": t_id,
                    "is_cross": is_c,
                    "target_top1": 1 if (t_id % 2 == 0) else 0,
                    "dino_s": 0.55,
                    "dino_c": 0.62,
                    "lpips": 0.50,
                    "clip_scores_5": [0.70] * 5,
                })
                rows_cand.append({
                    **rows_ref[-1],
                    "dino_s": 0.551,
                })
        save_json(tmp / "ref.json", {"rows": rows_ref})
        save_json(tmp / "cand.json", {"rows": rows_cand})
        boot = paired_cluster_bootstrap(tmp / "cand.json", tmp / "ref.json", n_boot=200)
        assert boot["non_inferiority_pass"] is True
        assert abs(boot["cross600"]["dino_s"]["delta"] - 0.001) < 1e-6

    mc = interp_matched_c([(0.50, 0.58, 70.0), (0.60, 0.55, 60.0), (0.70, 0.50, 45.0)], c_targets=(0.55, 0.65))
    assert abs(mc["C=0.55"]["dino_s"] - 0.565) < 1e-6
    assert abs(mc["C=0.65"]["csd_top1_pct"] - 52.5) < 1e-6


def test_bundle_roundtrip_if_assets_exist():
    from exp.deploy_v35 import P8_LOCKED_CFG
    from src.utils.runtime import REPO as ROOT
    if not (ROOT / P8_LOCKED_CFG["pcore_ckpt"]).exists():
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        manifest = build_deploy_bundle(out, overwrite=True)
        assert manifest["version"] == "cleave_v35_final"
        verified = verify_bundle_manifest(out)
        assert verified["total_bytes"] == manifest["total_bytes"]


def test_sdpa_and_hoisted_pcore_equivalence():
    from exp.v29_biglittle.pcore_model import PCoreSLFUNet
    from exp.deploy_v35.torch_opt import HoistedPCoreStep
    m = PCoreSLFUNet(base_dim=64, u_dim=417, cond_dim=128, use_gamma=True, normalize_u=True).eval()
    h = HoistedPCoreStep(m, dtype=torch.float32).eval()
    B = 2
    y16 = torch.randn(B, 1, 16, 16)
    q32 = torch.randn(B, 32, 16, 16)
    tid = torch.tensor([0, 3])
    u = torch.randn(B, 417)
    f0 = torch.zeros(B, dtype=torch.bool)
    f1 = torch.ones(B, dtype=torch.bool)
    ds4 = torch.cat([f0, f0, f1, f1], 0)
    dc4 = torch.cat([f0, f1, f0, f1], 0)
    lsc4 = torch.cat([torch.full((B,), 0.8), torch.zeros(B), torch.full((B,), 0.8), torch.zeros(B)], 0)
    z1 = torch.randn(B, 4, 64, 64)
    t1 = torch.full((B,), 0.65)
    with torch.inference_mode():
        gc1, bc1, gc2, bc2, cb4 = h.prepare_traj(y16, q32, tid, u, ds4, dc4, lsc4, 0.35)
        v_h = h(z1, t1, cb4, gc1, bc1, gc2, bc2)
        v_r = m(
            z1.repeat(4, 1, 1, 1), y16.repeat(4, 1, 1, 1), q32.repeat(4, 1, 1, 1),
            tid.repeat(4), u.repeat(4, 1), t1.repeat(4),
            drop_style=ds4, drop_content=dc4, layout_scale=lsc4, gamma=torch.full((4 * B,), 0.35),
        )
    assert float((v_h - v_r).abs().max().item()) < 1e-4
