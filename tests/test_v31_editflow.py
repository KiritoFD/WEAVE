"""Unit tests for V31 EditFlowPCoreUNet, RCSM 3 causal controls, and RegionalCorrectionNet."""
from __future__ import annotations

import torch

from exp.v31_editflow.data_5style import extract_patch16_style_stats
from exp.v31_editflow.models import EditFlowPCoreUNet, RCSMModule, RegionalCorrectionNet


def test_rcsm_controls_and_editflow_shapes():
    rgb = torch.rand(2, 3, 512, 512)
    st16 = extract_patch16_style_stats(rgb)
    assert st16.shape == (2, 16, 16, 16) and torch.isfinite(st16).all()

    m_base = EditFlowPCoreUNet(base_dim=144, u_dim=417, cond_dim=320, edit_flow=True, use_rcsm=False).eval()
    m_rcsm = EditFlowPCoreUNet(base_dim=144, u_dim=417, cond_dim=320, edit_flow=True, use_rcsm=True).eval()
    assert 22_000_000 <= m_rcsm.n_params() <= 26_000_000

    z = torch.randn(2, 4, 64, 64)
    y16 = torch.randn(2, 1, 16, 16)
    q32 = torch.randn(2, 32, 16, 16)
    tid = torch.tensor([0, 2], dtype=torch.long)
    u_st = torch.randn(2, 417)
    tu = torch.tensor([0.2, 0.8])
    tau0 = torch.tensor([0.75, 0.0])
    kp = torch.randn(2, 32, 32)
    mp = torch.randn(2, 32, 16)

    with torch.no_grad():
        v_base = m_base(z, y16, q32, tid, u_st, tu, tau0=tau0)
        assert v_base.shape == (2, 4, 64, 64) and torch.isfinite(v_base).all()
        for mode in ("matched", "random", "global_avg"):
            v_r = m_rcsm(z, y16, q32, tid, u_st, tu, tau0=tau0, k_proto=kp, m_proto=mp, rcsm_mode=mode)
            assert v_r.shape == (2, 4, 64, 64) and torch.isfinite(v_r).all()
        # Verify zero RCSM style leakage when drop_style=True across different k_proto/m_proto
        ds1 = torch.ones(2, dtype=torch.bool)
        v_ds_a = m_rcsm(z, y16, q32, tid, u_st, tu, tau0=tau0, k_proto=kp, m_proto=mp, drop_style=ds1)
        v_ds_b = m_rcsm(z, y16, q32, tid, u_st, tu, tau0=tau0, k_proto=kp * 3.0, m_proto=mp + 5.0, drop_style=ds1)
        assert torch.allclose(v_ds_a, v_ds_b)

    rc = RCSMModule()
    f_match = rc.compute_region_stats_map(q32, kp, mp, mode="matched")
    f_glob = rc.compute_region_stats_map(q32, kp, mp, mode="global_avg")
    f_rand = rc.compute_region_stats_map(q32, kp, mp, mode="random")
    assert not torch.allclose(f_match, f_glob) and not torch.allclose(f_match, f_rand)

    rnet = RegionalCorrectionNet().eval()
    assert 3_000_000 <= rnet.n_params() <= 6_500_000
    with torch.no_grad():
        g_out = rnet(z, y16, q32, tid, u_st, tu, tau0)
    assert g_out.shape == (2, 4, 64, 64) and torch.isfinite(g_out).all()
