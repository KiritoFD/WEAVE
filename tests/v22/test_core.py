"""V22-P0 数值与实现单测（P0.2 CFG 分支 / P0.3 Flow 数值 / V20 逐位等价 / Fix 4-5）。

全部可在 CPU 运行（autocast 在 CPU 为 nullcontext，fp32 精确）；亦可在 4090/3060 GPU 上运行。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v22.core import (
    CFGConfig,
    D5_SOLVER,
    GateConfig,
    PrismCore,
    effective_gate_components,
    predict_guided_velocity,
    rk2_integrate,
)
from exp.v22 import validate_raw_meta


def _tiny_slf(base_dim: int = 32, seed: int = 0):
    from exp.v20.slf_flow import SLFUNet

    torch.manual_seed(seed)
    return SLFUNet(base_dim=base_dim, u_dim=128, cond_dim=128).eval()


def _fake_bank(k_sub: int = 8, scene_dim: int = 6144, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    mid = 0.035
    sub_dsd = mid + 0.006 * torch.randn(5, k_sub, generator=g)
    return {
        "k_sub": k_sub,
        "sub_u": F.normalize(torch.randn(5, k_sub, 128, generator=g), dim=-1),
        "sub_scene": F.normalize(torch.randn(5, k_sub, scene_dim, generator=g), dim=-1),
        "sub_dsd": sub_dsd,
        "mid_dsd": mid,
        "u_exact_all": F.normalize(torch.randn(64, 128, generator=g), dim=-1),
        "core_idx_per_style": [torch.arange(4, dtype=torch.long) + 8 * s for s in range(5)],
    }


def _inputs(B=3, seed=7, scene_dim=6144):
    g = torch.Generator().manual_seed(seed)
    return dict(
        z_t4=torch.randn(B, 4, 64, 64, generator=g) * 0.6,
        eps=torch.randn(B, 4, 64, 64, generator=g),
        y16=torch.randn(B, 1, 16, 16, generator=g) * 0.5,
        d16=torch.randn(B, 384, 16, 16, generator=g),
        tid=torch.tensor([0, 2, 4][:B]),
    )


def f0(B):
    return torch.zeros(B, dtype=torch.bool)


def f1(B):
    return torch.ones(B, dtype=torch.bool)


# ---------------- P0.2: CFG 四分支 ----------------

def test_four_branch_batched_equals_independent():
    model = _tiny_slf()
    inp = _inputs(B=3)
    B = 3
    z, y, d, tid = inp["z_t4"], inp["y16"], inp["d16"], inp["tid"]
    u = F.normalize(torch.randn(B, 128, generator=torch.Generator().manual_seed(3)), dim=-1)
    tau = torch.tensor([0.3, 0.5, 0.7])
    lscale = torch.tensor([0.9, 0.7, 1.0])
    ls0 = torch.zeros(B)

    # batched（V20 顺序）
    z4, y4 = z.repeat(4, 1, 1, 1), y.repeat(4, 1, 1, 1)
    d4, s4, u4, t4 = d.repeat(4, 1, 1, 1), tid.repeat(4), u.repeat(4, 1), tau.repeat(4)
    ds = torch.cat([f0(B), f0(B), f1(B), f1(B)], 0)
    dc = torch.cat([f0(B), f1(B), f0(B), f1(B)], 0)
    lsc4 = torch.cat([lscale, ls0, lscale, ls0], 0)
    v_sc, v_s0, v_0c, v_00 = model(z4, y4, d4, s4, u4, t4, drop_style=ds, drop_content=dc, layout_scale=lsc4).chunk(4, 0)

    # independent
    vsc_i = model(z, y, d, tid, u, tau, drop_style=f0(B), drop_content=f0(B), layout_scale=lscale)
    vs0_i = model(z, y, d, tid, u, tau, drop_style=f1(B), drop_content=f1(B), layout_scale=ls0)
    v0c_i = model(z, y, d, tid, u, tau, drop_style=f1(B), drop_content=f0(B), layout_scale=lscale)
    v00_i = model(z, y, d, tid, u, tau, drop_style=f1(B), drop_content=f1(B), layout_scale=ls0)

    assert torch.allclose(v_sc, vsc_i, atol=1e-6)
    assert torch.allclose(v_s0, vs0_i, atol=1e-6)
    assert torch.allclose(v_0c, v0c_i, atol=1e-6)
    assert torch.allclose(v_00, v00_i, atol=1e-6)


def test_drop_style_invariance():
    model = _tiny_slf()
    inp = _inputs(B=2)
    z, y, d, tid = inp["z_t4"], inp["y16"], inp["d16"], inp["tid"]
    ds = f1(2)
    u1 = F.normalize(torch.randn(2, 128, generator=torch.Generator().manual_seed(1)), dim=-1)
    u2 = F.normalize(torch.randn(2, 128, generator=torch.Generator().manual_seed(2)), dim=-1)
    tau = torch.tensor([0.4, 0.6])
    v1 = model(z, y, d, tid, u1, tau, drop_style=ds, drop_content=f0(2), layout_scale=torch.ones(2))
    v2 = model(z, y, d, tid + 1, u2, tau, drop_style=ds, drop_content=f0(2), layout_scale=torch.ones(2))
    assert torch.allclose(v1, v2, atol=1e-6), "drop_style 后输出不得依赖 tid/u"


def test_drop_content_invariance():
    model = _tiny_slf()
    inp = _inputs(B=2)
    z, tid = inp["z_t4"], inp["tid"]
    dc = f1(2)
    u = F.normalize(torch.randn(2, 128, generator=torch.Generator().manual_seed(4)), dim=-1)
    tau = torch.tensor([0.4, 0.6])
    v1 = model(z, inp["y16"], inp["d16"], tid, u, tau, drop_style=f0(2), drop_content=dc, layout_scale=None)
    v2 = model(z, inp["y16"] * 7 + 3, inp["d16"] * -2 + 1, tid, u, tau, drop_style=f0(2), drop_content=dc, layout_scale=None)
    assert torch.allclose(v1, v2, atol=1e-6), "drop_content 且 lscale=0 后输出不得依赖内容条件"


def test_combination_algebra_identity():
    model = _tiny_slf()
    inp = _inputs(B=2)
    z, y, d, tid = inp["z_t4"], inp["y16"], inp["d16"], inp["tid"]
    u = F.normalize(torch.randn(2, 128, generator=torch.Generator().manual_seed(5)), dim=-1)
    tau = torch.tensor([0.35, 0.65])
    lscale = torch.tensor([0.8, 1.0])
    B = 2
    z4, y4 = z.repeat(4, 1, 1, 1), y.repeat(4, 1, 1, 1)
    d4, s4, u4, t4 = d.repeat(4, 1, 1, 1), tid.repeat(4), u.repeat(4, 1), tau.repeat(4)
    ds = torch.cat([f0(B), f0(B), f1(B), f1(B)], 0)
    dc = torch.cat([f0(B), f1(B), f0(B), f1(B)], 0)
    lsc4 = torch.cat([lscale, torch.zeros(B), lscale, torch.zeros(B)], 0)
    v_sc, v_s0, v_0c, v_00 = model(z4, y4, d4, s4, u4, t4, drop_style=ds, drop_content=dc, layout_scale=lsc4).chunk(4, 0)

    one = torch.ones(B, 1, 1, 1)
    lhs = v_00 + one * (v_0c - v_00) + one * (v_sc - v_0c) + one * (v_s0 - v_00)
    rhs = v_sc + v_s0 - v_00
    assert torch.allclose(lhs, rhs, atol=1e-5), "wc=wj=wp=1 的组合必须满足代数恒等式"


def test_predict_guided_velocity_matches_manual_combination():
    model = _tiny_slf()
    inp = _inputs(B=2)
    z, y, d, tid = inp["z_t4"], inp["y16"], inp["d16"], inp["tid"]
    u = F.normalize(torch.randn(2, 128, generator=torch.Generator().manual_seed(6)), dim=-1)
    tau_v = torch.tensor([0.4, 0.55])
    lscale = torch.tensor([0.85, 0.7])
    g_guid = torch.tensor([0.8, 0.3])
    content = {"y_in": y, "d_in": d}
    solver = dict(wc_fig=1.85, wc_abs=0.8, ws_fig=4.2, wsp_fig_late=2.4, wsp_abs=5.2)
    cfg = CFGConfig(branches="full", rescale_phi=0.48)
    prog = 0.62

    v = predict_guided_velocity(model, z, tau_v, content, tid, u, lscale, g_guid, cfg, solver, prog)

    B = 2
    gf = g_guid.view(B, 1, 1, 1)
    gl = min(1.0, max(0.0, (prog - 0.36) / 0.32))
    wc = gf * (solver["wc_fig"] - 0.45 * gl) + (1 - gf) * solver["wc_abs"]
    ws_j = gf * (solver["ws_fig"] - 1.1 * gl) + (1 - gf) * 1.4
    ws_p = gf * (gl * solver["wsp_fig_late"]) + (1 - gf) * solver["wsp_abs"]

    ls0 = torch.zeros(B)
    vsc = model(z, y, d, tid, u, tau_v, drop_style=f0(B), drop_content=f0(B), layout_scale=lscale)
    vs0 = model(z, y, d, tid, u, tau_v, drop_style=f1(B), drop_content=f1(B), layout_scale=ls0)
    v0c = model(z, y, d, tid, u, tau_v, drop_style=f1(B), drop_content=f0(B), layout_scale=lscale)
    v00 = model(z, y, d, tid, u, tau_v, drop_style=f1(B), drop_content=f1(B), layout_scale=ls0)
    s_ref = (gf * vsc + (1 - gf) * vs0).flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
    vr_manual = v00 + wc * (v0c - v00) + ws_j * (vsc - v0c) + ws_p * (vs0 - v00)
    s_raw = vr_manual.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
    vr_manual = cfg.rescale_phi * vr_manual * (s_ref / s_raw) + (1 - cfg.rescale_phi) * vr_manual
    assert torch.allclose(v, vr_manual, atol=1e-6)


# ---------------- P0.3: Flow 数值 ----------------

def test_rk2_constant_field_exact():
    tau = torch.tensor([0.25, 0.5])
    c = torch.randn(2, 4, 8, 8)
    z0 = torch.randn(2, 4, 8, 8)
    z = rk2_integrate(lambda zz, prog: c, z0, tau, steps=5)
    expected = z0 + c * (1.0 - tau).view(-1, 1, 1, 1)
    assert torch.allclose(z, expected, atol=1e-6), "常速度场 RK2 必须精确"


def test_rk2_linear_field_exact():
    a = torch.randn(2, 4, 8, 8)
    tau = torch.tensor([0.3, 0.6])
    z0 = torch.randn(2, 4, 8, 8)
    z = rk2_integrate(lambda zz, prog: a * (tau + prog * (1 - tau)).view(-1, 1, 1, 1), z0, tau, steps=4)
    expected = z0 + a * ((1.0 - tau**2) / 2.0).view(-1, 1, 1, 1)
    assert torch.allclose(z, expected, atol=1e-6), "线性时间场中点律必须精确"


def test_rk2_per_sample_tau():
    # v = t 场: z(1) = z0 + (1 - tau^2)/2，逐样本 tau 不同
    tau = torch.tensor([0.1, 0.7])
    z0 = torch.randn(2, 3, 4, 4)
    z = rk2_integrate(
        lambda zz, prog: (tau + prog * (1 - tau)).view(-1, 1, 1, 1).expand_as(zz),
        z0, tau, steps=6,
    )
    expected = z0 + ((1.0 - tau**2) / 2.0).view(-1, 1, 1, 1)
    assert torch.allclose(z, expected, atol=1e-6)


def test_rk2_heun_convergence():
    # z' = -z, 精确解 z0*exp(-(t-tau0))
    tau = torch.tensor([0.2])
    z0 = torch.randn(1, 2, 4, 4)
    exact = z0 * torch.exp(-(1.0 - tau)).view(-1, 1, 1, 1)

    def err(steps):
        z = rk2_integrate(lambda zz, prog: -zz, z0, tau, steps=steps)
        return float((z - exact).abs().max())

    e1, e2 = err(2), err(8)
    assert e2 < e1, "步长减小时 RK2 误差必须下降"
    assert e2 < 5e-3


def test_rescale_phi_zero_variance_stability():
    class ConstModel(nn.Module):
        def forward(self, z, y, d, s, u, t, **kw):
            B = z.shape[0]
            return torch.ones(B, 4, z.shape[-2], z.shape[-1]) * 0.3

    inp = _inputs(B=2)
    core = PrismCore(core_bank=_fake_bank(), model=ConstModel(), device="cpu")
    z, stats = core.integrate_flow(inp["eps"], inp["z_t4"], inp["y16"], inp["d16"], inp["tid"], solver=dict(D5_SOLVER, steps=3))
    assert torch.isfinite(z).all(), "零方差速度下 rescale 路径不得产生 NaN/Inf"


def test_integrate_flow_deterministic():
    torch.manual_seed(11)
    model = _tiny_slf(seed=11)
    core = PrismCore(core_bank=_fake_bank(), model=model, device="cpu")
    inp = _inputs(B=2, seed=9)
    kw = dict(solver=dict(D5_SOLVER, steps=4))
    z1, _ = core.integrate_flow(inp["eps"].clone(), inp["z_t4"].clone(), inp["y16"], inp["d16"], inp["tid"], **kw)
    z2, _ = core.integrate_flow(inp["eps"].clone(), inp["z_t4"].clone(), inp["y16"], inp["d16"], inp["tid"], **kw)
    assert torch.equal(z1, z2), "同 seed 同条件推理必须逐位确定"


# ---------------- V20 逐位等价 ----------------

def test_equivalence_with_v20_integrate():
    from exp.v20.slf_flow import integrate_slf_core_batch

    model = _tiny_slf(seed=13)
    bank = _fake_bank()
    core = PrismCore(core_bank=bank, model=model, device="cpu")
    inp = _inputs(B=3, seed=15)
    solver_kw = dict(
        tau_min=0.28, tau_hi=0.66, steps=6,
        ws_fig=4.2, wsp_fig_late=2.4, wsp_abs=5.2,
        wc_fig=1.85, wc_abs=0.80, lscale_abs=0.34, rescale_phi=0.48,
    )
    z_v20 = integrate_slf_core_batch(model, inp["z_t4"], inp["eps"], inp["y16"], inp["d16"], inp["tid"], bank, **solver_kw)
    z_v22, stats = core.integrate_flow(inp["eps"], inp["z_t4"], inp["y16"], inp["d16"], inp["tid"], solver=dict(solver_kw))
    max_diff = float((z_v20 - z_v22).abs().max())
    assert max_diff < 1e-5, f"V22 求解器与 V20 参考实现逐位等价失败: max_diff={max_diff}"


# ---------------- 门控解耦 ----------------

def test_gate_components_decoupling():
    dsd = torch.tensor([0.030, 0.038])
    bank_mid = 0.035
    g_tau, g_layout, g_guid, _ = effective_gate_components(dsd, GateConfig(mode="original"), bank_mid)
    raw = torch.sigmoid((dsd - bank_mid) * 550.0)
    assert torch.allclose(g_tau, raw) and torch.allclose(g_layout, raw) and torch.allclose(g_guid, raw)

    fixed = GateConfig(mode="fixed", fixed_g=0.5)
    g_tau2, g_layout2, g_guid2, _ = effective_gate_components(dsd, fixed, bank_mid)
    assert torch.allclose(g_tau2, torch.full_like(dsd, 0.5))
    assert torch.allclose(g_layout2, torch.full_like(dsd, 0.5))

    # C1: 仅 tau 自适应
    c1 = GateConfig(mode="original", drive_layout=False, drive_guidance=False)
    g_tau3, g_layout3, g_guid3, _ = effective_gate_components(dsd, c1, bank_mid)
    assert torch.allclose(g_tau3, raw)
    assert torch.allclose(g_layout3, torch.full_like(dsd, 0.5))
    assert torch.allclose(g_guid3, torch.full_like(dsd, 0.5))


# ---------------- Fix 4 / Fix 5 ----------------

def test_validate_raw_meta():
    good = {"batches": [{"ids": [f"t{i}" for i in range(750)]}]}
    assert validate_raw_meta(good)
    dup = {"batches": [{"ids": ["x"] * 750}]}
    assert not validate_raw_meta(dup)
    short = {"batches": [{"ids": [f"t{i}" for i in range(749)]}]}
    assert not validate_raw_meta(short)


def test_v21_reference_sampling_without_replacement():
    from exp.v21.prism_attn_flow import sample_without_replacement

    idx = sample_without_replacement(batch=64, pool=12, n_shots=4, device="cpu")
    assert idx.shape == (64, 4)
    for row in idx:
        assert len(set(row.tolist())) == 4, "4-Shot 参考必须无放回（无重复）"
