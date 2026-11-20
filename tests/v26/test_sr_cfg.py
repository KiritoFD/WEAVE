"""V26 契约 1–4: SR-CFG 自参考恒等 / 色度投影 / 分支数 (CPU)."""
import math

import pytest
import torch
import torch.nn as nn

from exp.v26.sr_cfg import SR_ARMS, apply_proj, chroma_projector, sr_velocity


class MockFM(nn.Module):
    """与 SLFUNetV24OneShot 调用签名兼容的可区分分支 mock: 输出依赖
    (z, drop_style, v16_ref 均值, u_glob 均值) — 条件相同则输出逐位相同."""

    def forward(self, z, y16, d16, sid, u, tau, *, drop_style=None, drop_content=None,
                layout_scale=None, d16_ref=None, v16_ref=None, u_glob=None):
        ds = drop_style.float().view(-1, 1, 1, 1)
        if v16_ref is not None:
            ref = v16_ref.mean(dim=(2, 3))[:, :1].view(-1, 1, 1, 1)
        else:
            ref = torch.zeros(z.shape[0], 1, 1, 1)
        if u_glob is not None:
            g = u_glob.mean(dim=-1).view(-1, 1, 1, 1)
        else:
            g = torch.zeros(z.shape[0], 1, 1, 1)
        return 0.5 * z + (1.0 - ds) * (0.1 * ref + 0.2 * g) + ds * 0.05


@pytest.fixture()
def sr_inputs():
    g = torch.Generator().manual_seed(20261185)
    B = 2
    return dict(
        d16=torch.randn(B, 384, 16, 16, generator=g),
        v16_t=torch.randn(B, 192, 16, 16, generator=g),
        v16_s=torch.randn(B, 192, 16, 16, generator=g),
        g_t=torch.randn(B, 128, generator=g),
        g_s=torch.randn(B, 128, generator=g),
        u1=torch.randn(B, 128, generator=g),
        y16=torch.randn(B, 1, 16, 16, generator=g),
        zz=torch.randn(B, 4, 64, 64, generator=g),
        tv=torch.rand(B, generator=g) * 0.4 + 0.45,
    )


def test_sr_selfref_zero_diff_w_independent(sr_inputs):
    """契约 1a: 参考 = 源 (全条件相同) → v_sr 与 w 无关 (== v_pos, 差分精确为零)."""
    s = sr_inputs
    common = dict(mode="sr", pc=None, device="cpu", wj=2.2)
    args = (MockFM(), None, s["u1"], s["u1"], s["g_t"], s["g_t"], s["d16"],
            s["d16"], s["v16_t"], s["v16_t"], s["d16"], s["y16"], s["zz"], s["tv"])
    v03, _ = sr_velocity(*args, w=0.3, **common)
    v09, _ = sr_velocity(*args, w=0.9, **common)
    v00, _ = sr_velocity(*args, w=0.0, **common)
    assert torch.equal(v03, v09), "自参考差分非零: w 改变了速度场"
    assert torch.equal(v03, v00), "自参考差分非零: v_sr != v_pos"


def test_sr_crossref_w_dependent(sr_inputs):
    """契约 1b: 参考 ≠ 源 → v_sr 随 w 变化 (排斥方向真实注入)."""
    s = sr_inputs
    common = dict(mode="sr", pc=None, device="cpu", wj=2.2)
    args = (MockFM(), None, s["u1"], s["u1"], s["g_t"], s["g_s"], s["d16"],
            s["d16"], s["v16_t"], s["v16_s"], s["d16"], s["y16"], s["zz"], s["tv"])
    v03, _ = sr_velocity(*args, w=0.3, **common)
    v09, _ = sr_velocity(*args, w=0.9, **common)
    assert not torch.allclose(v03, v09), "跨参考条件下 w 无效应 (差分通道失效?)"
    assert torch.isfinite(v03).all() and torch.isfinite(v09).all()


def test_chroma_projector_properties():
    """契约 2: P_C 对 wy 零投影 / 对称 / 幂等."""
    w = torch.tensor([0.3, -0.5, 0.8, 0.11])
    pc = chroma_projector(w)
    wy = w / w.norm()
    assert float((pc @ wy).abs().max()) < 1e-6, "P_C 未消除亮度轴"
    assert torch.allclose(pc, pc.T, atol=1e-6), "P_C 非对称"
    assert torch.allclose(pc @ pc, pc, atol=1e-5), "P_C 非幂等"


def test_apply_proj_removes_luma():
    g = torch.Generator().manual_seed(7)
    w = torch.randn(4, generator=g)
    pc = chroma_projector(w)
    wy = w / w.norm()
    v = torch.randn(2, 4, 8, 8, generator=g)
    proj = apply_proj(v, pc)
    luma_left = (proj * wy.view(1, 4, 1, 1)).sum(dim=1)
    assert float(luma_left.abs().max()) < 1e-5, "亮度分量未被投影移除"
    assert float((v - proj - (v * wy.view(1, 4, 1, 1)).sum(dim=1, keepdim=True) * wy.view(1, 4, 1, 1)).abs().max()) < 1e-5


def test_branch_counts(sr_inputs):
    """契约 4: baseline/sr 同 NFE (2 分支), sr3 为 3 分支 (1.5×)."""
    s = sr_inputs
    mock = MockFM()
    _, n_b = sr_velocity(mock, None, s["u1"], None, s["g_t"], None, s["d16"], s["d16"],
                         s["v16_t"], None, None, s["y16"], s["zz"], s["tv"],
                         mode="baseline", w=0.0, wj=2.2, device="cpu")
    _, n_sr = sr_velocity(mock, None, s["u1"], s["u1"], s["g_t"], s["g_t"], s["d16"],
                          s["d16"], s["v16_t"], s["v16_t"], s["d16"], s["y16"], s["zz"], s["tv"],
                          mode="sr", w=0.6, wj=2.2, device="cpu")
    _, n_3 = sr_velocity(mock, None, s["u1"], s["u1"], s["g_t"], s["g_t"], s["d16"],
                         s["d16"], s["v16_t"], s["v16_t"], s["d16"], s["y16"], s["zz"], s["tv"],
                         mode="sr3", w=0.6, wj=2.2, device="cpu")
    assert n_b == 2 and n_sr == 2 and n_3 == 3


def test_sr_arm_registry_valid():
    """契约 12a: SR 臂矩阵预注册完整性 (SR3 w=1.0 用于自参考审计; 投影合法)."""
    assert SR_ARMS["SR3_w10"]["w"] == 1.0
    assert SR_ARMS["SR5_w06_pc"]["proj"] == "chroma" and SR_ARMS["SR6_w10_pc"]["proj"] == "chroma"
    for arm, spec in SR_ARMS.items():
        assert spec["mode"] in ("baseline", "sr", "sr3"), arm
        assert spec["proj"] in ("none", "chroma"), arm
        assert spec["w"] >= 0.0


def test_baseline_cfg_direction(sr_inputs):
    """契约 15 (回归): baseline CFG 必须 = v_uncond + wj*(v_cond - v_uncond), 不得反转.

    回归背景: chunk 顺序反转曾把 CFG 写成负方向, 导致 FS/基线全部近恒等输出.
    """
    from exp.v26.sr_cfg import v_0c_ret
    s = sr_inputs
    mock = MockFM()
    v_base, _ = sr_velocity(mock, None, s["u1"], None, s["g_t"], None, s["d16"], s["d16"],
                            s["v16_t"], None, None, s["y16"], s["zz"], s["tv"],
                            mode="baseline", w=0.0, wj=2.2, device="cpu")
    # 手工按 mock 语义重算 cond / uncond 两分支
    def branch(drop_style_flag):
        ds = torch.tensor(float(drop_style_flag), dtype=torch.float32).view(1, 1, 1, 1).expand_as(s["zz"])
        ref = s["v16_t"].mean(dim=(2, 3))[:, :1].view(-1, 1, 1, 1)
        g = s["g_t"].mean(dim=-1).view(-1, 1, 1, 1)
        return 0.5 * s["zz"] + (1.0 - ds) * (0.1 * ref + 0.2 * g) + ds * 0.05
    v_cond, v_uncond = branch(False), branch(True)
    expect = v_uncond + 2.2 * (v_cond - v_uncond)
    assert torch.allclose(v_base, expect, atol=1e-5), "baseline CFG 方向错误 (chunk 反转回归)"
    assert torch.allclose(v_base, v_0c_ret(v_cond, v_uncond, 2.2), atol=1e-6)
