"""V26 契约 12–13: 少步注册表一致性 + Euler/RK2 常速场等价 (CPU)."""
import torch

from exp.v26.sr_cfg import FEW_STEP_ARMS, integrate_sr


def test_few_step_registry_valid():
    """契约 12b: 少步臂预注册完整 (步数 ≥ 2, 采样器合法, 覆盖 RK2 降步与 Euler)."""
    assert set(FEW_STEP_ARMS) == {"FS_t45_rk2_10", "FS_t45_rk2_08", "FS_t45_rk2_06",
                                  "FS_t45_rk2_04", "FS_t45_eul_04"}
    for arm, spec in FEW_STEP_ARMS.items():
        assert spec["steps"] >= 2, arm
        assert spec["sampler"] in ("rk2", "euler"), arm


def test_constant_field_euler_equals_rk2():
    """契约 13: 常速度场下 Euler 与 RK2 逐步等价 (中点值 == 端点值)."""
    tau0, steps = 0.45, 6
    z0 = torch.randn(2, 4, 64, 64, generator=torch.Generator().manual_seed(1))
    const = torch.ones(2, 4, 64, 64)

    def vel(zz, prog):
        return const

    ze = integrate_sr(vel, z0, tau0, steps, "euler")
    zr = integrate_sr(vel, z0, tau0, steps, "rk2")
    assert torch.allclose(ze, zr, atol=1e-6)


def test_integrate_determinism_and_progression():
    """契约 13b: 相同输入两次积分逐位一致; 步数减少 → 终点不同 (轨迹真实改变)."""
    tau0 = 0.5
    z0 = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(2))
    g = torch.Generator().manual_seed(3)
    w = torch.randn(1, 4, 32, 32, generator=g)

    def vel(zz, prog):
        return 0.1 * w + 0.01 * zz

    a = integrate_sr(vel, z0, tau0, 4, "rk2")
    b = integrate_sr(vel, z0, tau0, 4, "rk2")
    assert torch.equal(a, b)
    c = integrate_sr(vel, z0, tau0, 2, "rk2")
    assert not torch.allclose(a, c), "不同步数给出相同终点 (步数无效?)"


def test_refattn_stats_per_sample_plumbing():
    """契约 14: K1-A 插桩统计逐样本记录 (真实 SLFUNetV24OneShot, CPU)."""
    import torch
    from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot
    import exp.v26.diagnose_refattn as dr

    torch.manual_seed(0)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).eval()
    B = 3
    stats = []
    fm.ref_attn16.forward = dr.patch_refattn(fm, stats=stats, mode="R0_original")
    z = torch.randn(B, 4, 64, 64)
    out = fm(torch.cat([z, z], 0), torch.randn(2 * B, 1, 16, 16), torch.randn(2 * B, 384, 16, 16),
             torch.zeros(2 * B, dtype=torch.long), torch.randn(2 * B, 128),
             torch.full((2 * B,), 0.45),
             drop_style=torch.cat([torch.zeros(B, dtype=torch.bool), torch.ones(B, dtype=torch.bool)]),
             drop_content=torch.zeros(2 * B, dtype=torch.bool), layout_scale=torch.ones(2 * B),
             d16_ref=torch.randn(2 * B, 384, 16, 16), v16_ref=torch.randn(2 * B, 192, 16, 16),
             u_glob=torch.randn(2 * B, 795))
    assert len(stats) == 1
    for key in ("entropy_ratio", "max_qk", "gate", "local_norm", "global_norm", "out_rel_norm"):
        assert stats[0][key].shape == (2 * B,), key
        assert torch.isfinite(stats[0][key]).all(), key
    assert torch.isfinite(out).all()
    del fm.ref_attn16.forward
