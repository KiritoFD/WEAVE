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
