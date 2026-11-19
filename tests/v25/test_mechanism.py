"""V25 B01: 机制 A/B 数学与代码正确性 CPU 单测 (小张量 / mock / 禁真实 ckpt / 禁 GPU).

覆盖 (agent2.md §三 B01 验收 + 任务书逐条):
  - 初始门控 == 全局 γ 混合 (≤1e-6)           [A 末层零初始化恒等热启动]
  - clamp 后均值归一仍 mean=1 / 逐点 ≤ cap    [次序铁律: clamp 在前, mean-norm 在后]
  - edit_energy 定义与单调性 sanity            [相同总体编辑能量对照的度量基础]
  - per-task 能量匹配 γ* 精确恢复目标 E        [A2/A3 能量对照设计]
  - A1 启发式门输出有限值且归一
  - γ=0 逐元素精确回退 z_P1                    [B 显式 γ 残差构造性保证, 不依赖初始化]
  - TeacherSpec 不一致触发 AssertionError      [教师定义固定铁律]
  - 同 seed 确定性
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 必须在 import torch 之前 → 纯 CPU
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import math

import pytest
import torch

from exp.v25.mechanism_a import (
    A_SEED,
    CANONICAL_TEACHER_SPEC,
    GATE_CAP,
    TeacherSpec,
    apply_gate,
    arm_gate_field,
    assert_same_teacher,
    content_gradient_density,
    edit_energy,
    edge_continuity_loss,
    energy_matched_gamma,
    gate_input,
    heuristic_gate,
    load_teacher_cache,
    normalize_gate,
    tv_loss,
    upsample_gate,
    GateNet,
)
from exp.v25.mechanism_b import (
    B_GAMMA_POOL,
    B_SCALES,
    GammaResidualStudent,
    eval_student_gamma_sweep,
    gamma_fourier,
    sample_train_gammas,
)

torch.manual_seed(0)


def _small_inputs(B=3, seed=7):
    g = torch.Generator().manual_seed(seed)
    return dict(
        z_p1=torch.randn(B, 4, 64, 64, generator=g) * 0.6,
        dz=torch.randn(B, 4, 64, 64, generator=g) * 0.4,
        c64=torch.randn(B, 4, 64, 64, generator=g) * 0.5,
        chord=torch.randn(B, 128, generator=g),
    )


# ---------------- 机制 A: 门控数学 ----------------

def test_gatenet_zero_init_equals_global_blend():
    """末层零初始化 → 初始门控下 z_out 与全局 γ 混合逐元素相等 (≤1e-6)."""
    inp = _small_inputs()
    torch.manual_seed(A_SEED)
    net = GateNet().eval()
    M16 = net(inp["c64"], inp["dz"], inp["chord"])
    assert torch.allclose(M16, torch.ones_like(M16), atol=1e-6)
    for gamma in (0.0, 0.4, 0.55):
        z_out = apply_gate(inp["z_p1"], inp["dz"], M16, gamma)
        z_glob = inp["z_p1"] + gamma * inp["dz"]
        assert torch.allclose(z_out, z_glob, atol=1e-6), f"gamma={gamma} 初始门控偏离全局混合"


def test_gate_norm_mean_one_after_clamp():
    """clamp 后均值归一仍 mean=1 (可行域) / cap 触底域 max==cap / 全零角回退恒等门."""
    # 可行域: clamp 不触底 → 精确 mean=1 且逐点 ≤ cap
    raw = torch.tensor([[[[0.5, 1.5], [1.0, 0.5]]]])
    M = normalize_gate(raw)
    assert abs(float(M.mean()) - 1.0) <= 1e-6
    assert float(M.max()) <= GATE_CAP + 1e-6 and float(M.min()) >= 0.0
    # cap 触底域: clamp 截掉负值 → mean-norm 分母取 max(mean, mx/cap) → max 恰为 cap
    raw = torch.tensor([[[[-1.0, 3.0], [-1.0, 2.0]]]])
    M = normalize_gate(raw)
    assert float(M.max()) <= GATE_CAP + 1e-6 and float(M.min()) >= 0.0
    assert float(M.mean()) <= 1.0 + 1e-6
    # 全零角: 不可归一 → 回退恒等门 M≡1 (能量守恒唯一自洽解)
    M = normalize_gate(torch.full((1, 1, 2, 2), -1.0))
    assert torch.allclose(M, torch.ones_like(M))


def test_gate_norm_order_matters():
    """次序铁律单测: clamp 会破坏 mean=1 → mean-norm 必须在后;
    裸 'clamp→÷mean' (无 max/cap 分母下界) 会重新突破逐点上限, 故分母必须取 max(mean, mx/cap)."""
    # (i) 带负值的 raw: 归一化后 mean==1 (clamp 造成的均值亏空被恢复)
    raw = torch.tensor([[[[-1.0, 2.0], [-1.0, 2.0]]]])  # clamp → [0,2,0,2], mean=1
    M = normalize_gate(raw)
    assert abs(float(M.mean()) - 1.0) <= 1e-6
    # 反证: 若先 mean-norm 再 clamp (错误次序) → [−2,4,−2,4]→clamp→[0,2.5,0,2.5], mean=1.25≠1
    wrong = (raw / raw.mean()).clamp(0.0, GATE_CAP)
    assert abs(float(wrong.mean()) - 1.0) > 1e-3
    # (ii) 裸 ÷mean 会突破 cap: mx=2.5, mean=0.7 → 2.5/0.7=3.57 > 2.5
    raw = torch.tensor([[[[2.5, 0.1], [0.1, 0.1]]]])
    naive = (raw.clamp(0.0, GATE_CAP) / raw.clamp(0.0, GATE_CAP).mean())
    assert float(naive.max()) > GATE_CAP  # 错误归一违反上限
    M = normalize_gate(raw)
    assert float(M.max()) <= GATE_CAP + 1e-6  # 实现的 max/cap 分母下界保住上限


def test_gate_latent_grid_mean_one():
    """16×16 → 64×64 上采样后 (双线性不保均值) 二次归一 → latent 网格 mean==1.

    不变量: mean==1 (可行域) 或 max==cap (cap 触底域, 模型想降总能量被逐点上限截住) 二者必居其一.
    """
    torch.manual_seed(3)
    # 可行域: 中等 logits (tanh 不饱和) → 16×16 与 64×64 两级均精确 mean=1
    logits = torch.randn(4, 1, 16, 16) * 0.3
    M16 = normalize_gate(1.0 + 1.5 * torch.tanh(logits))
    assert abs(float(M16.mean()) - 1.0) <= 1e-6
    M64 = upsample_gate(M16, size=64)
    assert M64.shape == (4, 1, 64, 64)
    assert abs(float(M64.mean()) - 1.0) <= 1e-6
    assert float(M64.max()) <= GATE_CAP + 1e-6 and float(M64.min()) >= 0.0
    # 饱和域: 不变量为析取 (mean==1 或 max==cap), 不允许出现 NaN/负值/超限
    sat = normalize_gate(1.0 + 1.5 * torch.tanh(torch.randn(4, 1, 16, 16) * 3.0))
    sat64 = upsample_gate(sat, size=64)
    ok = (abs(float(sat64.mean()) - 1.0) <= 1e-6) or (float(sat64.max()) >= GATE_CAP - 1e-6)
    assert ok and torch.all(torch.isfinite(sat64)) and float(sat64.min()) >= 0.0


def test_edit_energy_definition_and_monotonicity():
    """edit_energy = ||γ·M⊙Δz||₂/||Δz||₂: A0 (M≡1) 恒等于 γ; 固定 M 下随 γ 线性单调增."""
    inp = _small_inputs(B=2)
    M16 = torch.full((2, 1, 16, 16), 1.0)
    for gamma in (0.3, 0.55):
        z_out = apply_gate(inp["z_p1"], inp["dz"], M16, gamma)
        e = edit_energy(z_out, inp["z_p1"], inp["dz"])
        assert torch.allclose(e, torch.full_like(e, gamma), atol=1e-6), "A0 能量应恒等于名义 γ"
    # 固定非平凡 M: E(γ) = γ·m_rms 线性
    torch.manual_seed(1)
    M16 = normalize_gate(1.0 + 1.5 * torch.tanh(torch.randn(2, 1, 16, 16)))
    e1 = edit_energy(apply_gate(inp["z_p1"], inp["dz"], M16, 0.2), inp["z_p1"], inp["dz"])
    e2 = edit_energy(apply_gate(inp["z_p1"], inp["dz"], M16, 0.8), inp["z_p1"], inp["dz"])
    assert torch.all(e2 > e1), "能量必须随 γ 单调增"
    M64 = upsample_gate(M16, size=64)
    m_rms = (M64 * inp["dz"]).flatten(1).norm(dim=1) / inp["dz"].flatten(1).norm(dim=1)
    assert torch.allclose(e2, 0.8 * m_rms, atol=1e-6), "E 应精确等于 γ·||M⊙Δz||/||Δz||"


def test_energy_matched_gamma_restores_target():
    """per-task 能量匹配 γ* = target/m_rms → 应用后逐任务 E 精确等于 target (A2/A3 能量对照)."""
    inp = _small_inputs(B=3)
    torch.manual_seed(5)
    M16 = normalize_gate(1.0 + 1.5 * torch.tanh(torch.randn(3, 1, 16, 16) * 2.0))
    target = 0.5
    g_star = energy_matched_gamma(M16, inp["dz"], target)
    assert g_star.shape == (3,) and torch.all(torch.isfinite(g_star))
    z_out = apply_gate(inp["z_p1"], inp["dz"], M16, g_star)
    e = edit_energy(z_out, inp["z_p1"], inp["dz"])
    assert torch.allclose(e, torch.full_like(e, target), atol=1e-5)


def test_heuristic_gate_finite_and_normalized():
    """A1 启发式门 (κ∈{0.5,1.0}): 输出有限 / [0,cap] / mean=1; 常量图 (ē≡0) 退化为均匀门 1."""
    inp = _small_inputs(B=3)
    for kappa in (0.5, 1.0):
        M = heuristic_gate(inp["c64"], kappa)
        assert torch.all(torch.isfinite(M)), "启发式门必须有限"
        assert float(M.min()) >= 0.0 and float(M.max()) <= GATE_CAP + 1e-6
        assert abs(float(M.mean()) - 1.0) <= 1e-5
    const_c64 = torch.full((3, 4, 64, 64), 0.7)
    M = heuristic_gate(const_c64, 1.0)
    assert torch.allclose(M, torch.ones_like(M), atol=1e-6)


def test_arm_gate_field_dispatch():
    """四臂门控分发: A0 恒 1 / A1 归一有限 / A2,A3 需网络; 未知臂报错."""
    inp = _small_inputs(B=2)
    m0 = arm_gate_field("A0_global", inp["c64"], inp["dz"], inp["chord"])
    assert torch.equal(m0, torch.ones_like(m0))
    m1 = arm_gate_field("A1_heuristic", inp["c64"], inp["dz"], inp["chord"], kappa=0.5)
    assert torch.all(torch.isfinite(m1))
    with pytest.raises(ValueError):
        arm_gate_field("A2_learned", inp["c64"], inp["dz"], inp["chord"], net=None)
    with pytest.raises(ValueError):
        arm_gate_field("A9_bogus", inp["c64"], inp["dz"], inp["chord"])


def test_gate_input_shape():
    """GateNet 输入拼接: c_16(4) ⊕ Δz16(4) ⊕ 风格弦 tile(128) = (B,136,16,16)."""
    inp = _small_inputs(B=2)
    x = gate_input(inp["c64"], inp["dz"], inp["chord"])
    assert x.shape == (2, 136, 16, 16)


def test_content_gradient_density_range():
    """ē ∈ [0,1]; 常量结构图 → ē≡0 (低细节极值); 高频图 min/max 跨度拉开."""
    const = torch.full((2, 4, 64, 64), 0.3)
    e = content_gradient_density(const)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-6)
    torch.manual_seed(2)
    e = content_gradient_density(torch.randn(2, 4, 64, 64))
    assert float(e.min()) >= 0.0 and float(e.max()) <= 1.0


def test_tv_and_edge_losses():
    """A3 正则: 常量门 TV=0 / 边缘惩罚=0; 变化门 > 0 且有限."""
    inp = _small_inputs(B=2)
    ones = torch.ones(2, 1, 16, 16)
    assert float(tv_loss(ones)) == 0.0
    assert float(edge_continuity_loss(ones, inp["c64"])) == 0.0
    M = normalize_gate(1.0 + 1.5 * torch.tanh(torch.randn(2, 1, 16, 16)))
    assert float(tv_loss(M)) > 0.0
    edge = edge_continuity_loss(M, inp["c64"])
    assert math.isfinite(float(edge)) and float(edge) >= 0.0


# ---------------- TeacherSpec: 教师定义固定铁律 ----------------

def test_teacher_spec_roundtrip():
    d = CANONICAL_TEACHER_SPEC.to_dict()
    spec = TeacherSpec.from_dict(d)
    assert_same_teacher(spec, CANONICAL_TEACHER_SPEC)
    assert d == spec.to_dict(), "to_dict 必须幂等 (posix 归一)"
    assert d["branches"] == "cs_j" and d["gate"] == "original" and d["p1_lambda"] == 1.0


def test_teacher_spec_mismatch_asserts():
    """不一致教师定义必须触发 AssertionError (逐键报告) — 训练/评测共用同一 spec 的机器保障."""
    d = CANONICAL_TEACHER_SPEC.to_dict()
    for key, bad in (("branches", "full"), ("gate", "fixed"), ("anchor", "z_src"),
                     ("d6_ckpt", "runs/other/ckpt.pt"), ("p1_lambda", 1.3)):
        tampered = dict(d)
        tampered[key] = bad
        with pytest.raises(AssertionError, match=key):
            assert_same_teacher(tampered, CANONICAL_TEACHER_SPEC, context="unit")
    tampered = dict(d)
    tampered["solver"] = dict(d["solver"], tau_min=0.40)
    with pytest.raises(AssertionError, match="solver"):
        assert_same_teacher(tampered, CANONICAL_TEACHER_SPEC)


def test_teacher_spec_schema_errors():
    d = CANONICAL_TEACHER_SPEC.to_dict()
    missing = {k: v for k, v in d.items() if k != "solver"}
    with pytest.raises(ValueError, match="missing"):
        TeacherSpec.from_dict(missing)
    extra = dict(d, unknown_field=1)
    with pytest.raises(ValueError, match="unexpected"):
        TeacherSpec.from_dict(extra)


def test_teacher_spec_separator_normalization():
    """Windows 缓存的 'runs\\v20\\...' 与 Linux 'runs/v20/...' 必须判等 (跨平台队列)."""
    d = CANONICAL_TEACHER_SPEC.to_dict()
    win = dict(d)
    win["d6_ckpt"] = d["d6_ckpt"].replace("/", "\\")
    win["p1_ckpt"] = d["p1_ckpt"].replace("/", "\\")
    assert_same_teacher(win, CANONICAL_TEACHER_SPEC)  # 不应触发


def test_teacher_cache_contract(tmp_path):
    """缓存消费契约: teacher_spec 与权威 spec 一致才放行; 篡改即 AssertionError."""
    payload = dict(
        z0_tr=torch.randn(4, 4, 16, 16).half(), z_anc_tr=torch.randn(4, 4, 16, 16).half(),
        z_p1_tr=torch.randn(4, 4, 16, 16).half(), z_d6_tr=torch.randn(4, 4, 16, 16).half(),
        sid_tr=torch.zeros(4, dtype=torch.long), tid_tr=torch.ones(4, dtype=torch.long),
        teacher_spec=CANONICAL_TEACHER_SPEC.to_dict(),
    )
    p = tmp_path / "train_latents_mock.pt"
    torch.save(payload, p)
    loaded = load_teacher_cache(p)
    assert set(("z_p1_tr", "z_d6_tr", "teacher_spec")) <= set(loaded)
    bad = dict(payload)
    bad["teacher_spec"] = dict(payload["teacher_spec"], gate="fixed")
    p2 = tmp_path / "train_latents_bad.pt"
    torch.save(bad, p2)
    with pytest.raises(AssertionError, match="gate"):
        load_teacher_cache(p2)


# ---------------- 机制 B: γ 残差学生 ----------------

def _make_student(width=32, blocks=2, dilations=(1, 2), randomize_out=True, seed=11):
    torch.manual_seed(seed)
    stu = GammaResidualStudent(width=width, blocks=blocks, dilations=dilations)
    if randomize_out:  # 打破零初始化: 证明 γ=0 回退不依赖任何初始化技巧
        torch.manual_seed(seed + 100)
        for m in stu.modules():
            if isinstance(m, torch.nn.Conv2d) and m.weight.requires_grad:
                torch.nn.init.normal_(m.weight, std=0.05)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, std=0.05)
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.05)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, std=0.05)
    return stu.eval()


def _student_inputs(B=3, seed=9):
    g = torch.Generator().manual_seed(seed)
    return dict(
        z_p1=torch.randn(B, 4, 64, 64, generator=g) * 0.6,
        c64=torch.randn(B, 4, 64, 64, generator=g) * 0.5,
        d16=torch.randn(B, 384, 16, 16, generator=g),
        chord=torch.randn(B, 128, generator=g),
    )


def test_gamma_zero_exact_fallback():
    """γ=0 时输出与 z_P1 逐元素严格相等 (随机权重, 显式 γ 残差构造性保证; 不依赖零初始化)."""
    inp = _student_inputs()
    stu = _make_student()
    z_out = stu(inp["z_p1"], inp["c64"], inp["d16"], inp["chord"], 0.0)
    assert torch.equal(z_out, inp["z_p1"]), "γ=0 必须逐元素精确回退 (torch.equal)"
    # per-sample γ 向量: γ=0 的样本精确回退, γ>0 的样本偏离
    gv = torch.tensor([0.0, 0.5, 0.0])
    z_mix = stu(inp["z_p1"], inp["c64"], inp["d16"], inp["chord"], gv)
    assert torch.equal(z_mix[0], inp["z_p1"][0]) and torch.equal(z_mix[2], inp["z_p1"][2])
    assert not torch.equal(z_mix[1], inp["z_p1"][1])
    assert torch.all(torch.isfinite(z_mix))


def test_student_init_identity_any_gamma():
    """零初始化附加性质: 初始任意 γ 下 R_θ≡0 → z_student ≡ z_P1 (训练从 PRISM-1 热启动)."""
    inp = _student_inputs()
    stu = _make_student(randomize_out=False)
    for gamma in (0.25, 0.55, 0.70):
        z_out = stu(inp["z_p1"], inp["c64"], inp["d16"], inp["chord"], gamma)
        assert torch.allclose(z_out, inp["z_p1"], atol=1e-6), f"γ={gamma}"


def test_residual_scales_with_gamma():
    """残差输出随 γ 连续变化 (学生覆盖 (1−γ)z_p1+γ·z_d6 混合曲线的机制前提)."""
    inp = _student_inputs()
    stu = _make_student()
    with torch.no_grad():
        zs = [stu(inp["z_p1"], inp["c64"], inp["d16"], inp["chord"], g)
              for g in (0.25, 0.40, 0.55, 0.70)]
        d01 = float((zs[0] - zs[1]).abs().max())
    assert d01 > 0.0
    assert all(torch.all(torch.isfinite(z)) for z in zs)


def test_param_budgets():
    """B01 预算: GateNet ≤0.5M; B0 残差 ≤1M; B1 宽化 ~3–5M."""
    assert GateNet().n_params() <= 500_000
    b0 = GammaResidualStudent(**{k: v for k, v in B_SCALES["B0"].items() if k != "updates"})
    b1 = GammaResidualStudent(**{k: v for k, v in B_SCALES["B1"].items() if k != "updates"})
    assert b0.n_params() <= 1_000_000, f"B0={b0.n_params()}"
    assert 3_000_000 <= b1.n_params() <= 5_000_000, f"B1={b1.n_params()}"


def test_gamma_fourier_properties():
    """γ Fourier 32 维: 形状 / γ=0 → sin≡0,cos≡1 / 频率倍增 / 确定性."""
    g = torch.tensor([0.0, 0.25, 0.55])
    f = gamma_fourier(g)
    assert f.shape == (3, 32)
    assert torch.allclose(f[0, :16], torch.zeros(16), atol=1e-7)
    assert torch.allclose(f[0, 16:], torch.ones(16), atol=1e-7)
    assert abs(float(f[1, 0]) - math.sin(math.pi * 0.25)) <= 1e-6
    assert abs(float(f[1, 1]) - math.sin(2 * math.pi * 0.25)) <= 1e-6  # k=1 频率翻倍
    assert torch.equal(gamma_fourier(g), gamma_fourier(g))
    f_scalar = gamma_fourier(0.5)
    assert f_scalar.shape == (1, 32) and abs(float(f_scalar[0, 0]) - 1.0) <= 1e-6


def test_sample_train_gammas_weighting():
    """γ 池 {0,0,0.25,0.40,0.55,0.70}: γ=0 权重 ×2 (P≈1/3); 同 generator 确定性."""
    g1 = torch.Generator().manual_seed(42)
    a = sample_train_gammas(12000, g1)
    g2 = torch.Generator().manual_seed(42)
    b = sample_train_gammas(12000, g2)
    assert torch.equal(a, b), "同 seed 采样必须确定"
    pool = torch.tensor(B_GAMMA_POOL)
    assert bool(torch.isin(a, pool).all())
    freq0 = float((a == 0.0).float().mean())
    assert 0.28 < freq0 < 0.39, f"P(γ=0) 应 ≈2/6, 实测 {freq0:.3f}"


def test_same_seed_determinism():
    """同 seed → GateNet / 学生 初始化与前向逐位一致 (r2 工程规范)."""
    from exp.v25.mechanism_a import seed_all
    gate_inp = _small_inputs()
    seed_all(1234)
    net_a = GateNet().eval()
    out_a = net_a(gate_inp["c64"], gate_inp["dz"], gate_inp["chord"])
    seed_all(1234)
    net_b = GateNet().eval()
    out_b = net_b(gate_inp["c64"], gate_inp["dz"], gate_inp["chord"])
    assert torch.equal(out_a, out_b)
    inp = _student_inputs()
    seed_all(2026)
    stu_a = _make_student(seed=2026)
    za = stu_a(inp["z_p1"], inp["c64"], inp["d16"], inp["chord"], 0.55)
    seed_all(2026)
    stu_b = _make_student(seed=2026)
    zb = stu_b(inp["z_p1"], inp["c64"], inp["d16"], inp["chord"], 0.55)
    assert torch.equal(za, zb)


# ---------------- γ 扫描评测 (mock student_fn, 无 GPU / 无真实 ckpt) ----------------

def _sweep_fixture(B=6, seed=13):
    g = torch.Generator().manual_seed(seed)
    z_src = torch.randn(B, 4, 16, 16, generator=g) * 0.6
    z_p1 = z_src + torch.randn(B, 4, 16, 16, generator=g) * 0.3
    z_d6 = z_p1 + torch.randn(B, 4, 16, 16, generator=g) * 0.5
    same = torch.tensor([False] * 5 + [True])
    tid = torch.tensor([0, 1, 2, 3, 4, 0])
    return z_src, z_p1, z_d6, same, tid


def test_eval_student_gamma_sweep_identity_mock():
    """恒等学生 mock: 逐任务残差与教师曲线手工值一致; 未见 γ 插值 gap≈0 (教师对 γ 线性)."""
    z_src, z_p1, z_d6, same, tid = _sweep_fixture()
    sweep = eval_student_gamma_sweep(
        lambda zp, gv, cb: zp, z_src, z_p1, z_d6, same, tid, cond=None, device="cpu")
    recs = sweep["records"]
    assert len(recs) == 7
    assert [r["gamma"] for r in recs] == [0.0, 0.25, 0.33, 0.40, 0.55, 0.63, 0.70]
    assert {r["gamma"] for r in recs if r["unseen"]} == {0.33, 0.63}, "0.33/0.63 须标记为未见泛化点"
    by_g = {r["gamma"]: r for r in recs}
    # γ=0: 教师 z*=z_p1, 恒等学生 → 残差 ≡ 0
    assert by_g[0.0]["residual_mean"] <= 1e-6
    # γ=0.25 手工核验逐任务残差 (cross 任务)
    g = 0.25
    teacher = (1 - g) * z_p1 + g * z_d6
    cross = ~same
    expected = ((z_p1 - teacher).flatten(1).norm(dim=1)
                / teacher.flatten(1).norm(dim=1).clamp_min(1e-8))[cross]
    got = torch.tensor(by_g[g]["residual_per_task"])
    assert torch.allclose(got, expected, atol=1e-5), "逐任务残差必须与教师曲线手工值一致"
    # per-style 分解: 5 画派键齐全
    assert set(by_g[g]["residual_per_style"]) == {"Ear", "Imp", "Min", "Roc", "Uki"}
    # 恒等学生精确线性 → 未见 γ 插值 gap = 0
    for gu in (0.33, 0.63):
        assert by_g[gu]["interp_neighbors"] is not None
        assert abs(by_g[gu]["interp_gap"]) <= 1e-6, "教师曲线线性 → 完美学生插值 gap=0"
    assert set(sweep["z_by_gamma"]) == {float(x) for x in (0.0, 0.25, 0.33, 0.40, 0.55, 0.63, 0.70)}


def test_eval_student_gamma_sweep_gamma_residual_mock():
    """γ 残差 mock: 学生 z = z_p1 + γ·δ → 残差随 γ 变化且有限; same 任务 bypass 后残差不含 same."""
    z_src, z_p1, z_d6, same, tid = _sweep_fixture()
    delta = torch.randn_like(z_p1) * 0.4

    def fn(zp, gv, cb):
        return zp + gv.view(-1, 1, 1, 1) * delta

    sweep = eval_student_gamma_sweep(fn, z_src, z_p1, z_d6, same, tid, cond=None, device="cpu")
    recs = {r["gamma"]: r for r in sweep["records"]}
    vals = [recs[g]["residual_mean"] for g in (0.25, 0.40, 0.55, 0.70)]
    assert all(math.isfinite(v) for v in vals) and len(set(vals)) == 4
    assert all(len(r["residual_per_task"]) == int((~same).sum()) for r in sweep["records"])


def test_no_cuda_in_tests():
    """护栏: 本测试文件必须保持纯 CPU (CUDA_VISIBLE_DEVICES='' 生效)."""
    assert not torch.cuda.is_available() or torch.cuda.device_count() == 0
