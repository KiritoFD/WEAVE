"""B05 CPU 测试: 统一 Minimal PRISM 配置规约 + 双接口骨架（agent2.md §五 B05）。

覆盖（全部 CPU / 无 CUDA / 不加载任何真实 checkpoint / 无绝对路径）：
1. 默认配置 hash 确定性（含冻结 hash pin —— 有意变更冻结默认值时须同步更新 pin）；
2. strength→旋钮映射消费 a04 JSON（含 UNMEASURED clamp 行为 + 文件缺失恒等退化）；
3. from_descriptor 重建历史臂配置（B 曲线 τ-shift / Hybrid γ / MechA A0/A1）且 param_hash 一致；
4. student engine raise NotImplementedError("mechanism B pending")；
5. Ref/Core 接口签名 inspect 断言（仅 content/style|reference/strength）；
6. 同 config 不同 strength 确定性一致（seed 固定路径，fake runtime 注入）；
7. lazy import：import src.model.prism_v25 不拉起 exp 重模块，device_count()==0 不崩。
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.model.prism_v25 import PRISMCore, PRISMRef, STUDENT_PENDING_MESSAGE
from src.v25.artifact_registry import param_hash
from src.v25.config import (
    DEFAULT_CORE_ENGINE,
    DEFAULT_REF_ENGINE,
    DEFAULT_STRENGTH,
    DESCRIPTOR_FORMAT_VERSION,
    ENGINE_REGISTRY,
    HISTORICAL_ARM_DESCRIPTORS,
    HISTORICAL_ARM_NAMES,
    RefConfig,
    TransferConfig,
    _norm_separators,
    a04_audit_path,
    clear_strength_mapping_cache,
    config_hash,
    describe,
    execution_seed,
    get_engine,
    load_strength_mapping,
    restore_historical_arm,
    strength_to_knob,
    with_override,
)

# --------------------------------------------------------------------------- #
# 冻结 hash pin（冻结证据）：有意更改冻结默认值（总控切换点）时必须同步更新本表，
# 并在交付说明中记录变更 —— pin 漂移 = 默认值漂移的报警器。
# --------------------------------------------------------------------------- #
PINNED_HASHES = {
    "default_core": "33e5d95f938e044fe0f99f25be85b808b9eeed8a15044e8f42947a99192231d6",
    "default_ref": "11b840bd93f08e58d9463a48e4265b2df7108b368acdd7fb89ba6097b0ebd7f5",
    "B_d6_t050": "f51e6af34d94e9d1094334522cd1e02d4c2e29c498cfbb952a884a81deb54ec2",
    "B_d6_t072": "2bee7c3128012975b2572f447f4a07f1d6de18ddd00b1b757a7e034cd23146f3",
    "C_hyb_g70": "30957e06259f4205aeeca718d59ae2010f1e07c9646fdc9fe6852b81145ad04e",
    "C_hyb_g85": "945fbbc4e4a0f79643276ab83128d1feae60649e8cb201fab787bbbb7e552fd9",
    "MechA_A0_g55": "4956137a0ee5c16263541d2c0b97f83db175cab2bb7e8abf22d015a042ccbc68",
    "MechA_A1_k50_g55": "d5dbaafc58835f8dba28d535bdd23e67e9a8fd9f68f1ba9a75f86a1a80672cee",
}

#: (a04 family, strength 消费配置) 对齐表（core 四族 + ref 一族）
_FAMILY_CONFIGS = [
    ("hybrid_gamma", TransferConfig()),
    ("prism1_lambda", TransferConfig.for_engine("prism1")),
    ("d6_tau_min", TransferConfig.for_engine("d6_flow")),
    ("o3_tau0", RefConfig()),
]


class _RecordingRuntime:
    """fake runtime：记录 run() 调用并返回确定性 token（CPU 注入，验证 seed/knob 路径）。"""

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "engine": kwargs["engine"], "interface": kwargs["interface"],
            "knob": kwargs["knob"], "knob_value": kwargs["knob_value"],
            "seed": kwargs["seed"], "strength": kwargs["strength"],
            # 确定性 token：同 (seed, knob_value) 逐位一致
            "token": f"{kwargs['seed']}:{kwargs['knob_value']!r}",
            "z": None, "rgb01": None,
        }


# --------------------------------------------------------------------------- #
# 1) 默认配置 hash 确定性 + 冻结 pin
# --------------------------------------------------------------------------- #
def test_default_core_config_hash_deterministic_and_pinned():
    h1 = config_hash(TransferConfig())
    h2 = config_hash(TransferConfig())  # 重复构造 → 同 hash
    h3 = config_hash(TransferConfig.for_engine(DEFAULT_CORE_ENGINE))
    assert h1 == h2 == h3 == PINNED_HASHES["default_core"]
    # 单一 hash 约定：config_hash == param_hash(descriptor)（artifact_registry.param_hash）
    assert h1 == param_hash(TransferConfig().to_descriptor())
    # solver dict 键序无关（sort_keys 序列化约定）
    cfg = TransferConfig()
    desc_shuffled = cfg.to_descriptor()
    desc_shuffled["solver"] = dict(reversed(list(desc_shuffled["solver"].items())))
    assert config_hash(TransferConfig.from_descriptor(desc_shuffled)) == h1
    # 任何冻结字段变化 → hash 必变（可检测性）
    changed = with_override(TransferConfig(), {"gate_fixed_g": 0.4})
    assert config_hash(changed) != h1


def test_default_ref_config_hash_deterministic_and_pinned():
    h1 = config_hash(RefConfig())
    assert h1 == config_hash(RefConfig()) == config_hash(RefConfig.for_engine("o3_ref"))
    assert h1 == PINNED_HASHES["default_ref"] == param_hash(RefConfig().to_descriptor())
    assert config_hash(with_override(RefConfig(), {"wj": 2.0})) != h1


def test_bare_core_config_equals_for_engine_default():
    """裸构造 = 冻结默认 = for_engine(DEFAULT_CORE_ENGINE)（默认值集中一处）。"""
    assert TransferConfig() == TransferConfig.for_engine(DEFAULT_CORE_ENGINE)
    assert DEFAULT_CORE_ENGINE == "hybrid"
    assert DEFAULT_REF_ENGINE == "o3_ref"
    assert DEFAULT_STRENGTH == 0.5


# --------------------------------------------------------------------------- #
# 2) strength→旋钮映射消费 a04 JSON（UNMEASURED clamp / 恒等退化）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family,cfg", _FAMILY_CONFIGS, ids=[f for f, _ in _FAMILY_CONFIGS])
def test_strength_mapping_consumes_a04_json(family, cfg):
    with open(a04_audit_path(), "r", encoding="utf-8") as fh:
        audit = json.load(fh)
    fam = audit["families"][family]
    sm = fam["strength_mapping"]
    m = load_strength_mapping(family)
    assert m is not None
    assert m["knob"] == fam["knob"] and m["direction"] == fam["direction"]
    knots = m["knots"]
    assert [k["strength"] for k in knots] == sorted(k["strength"] for k in knots)
    assert knots[0]["strength"] == 0.0 and knots[-1]["strength"] == 1.0
    assert len({k["arm"] for k in knots}) == len(knots)

    # a04 预算的 strength=0.5 旋钮值（分段线性插值逐位复现，容差为 JSON 5 位小数舍入）
    r_half = strength_to_knob(cfg, 0.5)
    expect = sm["strength_at_0.5"]["knob_value"]
    assert abs(r_half.value - expect) < 1e-4
    assert r_half.source == f"a04:{family}"
    assert r_half.clamped is False
    assert r_half.knob == fam["knob"]

    # UNMEASURED clamp：s<0 → 最弱实测端点；s>1 → 最强实测端点（clamped=True 如实标注）
    r_lo = strength_to_knob(cfg, -0.2)
    r_hi = strength_to_knob(cfg, 1.2)
    assert r_lo.value == knots[0]["knob_value"] and r_lo.clamped is True
    assert r_hi.value == knots[-1]["knob_value"] and r_hi.clamped is True

    # 方向语义：direct = 旋钮随 s 增大而增大；inverse = 减小（与 a04 direction 一致）
    s0, s9 = strength_to_knob(cfg, 0.05).value, strength_to_knob(cfg, 0.95).value
    if fam["direction"] == "direct":
        assert s9 > s0
    else:
        assert s9 < s0


def test_strength_mapping_missing_file_falls_back_to_identity(tmp_path, monkeypatch):
    import src.v25.config as cfgmod
    missing = tmp_path / "no_such_a04.json"
    with pytest.warns(RuntimeWarning, match="strength"):
        m = load_strength_mapping("hybrid_gamma", path=missing, use_cache=False)
    assert m is None  # 退化标记：load 返回 None（不抛错）
    # 恒等映射行为（缺文件默认路径 → WARNING + knob = clamp(s, 0, 1)，来源如实标注）
    monkeypatch.setattr(cfgmod, "a04_audit_path", lambda: missing)
    clear_strength_mapping_cache()
    try:
        with pytest.warns(RuntimeWarning, match="strength"):
            r_mid = strength_to_knob(TransferConfig(), 0.5)
        assert r_mid.value == 0.5 and r_mid.source == "identity_fallback"
        assert r_mid.clamped is False
        r_over = strength_to_knob(TransferConfig(), 1.7)
        assert r_over.value == 1.0 and r_over.clamped is True
        r_under = strength_to_knob(TransferConfig(), -0.3)
        assert r_under.value == 0.0 and r_under.clamped is True
    finally:
        clear_strength_mapping_cache()  # 缺文件缓存键清理，避免污染其它用例


def test_strength_mapping_missing_family_raises_clear_error(tmp_path):
    import shutil
    real = a04_audit_path()
    if not real.is_file():
        pytest.skip("a04 audit 文件不在本 checkout")
    copy = tmp_path / "a04_copy.json"
    shutil.copyfile(real, copy)
    with pytest.raises(ValueError, match="family"):
        load_strength_mapping("no_such_family", path=copy, use_cache=False)


def test_student_engine_strength_knob_identity_family_none():
    """family=None（无实测曲线）引擎的旋钮解析 = 恒等退化（transfer 层另有 pending 闸门）。"""
    cfg = TransferConfig.for_engine("student")
    r = strength_to_knob(cfg, 0.7)
    assert r.value == 0.7 and r.knob is None and r.source == "identity_fallback"


# --------------------------------------------------------------------------- #
# 3) engine 注册表契约（与 a04 交叉核对，防漂移）
# --------------------------------------------------------------------------- #
def test_engine_registry_contract_against_a04():
    with open(a04_audit_path(), "r", encoding="utf-8") as fh:
        audit = json.load(fh)
    families = audit["families"]
    # 默认冻结点
    spec = ENGINE_REGISTRY[DEFAULT_CORE_ENGINE]
    assert spec.status == "available" and spec.knob == "gamma"
    # student：pending + 无实测族
    student = ENGINE_REGISTRY["student"]
    assert student.status == "pending" and student.family is None and student.knob is None
    assert student.pending_reason == "mechanism B pending"
    # 每个 engine 的 family/knob/direction/interface 与 a04 逐键一致（mecha_a 复用 hybrid_gamma 族）
    for name, es in ENGINE_REGISTRY.items():
        if es.family is None:
            continue
        fam = families[es.family]
        assert fam["knob"] == es.knob, name
        assert fam["direction"] == es.direction, name
        expect_iface = "PRISMCore" if es.interface == "core" else "PRISMRef"
        assert expect_iface in fam["interface"], name
        # knob_default 取自 a04 实测臂的引擎逐一核对（d6_flow 的 0.48 为教师口径 D6H 值，
        # 非曲线 B 实测臂；mecha_a 的 0.55 为机制 A 工作点 γ，两者单独钉在臂恢复测试中）
        if name in {"prism1", "hybrid", "o3_ref"}:
            knot_knobs = {k["knob_value"] for k in fam["strength_mapping"]["knots_measured"]}
            assert es.knob_default in knot_knobs, name


def test_engine_registry_interfaces_consistent_with_config_classes():
    for name, es in ENGINE_REGISTRY.items():
        if es.interface == "core" and es.status == "available":
            cfg = TransferConfig.for_engine(name)
            assert cfg.engine == name
        elif es.interface == "ref":
            cfg = RefConfig.for_engine(name)
            assert cfg.engine == name


# --------------------------------------------------------------------------- #
# 4) student engine raise（pending 路径）
# --------------------------------------------------------------------------- #
def test_student_engine_raises_not_implemented():
    core = PRISMCore(config=TransferConfig.for_engine("student"))
    with pytest.raises(NotImplementedError, match="mechanism B pending"):
        core.transfer(None, None, strength=0.5)
    # 经 override 切到 student 同样触发 pending 闸门（总控机制 B0 完成前的保护）
    core2 = PRISMCore(override={"engine": "student"})
    assert core2.engine == "student"
    with pytest.raises(NotImplementedError, match=STUDENT_PENDING_MESSAGE):
        core2.transfer(None, None)
    # pending 引擎即使注入 runtime 也在接口层被拦截（不落到 runtime）
    core3 = PRISMCore(config=TransferConfig.for_engine("student"), runtime=_RecordingRuntime())
    with pytest.raises(NotImplementedError):
        core3.transfer(None, None)


def test_available_engine_without_runtime_raises_clear_runtime_error():
    """available 引擎 + 未 prepare：清晰 RuntimeError（GPU 资源路径），而非静默成功/假结果。"""
    core = PRISMCore()  # 冻结默认 hybrid
    with pytest.raises(RuntimeError, match="runtime"):
        core.transfer(None, None, strength=0.5)
    ref = PRISMRef()  # 冻结默认 o3_ref
    with pytest.raises(RuntimeError, match="runtime"):
        ref.transfer(None, None, strength=0.5)


# --------------------------------------------------------------------------- #
# 5) 接口签名契约（inspect 断言：仅 content/style|reference/strength）
# --------------------------------------------------------------------------- #
def test_transfer_signature_contract():
    core_sig = inspect.signature(PRISMCore.transfer)
    params = [p for p in core_sig.parameters.values()
              if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    assert [p.name for p in params] == ["self", "content", "style", "strength"]
    assert params[3].default == 0.5 == DEFAULT_STRENGTH
    assert all(p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in core_sig.parameters.values())

    ref_sig = inspect.signature(PRISMRef.transfer)
    rparams = [p for p in ref_sig.parameters.values()
               if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    assert [p.name for p in rparams] == ["self", "content", "reference", "strength"]
    assert rparams[3].default == 0.5
    assert all(p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in ref_sig.parameters.values())


# --------------------------------------------------------------------------- #
# 6) 同 config 不同 strength 确定性一致（seed 固定路径）
# --------------------------------------------------------------------------- #
def test_determinism_same_config_across_strengths():
    runtime = _RecordingRuntime()
    core = PRISMCore(runtime=runtime)
    cfg = TransferConfig()
    seed = execution_seed(cfg)
    assert seed == 20261185  # 冻结默认 = EVAL_SEED（Round1/3 同 eps 跨臂方法学）

    r_lo = core.transfer(None, None, strength=0.2)
    r_hi = core.transfer(None, None, strength=0.8)
    # seed 与 strength 无关（共享噪声路径），两次调用同一 config → 同 seed
    assert r_lo["seed"] == r_hi["seed"] == seed
    assert len({c["seed"] for c in runtime.calls}) == 1
    # 旋钮值随 strength 变化且由 a04 映射决定（direct 族单调）
    assert r_lo["knob_value"] < r_hi["knob_value"]
    assert r_lo["knob"] == r_hi["knob"] == "gamma"
    assert r_lo["strength"] == 0.2 and r_hi["strength"] == 0.8
    # 同 strength 重复调用 → 逐位一致（确定性）
    r_lo_again = core.transfer(None, None, strength=0.2)
    assert r_lo_again == r_lo
    # strength 不进配置 hash（配置与强度解耦）
    assert core.config_hash == config_hash(cfg)


def test_execution_seed_deterministic_and_config_bound():
    cfg = TransferConfig()
    assert execution_seed(cfg) == execution_seed(cfg)
    # seed=None → 由 descriptor hash 派生（仍确定性、仍 strength 无关）
    cfg_free = with_override(cfg, {"seed": None})
    assert execution_seed(cfg_free) == execution_seed(cfg_free)
    assert execution_seed(cfg_free) != execution_seed(cfg)
    # 不同冻结配置 → 不同派生种子
    assert execution_seed(with_override(cfg, {"gate_fixed_g": 0.4})) != execution_seed(cfg_free)


def test_lazy_import_no_exp_modules_and_device_count_zero():
    """子进程断言（B02 lazy-import 手法）：``import src.model.prism_v25`` 不拉起任何
    exp 重模块；CUDA_VISIBLE_DEVICES=\"\" 环境下 torch.cuda.device_count()==0 不崩。"""
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import sys; import src.model.prism_v25; "
        "bad = [m for m in ('exp.v22.core', 'exp.v25.r1_inference_ablation', "
        "'exp.v25.mechanism_a', 'exp.v24.p0_o1_o3_value', 'exp.v19.fiber_duality', "
        "'exp.v20.submode_patchswd') if m in sys.modules]; "
        "assert not bad, f'lazy import violated: {bad}'; "
        "import torch; n = torch.cuda.device_count(); "
        "assert n == 0, f'device_count={n} in CUDA-hidden env'; "
        "print('LAZY_OK device_count=0')"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(repo), env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"stdout={proc.stdout[-1000:]} stderr={proc.stderr[-2000:]}"
    assert "LAZY_OK device_count=0" in proc.stdout


# --------------------------------------------------------------------------- #
# 7) from_descriptor：历史臂精确恢复 + param_hash 一致
# --------------------------------------------------------------------------- #
def test_from_descriptor_restores_all_historical_arms_with_pinned_hash():
    assert set(HISTORICAL_ARM_NAMES) == set(PINNED_HASHES) - {"default_core", "default_ref"}
    for name in HISTORICAL_ARM_NAMES:
        desc = HISTORICAL_ARM_DESCRIPTORS[name]
        cfg = restore_historical_arm(name)
        assert cfg.to_descriptor() == desc, name                      # descriptor 往返逐键一致
        assert config_hash(cfg) == param_hash(desc), name             # 单一 hash 约定
        assert config_hash(cfg) == PINNED_HASHES[name], name          # 冻结 pin
        assert cfg.knob_resolution == "fixed"                         # 历史臂 = 固定旋钮值执行


def test_restored_b_curve_tau_shift_arm_matches_round1_definition():
    from exp.v22.core import D6_SOLVER
    cfg = restore_historical_arm("B_d6_t050")
    assert cfg.engine == "d6_flow"
    # r1 曲线 B: dict(D6_SOLVER, tau_min=0.50, tau_hi=0.78) + branches="full" + original Gate + T4 锚
    assert dict(cfg.solver) == {**D6_SOLVER, "tau_min": 0.50, "tau_hi": 0.78}
    assert cfg.branches == "full" and cfg.gate == "original" and cfg.anchor == "t4"
    assert cfg.steps == D6_SOLVER["steps"] == 8
    r = strength_to_knob(cfg, 0.3)
    assert r.value == 0.50 and r.source == "fixed" and r.knob == "tau_min"
    # strength 映射仍可用（fixed 模式下不消费，但 describe/审计可查）
    mapped = strength_to_knob(with_override(cfg, {"knob_resolution": "strength_map"}), 0.5)
    assert abs(mapped.value - 0.52774) < 1e-4


def test_restored_hybrid_gamma_arm_matches_round1_definition():
    from exp.v22.core import D6_SOLVER
    cfg = restore_historical_arm("C_hyb_g70")
    assert cfg.engine == "hybrid"
    assert cfg.knob_default == 0.70 and cfg.p1_lambda == 1.0
    # d6 分支 = V24 基座缓存口径（D6_SOLVER τ∈[0.40,0.74] + full + original + T4）
    assert dict(cfg.solver) == dict(D6_SOLVER)
    assert cfg.branches == "full" and cfg.gate == "original" and cfg.anchor == "t4"
    assert strength_to_knob(cfg, 1.0).value == 0.70


def test_restored_mecha_arms_match_canonical_teacher_spec():
    from exp.v25.mechanism_a import CANONICAL_TEACHER_SPEC, as_teacher_spec
    from exp.v25.precompute_train_latents import TEACHER_SPEC
    a0 = restore_historical_arm("MechA_A0_g55")
    a1 = restore_historical_arm("MechA_A1_k50_g55")
    assert a0.engine == a1.engine == "mecha_a"
    assert a0.mecha_arm == "A0_global" and a0.mecha_kappa is None
    assert a1.mecha_arm == "A1_heuristic" and a1.mecha_kappa == 0.5
    # 教师三元组铁律：frozen 副本 == precompute TEACHER_SPEC（分隔符归一后）==
    # mechanism_a CANONICAL_TEACHER_SPEC（assert_same_teacher 同一比较语义）
    assert a0.teacher_spec == a1.teacher_spec
    assert as_teacher_spec(a0.teacher_spec).to_dict() == CANONICAL_TEACHER_SPEC.to_dict()
    assert _norm_separators(dict(TEACHER_SPEC)) == a0.teacher_spec
    assert a0.knob_default == 0.55  # TRAIN_GAMMA_DEFAULT（Round1 曲线 C 工作点）


def test_from_descriptor_strictness():
    desc = HISTORICAL_ARM_DESCRIPTORS["C_hyb_g70"]
    bad_extra = dict(desc, rogue_key=1)
    with pytest.raises(ValueError, match="unexpected"):
        TransferConfig.from_descriptor(bad_extra)
    bad_missing = {k: v for k, v in desc.items() if k != "gate"}
    with pytest.raises(ValueError, match="missing"):
        TransferConfig.from_descriptor(bad_missing)
    with pytest.raises(ValueError, match="kind"):
        TransferConfig.from_descriptor(dict(desc, kind="RefConfig"))
    with pytest.raises(ValueError, match="format_version"):
        TransferConfig.from_descriptor(dict(desc, format_version=DESCRIPTOR_FORMAT_VERSION + 1))
    with pytest.raises(ValueError, match="unknown historical arm"):
        restore_historical_arm("no_such_arm")


# --------------------------------------------------------------------------- #
# 8) describe / override / RefConfig 契约
# --------------------------------------------------------------------------- #
def test_describe_recovers_full_params_via_from_descriptor():
    for cfg in (TransferConfig(), TransferConfig.for_engine("d6_flow"), RefConfig()):
        d = describe(cfg)
        assert d["config_hash"] == config_hash(cfg)
        assert d["format_version"] == DESCRIPTOR_FORMAT_VERSION
        assert d["engine"]["name"] == cfg.engine
        assert d["engine"]["interface"] == get_engine(cfg.engine).interface
        assert d["engine"]["status"] == get_engine(cfg.engine).status
        # hash + descriptor → 完整历史参数恢复
        rebuilt = (RefConfig if cfg.engine == "o3_ref" else TransferConfig).from_descriptor(
            d["descriptor"])
        assert rebuilt == cfg
        assert config_hash(rebuilt) == d["config_hash"]
    # a04 引擎的映射摘要（含 clamp 策略与文件血缘 sha12）
    d = describe(TransferConfig())
    assert d["strength_mapping"]["status"] == "ok"
    assert d["strength_mapping"]["family"] == "hybrid_gamma"
    assert "clamp" in d["strength_mapping"]["clamp_policy"]
    assert d["strength_mapping"]["source_path"] == "docs/v25/audits/a04_strength_curves.json"
    assert d["strength_mapping"]["a04_sha256_12"] is not None
    # 无实测族引擎 → 摘要如实为 None
    assert describe(TransferConfig.for_engine("student"))["strength_mapping"] is None


def test_override_validation():
    cfg = with_override(TransferConfig(), {"knob_default": 0.6, "gate": "fixed"})
    assert cfg.knob_default == 0.6 and cfg.gate == "fixed"
    assert config_hash(cfg) != config_hash(TransferConfig())
    with pytest.raises(ValueError, match="unknown keys"):
        with_override(TransferConfig(), {"not_a_field": 1})
    with pytest.raises(ValueError, match="接口"):
        with_override(TransferConfig(), {"engine": "o3_ref"})  # ref 引擎进 core 配置
    with pytest.raises(ValueError):
        with_override(RefConfig(), {"engine": "hybrid"})


def test_ref_config_frozen_defaults():
    cfg = RefConfig()
    assert cfg.engine == "o3_ref" and cfg.knob_default == 0.45
    assert cfg.v_type == "cnx" and cfg.use_conf_gate and cfg.use_global_fib
    assert cfg.wj == 2.2 and cfg.steps == 10          # t45 canonical（eval_o_tau_list 口径）
    assert cfg.feature_cache == "reference_v2"        # B02 ReferenceFeatureCache 契约
    assert cfg.same_mask is True                      # 协议 same150 恒等 bypass
    assert RefConfig.for_engine("o3_ref") == cfg
    with pytest.raises(ValueError):
        RefConfig.for_engine("hybrid")
    # O3 旋钮 = tau0（反向）：s 增大 → tau0 减小 → 编辑增强
    r_lo, r_hi = strength_to_knob(cfg, 0.1), strength_to_knob(cfg, 0.9)
    assert r_lo.knob == "tau0" and r_lo.value > r_hi.value
