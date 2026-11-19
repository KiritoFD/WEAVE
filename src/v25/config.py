"""B05: 统一 Minimal PRISM 配置规约（agent2.md §五 B05）。

用户接口只有两个（高级配置全部冻结在配置对象内部；``override`` 仅留给研究人员，
不进 ``transfer`` 签名）::

    PRISMCore.transfer(content, style, strength=0.5)      # engine 注册表见 ENGINE_REGISTRY
    PRISMRef.transfer(content, reference, strength=0.5)

本模块职责（纯 stdlib，不 import torch/numpy/exp，CPU 可全量测试）：

1. ``TransferConfig`` / ``RefConfig`` —— 高级配置冻结点（engine 选择、checkpoint、
   anchor、CFG 分支、Gate、solver、NFE、feature cache policy、seed）；裸构造
   ``TransferConfig()`` 即冻结的默认 core 配置（= ``for_engine(DEFAULT_CORE_ENGINE)``）。
2. engine 注册表 ``ENGINE_REGISTRY`` —— 多引擎并存，默认值集中一处
   （``DEFAULT_CORE_ENGINE`` / ``DEFAULT_REF_ENGINE`` 两个常量），便于总控在机制 B0
   出结果后一处切换；未就绪路径标 ``status="pending"``，不虚构能力。
3. 配置 hash —— ``config_hash`` 直接复用 :func:`src.v25.artifact_registry.param_hash`
   （与 A02 solver_hash / exp.v22.config_hash 同一序列化约定，**不另起第二套 hash**）；
   ``describe(config)`` 产出可恢复完整历史参数的台账 dict（hash + descriptor +
   engine 元数据 + strength 映射摘要）。
4. ``TransferConfig.from_descriptor`` —— 精确重建 V25 历史臂评测配置（B 曲线 τ-shift
   臂、Hybrid γ 臂、MechA A0/A1 臂），canonical descriptor 见
   ``HISTORICAL_ARM_DESCRIPTORS``。
5. strength → 旋钮映射 —— **只消费** docs/v25/audits/a04_strength_curves.json
   （A04 交付物，只读；本模块不重算曲线、不存第二份映射数值），含 UNMEASURED
   clamp 策略（s<0 → clamp 至最弱实测端点；s>1 → clamp 至最强实测端点）；
   文件缺失时 ``RuntimeWarning`` + 退化为恒等映射（knob = clamp(strength, 0, 1)）。

冻结常量的单一权威仍在 exp 侧（torch 环境）：``D6_SOLVER_FROZEN`` /
``D6H_SOLVER_FROZEN`` / ``TEACHER_SPEC_CANONICAL`` 是逐字段 frozen 副本，由
tests/v25/test_prism_v25.py 与 exp.v22.core.D6_SOLVER / exp.v25.r1_inference_ablation.
D6H_SOLVER / exp.v25.precompute_train_latents.TEACHER_SPEC 逐字段钉住 —— 权威漂移时
测试先红，保证不出现两套数值。配置 hash 只依赖本模块常量（不含 a04 文件内容，
避免 a04 增补实测臂导致已冻结 hash 漂移）；a04 文件血缘经
``describe()["strength_mapping"]["a04_sha256_12"]`` 单独记录。
"""
from __future__ import annotations

import dataclasses
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.v25.artifact_registry import REPO_ROOT, UNKNOWN, file_sha256, param_hash

__all__ = [
    "A04_AUDIT_RELPATH",
    "D6H_SOLVER_FROZEN",
    "D6_SOLVER_FROZEN",
    "DEFAULT_CORE_ENGINE",
    "DEFAULT_REF_ENGINE",
    "DEFAULT_STRENGTH",
    "DESCRIPTOR_FORMAT_VERSION",
    "ENGINE_REGISTRY",
    "EngineSpec",
    "HISTORICAL_ARM_DESCRIPTORS",
    "HISTORICAL_ARM_NAMES",
    "KnobReading",
    "O3_CKPT_RELPATH",
    "P1_CKPT_RELPATH",
    "RefConfig",
    "TEACHER_SPEC_CANONICAL",
    "TransferConfig",
    "V20_CKPT_RELPATH",
    "V25_EVAL_SEED",
    "a04_audit_path",
    "clear_strength_mapping_cache",
    "config_depends",
    "config_hash",
    "describe",
    "execution_seed",
    "get_engine",
    "load_strength_mapping",
    "restore_historical_arm",
    "strength_mapping_summary",
    "strength_to_knob",
    "with_override",
]

# --------------------------------------------------------------------------- #
# 冻结的 checkpoint 相对路径（仓内 posix；解析到磁盘时以 REPO_ROOT 为基）
# --------------------------------------------------------------------------- #
P1_CKPT_RELPATH = "runs/v19/students/s2_prism1_v2.pt"        # PRISM-1 (V19_S2, 1.5M 单步)
V20_CKPT_RELPATH = "runs/v20/slf_flow/slf_core_v2.pt"        # V20 D6 SLFUNet (22M 分段视界流)
O3_CKPT_RELPATH = "runs/v24/oneshot_val/O3_rgb_cnx_gated.pt"  # O3 One-Shot Minimal（参考条件）
A04_AUDIT_RELPATH = "docs/v25/audits/a04_strength_curves.json"  # A04 交付（只读消费）

# --------------------------------------------------------------------------- #
# 冻结 solver / 教师定义（frozen 副本；权威在 exp 侧，tests 钉住逐字段一致）
# --------------------------------------------------------------------------- #
#: 权威 = exp.v22.core.D6_SOLVER（V20 D6 默认 solver；τ∈[0.40, 0.74], 8 步）
D6_SOLVER_FROZEN: dict = {
    "tau_min": 0.40, "tau_hi": 0.74, "steps": 8,
    "ws_fig": 4.4, "wsp_fig_late": 1.8, "wsp_abs": 4.8,
    "wc_fig": 1.95, "wc_abs": 1.00, "lscale_abs": 0.44, "rescale_phi": 0.48,
}
#: 权威 = exp.v25.r1_inference_ablation.D6H_SOLVER = dict(D6_SOLVER, tau_min=0.48, tau_hi=0.78)
D6H_SOLVER_FROZEN: dict = {**D6_SOLVER_FROZEN, "tau_min": 0.48, "tau_hi": 0.78}

#: 权威 = exp.v25.precompute_train_latents.TEACHER_SPEC（机制 A/B 教师三元组铁律；
#: 与 exp.v25.mechanism_a.CANONICAL_TEACHER_SPEC 逐字段一致）
TEACHER_SPEC_CANONICAL: dict = {
    "d6_ckpt": V20_CKPT_RELPATH,
    "anchor": "B4_distill50 + canonicalizer(c1_gentle) [T4-equivalent]",
    "solver": dict(D6H_SOLVER_FROZEN),
    "branches": "cs_j",
    "gate": "original",
    "p1_ckpt": P1_CKPT_RELPATH,
    "p1_lambda": 1.0,
}

#: test750 评测 eps 生成种子（src.utils.protocol.EVAL_SEED；Round1/3 全臂共用同一
#: eps_750 —— 同 eps 跨臂是 a04 曲线配对插值的方法学前提）
V25_EVAL_SEED = 20261185
DEFAULT_STRENGTH = 0.5
DESCRIPTOR_FORMAT_VERSION = 1


def _norm_separators(obj: Any) -> Any:
    """路径分隔符归一（Windows 'runs\\\\x' 与 Linux 'runs/x' 必须判等；同 mechanism_a）。"""
    if isinstance(obj, str):
        return obj.replace("\\", "/")
    if isinstance(obj, dict):
        return {k: _norm_separators(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_norm_separators(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# engine 注册表：多引擎并存；默认值集中一处（DEFAULT_* 两个常量）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EngineSpec:
    """一个推理引擎的注册项（接口归属 / a04 映射族 / 旋钮 / 就绪状态 / 能力边界）。"""

    name: str
    interface: str                     # "core" | "ref"
    family: str | None                 # a04_strength_curves.json 的 families 键（None=无实测曲线）
    knob: str | None                   # strength 映射的旋钮名（lambda / gamma / tau_min / tau0）
    direction: str | None              # "direct"（旋钮↑=编辑↑）| "inverse" | None
    status: str                        # "available" | "pending"
    capabilities: tuple = ()
    ckpt_relpath: str | None = None
    knob_default: float | None = None  # 冻结的默认旋钮值（knob_resolution="fixed" 时生效）
    description: str = ""
    pending_reason: str = ""


#: 当前全部 available 引擎的可复用推理路径均为 test750 协议批形态（N=750，
#: 与 exp.v18.tasks_and_sources() 对齐）；单样本 / 显式 style 覆盖 / 任意单参考图
#: 编码路径尚未集成 —— capabilities 如实标注，runtime 层显式报 NotImplementedError。
CAP_PROTOCOL = "protocol_batch_750"
CAP_PROTO_REF = "protocol_reference_prototypes"

ENGINE_REGISTRY: dict[str, EngineSpec] = {
    # ---- PRISMCore 引擎 ----
    "prism1": EngineSpec(
        name="prism1", interface="core", family="prism1_lambda", knob="lambda",
        direction="direct", status="available", capabilities=(CAP_PROTOCOL,),
        ckpt_relpath=P1_CKPT_RELPATH, knob_default=1.0,
        description="PRISM-1 单步前馈（V19_S2, 1.5M, 17 ms/img）；旋钮 λ 直接加法风格注入 "
                    "(a04 曲线 A / prism1_lambda)；锚=latent chroma-shift α=0.20 内部固定。",
    ),
    "d6_flow": EngineSpec(
        name="d6_flow", interface="core", family="d6_tau_min", knob="tau_min",
        direction="inverse", status="available", capabilities=(CAP_PROTOCOL,),
        ckpt_relpath=V20_CKPT_RELPATH, knob_default=0.48,
        description="V20 D6 SLFUNet 分段视界流 τ-shift（04 文档 B 曲线 / 收敛主配置候选）；"
                    "旋钮 tau_min 反向（τ↑=编辑↓）；默认 solver=D6H(τ∈[0.48,0.78])+cs_j+T4 锚+original Gate。",
    ),
    "hybrid": EngineSpec(
        name="hybrid", interface="core", family="hybrid_gamma", knob="gamma",
        direction="direct", status="available", capabilities=(CAP_PROTOCOL,),
        ckpt_relpath=V20_CKPT_RELPATH, knob_default=0.55,
        description="Hybrid 线性混合 z=(1-γ)·PRISM-1(λ=1.0)+γ·V20-D6（04 文档 C 曲线；"
                    "★当前冻结默认 core engine，待机制 B0 出结果后可切 'student'）；"
                    "d6 分支冻结口径=T4 锚+full CFG+original Gate+D6_SOLVER(τ∈[0.40,0.74])（V24 基座缓存同口径）。",
    ),
    "mecha_a": EngineSpec(
        name="mecha_a", interface="core", family="hybrid_gamma", knob="gamma",
        direction="direct", status="available", capabilities=(CAP_PROTOCOL,),
        ckpt_relpath=V20_CKPT_RELPATH, knob_default=0.55,
        description="机制 A 空间门控臂（A0_global M≡1 / A1_heuristic κ(1-ē)），教师三元组="
                    "TEACHER_SPEC_CANONICAL（D6H+cs_j+B4 锚）；γ 旋钮复用 a04 hybrid_gamma 族校准"
                    "（注意：该曲线实测于 full-CFG z_d6，cs_j 教师口径下为近似映射，已如实标注）。",
    ),
    "student": EngineSpec(
        name="student", interface="core", family=None, knob=None, direction=None,
        status="pending", capabilities=(), ckpt_relpath=None, knob_default=None,
        description="机制 B 学生蒸馏（B0 PRISM-1 热启动 → B1 更宽学生）；checkpoint 未产出。",
        pending_reason="mechanism B pending",
    ),
    # ---- PRISMRef 引擎 ----
    "o3_ref": EngineSpec(
        name="o3_ref", interface="ref", family="o3_tau0", knob="tau0",
        direction="inverse", status="available", capabilities=(CAP_PROTOCOL, CAP_PROTO_REF),
        ckpt_relpath=O3_CKPT_RELPATH, knob_default=0.45,
        description="O3 One-Shot Minimal（RGB+ConvNeXt gated, 参考条件, 10 步 RK2+wj 引导）；"
                    "旋钮 tau0 反向（τ0↑=编辑↓）。已知混杂（a04 notes）：t40/t60 补扫臂的 wj 不同"
                    "（2.3/1.9），曲线非纯 τ0 单变量；默认 wj 冻结 2.2（t45 canonical 口径）。",
    ),
}

#: 冻结默认值集中点：总控切换默认 engine 只改这两个常量（+各 EngineSpec.knob_default）
DEFAULT_CORE_ENGINE = "hybrid"
DEFAULT_REF_ENGINE = "o3_ref"


def get_engine(name: str) -> EngineSpec:
    """engine 注册表查找；未知 engine 报 ValueError 并列出可用项。"""
    spec = ENGINE_REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"unknown engine '{name}'; available: {sorted(ENGINE_REGISTRY)}")
    return spec


# --------------------------------------------------------------------------- #
# TransferConfig / RefConfig：高级配置冻结点（frozen dataclass）
# --------------------------------------------------------------------------- #
_ENGINE_BASES: dict[str, dict] = {
    # 裸构造 TransferConfig() 的类级默认 = "hybrid" 行（冻结默认配置）。
    "prism1": dict(
        knob_default=1.0, p1_ckpt=P1_CKPT_RELPATH, p1_lambda=1.0, d6_ckpt=None,
        solver=None, branches=None, gate=None, anchor=None, steps=1,
        mecha_arm=None, mecha_kappa=None, mecha_energy_match=False, teacher_spec=None,
    ),
    "d6_flow": dict(
        knob_default=0.48, p1_ckpt=None, p1_lambda=None, d6_ckpt=V20_CKPT_RELPATH,
        solver=dict(D6H_SOLVER_FROZEN), branches="cs_j", gate="original", anchor="t4", steps=8,
        mecha_arm=None, mecha_kappa=None, mecha_energy_match=False, teacher_spec=None,
    ),
    "hybrid": dict(
        knob_default=0.55, p1_ckpt=P1_CKPT_RELPATH, p1_lambda=1.0, d6_ckpt=V20_CKPT_RELPATH,
        solver=dict(D6_SOLVER_FROZEN), branches="full", gate="original", anchor="t4", steps=8,
        mecha_arm=None, mecha_kappa=None, mecha_energy_match=False, teacher_spec=None,
    ),
    "mecha_a": dict(
        knob_default=0.55, p1_ckpt=None, p1_lambda=None, d6_ckpt=None,
        solver=None, branches=None, gate=None, anchor=None, steps=None,
        mecha_arm="A0_global", mecha_kappa=None, mecha_energy_match=False,
        teacher_spec=dict(TEACHER_SPEC_CANONICAL),
    ),
    "student": dict(
        knob_default=None, p1_ckpt=None, p1_lambda=None, d6_ckpt=None,
        solver=None, branches=None, gate=None, anchor=None, steps=None,
        mecha_arm=None, mecha_kappa=None, mecha_energy_match=False, teacher_spec=None,
    ),
}


@dataclass(frozen=True)
class TransferConfig:
    """PRISMCore 高级配置（冻结默认 = Hybrid γ 路线）。

    用户只需 ``PRISMCore().transfer(content, style, strength)``；本对象由
    ``for_engine()`` 构造或研究人员经 ``with_override`` 显式覆盖。字段语义：

    - ``engine``                  core 引擎名（注册表键；默认冻结 "hybrid"）；
    - ``strength_map_source``     "a04"（消费 A04 审计 JSON；恒等退化是运行时行为，不改配置）；
    - ``knob_resolution``         "strength_map"：transfer(strength) 经 a04 映射求旋钮值；
                                  "fixed"：恒用 ``knob_default``（历史臂精确恢复路径）；
    - ``knob_default``            冻结默认旋钮值（hybrid γ=0.55 = Round1 曲线 C 工作点，
                                  同 exp.v25.mechanism_a.TRAIN_GAMMA_DEFAULT）；
    - ``p1_ckpt`` / ``p1_lambda`` PRISM-1 分支（hybrid 的 λ 固定 1.0，同曲线 C 定义）；
    - ``d6_ckpt`` / ``solver``    D6 分支权重与 solver dict（hybrid=V24 基座口径 D6_SOLVER；
                                  d6_flow=教师口径 D6H_SOLVER）；
    - ``branches`` / ``gate`` / ``gate_fixed_g`` / ``anchor`` CFG 分支 / 门控 / 锚点；
    - ``steps``                   NFE（单步引擎=1；流引擎=solver steps；mecha_a 由 teacher_spec 决定）；
    - ``mecha_*``                 机制 A 臂参数（A0_global / A1_heuristic + κ + 能量匹配）；
    - ``teacher_spec``            TEACHER_SPEC_CANONICAL 副本（mecha_a 的唯一教师权威）；
    - ``seed``                    执行 eps 种子（默认 20261185 = EVAL_SEED；Round1/3 全臂同 eps
                                  是 a04 配对插值的方法学前提；None → 由 config hash 派生）。
    """

    engine: str = DEFAULT_CORE_ENGINE
    strength_map_source: str = "a04"
    knob_resolution: str = "strength_map"
    knob_default: float | None = 0.55
    p1_ckpt: str | None = P1_CKPT_RELPATH
    p1_lambda: float | None = 1.0
    d6_ckpt: str | None = V20_CKPT_RELPATH
    solver: dict | None = field(default_factory=lambda: dict(D6_SOLVER_FROZEN))
    branches: str | None = "full"
    gate: str | None = "original"
    gate_fixed_g: float = 0.5
    anchor: str | None = "t4"
    steps: int | None = 8
    mecha_arm: str | None = None
    mecha_kappa: float | None = None
    mecha_energy_match: bool = False
    teacher_spec: dict | None = None
    seed: int | None = V25_EVAL_SEED
    format_version: int = DESCRIPTOR_FORMAT_VERSION

    # ------------------------------------------------------------ 构造 ----
    @classmethod
    def for_engine(cls, name: str, **overrides: Any) -> "TransferConfig":
        """按注册表构造某 core 引擎的冻结默认配置（可再叠加显式覆盖）。"""
        spec = get_engine(name)
        if spec.interface != "core":
            raise ValueError(f"engine '{name}' 是 {spec.interface} 引擎，不属于 TransferConfig（用 RefConfig）")
        if spec.status != "available" and name != "student":
            raise ValueError(f"engine '{name}' status={spec.status}，禁止构造")
        base = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _ENGINE_BASES[name].items()}
        base["engine"] = name
        base.update(overrides)
        return cls(**base)

    # ------------------------------------------------------- descriptor ----
    def to_descriptor(self) -> dict:
        """canonical descriptor（含 kind + 全字段；路径分隔符归一；hash 的唯一输入）。"""
        d: dict[str, Any] = {"kind": "TransferConfig", "format_version": self.format_version}
        for f in dataclasses.fields(self):
            d[f.name] = _norm_separators(getattr(self, f.name))
        return d

    @classmethod
    def from_descriptor(cls, d: Mapping[str, Any]) -> "TransferConfig":
        """从 descriptor 精确重建配置（V25 历史臂恢复入口；键严格校验）。"""
        field_names = {f.name for f in dataclasses.fields(cls)}
        expected = field_names | {"kind"}
        keys = set(d.keys())
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing or extra:
            raise ValueError(f"descriptor keys mismatch: missing={missing}, unexpected={extra}")
        if d["kind"] != "TransferConfig":
            raise ValueError(f"descriptor kind '{d['kind']}' != 'TransferConfig'")
        if int(d["format_version"]) != DESCRIPTOR_FORMAT_VERSION:
            raise ValueError(
                f"descriptor format_version {d['format_version']} != {DESCRIPTOR_FORMAT_VERSION}")
        kwargs = {k: _norm_separators(d[k]) for k in sorted(field_names)}
        cfg = cls(**kwargs)
        _validate_config_engine(cfg, "core")
        return cfg

    # ------------------------------------------------------------ hash ----
    @property
    def config_hash(self) -> str:
        """确定性配置 hash（= param_hash(descriptor)，单一约定）。"""
        return config_hash(self)


@dataclass(frozen=True)
class RefConfig:
    """PRISMRef 高级配置（冻结默认 = O3 One-Shot t45 canonical 口径）。

    - ``knob_default``     τ0=0.45（O3_rgb_cnx_gated_t45 canonical 臂，v24 评测口径）；
    - ``ckpt``             O3_rgb_cnx_gated.pt（runs/v24/oneshot_val，r3 复用同权重）；
    - ``v_type``/``use_conf_gate``/``use_global_fib``  O_SPECS["O3_rgb_cnx_gated"] 镜像
                           (cnx / conf-gated / global-fib)；
    - ``wj``               内容引导权重冻结 2.2（t45 canonical；a04 notes：补扫臂 t40/t60
                           的 wj 2.3/1.9 属混杂，不改默认）；
    - ``steps``            10（RK2 中点，同 eval_o_tau_list）；
    - ``feature_cache``    "reference_v2"：B02 ReferenceFeatureCache（runs/v25/cache/ref_feats，
                           fp32 + 批维 v2 契约）；
    - ``same_mask``        True = 协议 same150 恒等 bypass（O3 评测口径；协议批恒开）；
    - ``seed``             执行 eps 种子（默认 20261185，同 O3 评测 eps_750 口径）。
    """

    engine: str = DEFAULT_REF_ENGINE
    strength_map_source: str = "a04"
    knob_resolution: str = "strength_map"
    knob_default: float = 0.45
    ckpt: str = O3_CKPT_RELPATH
    v_type: str = "cnx"
    use_conf_gate: bool = True
    use_global_fib: bool = True
    wj: float = 2.2
    steps: int = 10
    feature_cache: str = "reference_v2"
    same_mask: bool = True
    seed: int | None = V25_EVAL_SEED
    format_version: int = DESCRIPTOR_FORMAT_VERSION

    @classmethod
    def for_engine(cls, name: str, **overrides: Any) -> "RefConfig":
        spec = get_engine(name)
        if spec.interface != "ref":
            raise ValueError(f"engine '{name}' 是 {spec.interface} 引擎，不属于 RefConfig（用 TransferConfig）")
        if name != "o3_ref":
            raise ValueError(f"engine '{name}' 无冻结 RefConfig 基座；available ref engines: ['o3_ref']")
        base = dict(engine=name)
        base.update(overrides)
        return cls(**base)

    def to_descriptor(self) -> dict:
        d: dict[str, Any] = {"kind": "RefConfig", "format_version": self.format_version}
        for f in dataclasses.fields(self):
            d[f.name] = _norm_separators(getattr(self, f.name))
        return d

    @classmethod
    def from_descriptor(cls, d: Mapping[str, Any]) -> "RefConfig":
        field_names = {f.name for f in dataclasses.fields(cls)}
        expected = field_names | {"kind"}
        keys = set(d.keys())
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing or extra:
            raise ValueError(f"descriptor keys mismatch: missing={missing}, unexpected={extra}")
        if d["kind"] != "RefConfig":
            raise ValueError(f"descriptor kind '{d['kind']}' != 'RefConfig'")
        if int(d["format_version"]) != DESCRIPTOR_FORMAT_VERSION:
            raise ValueError(
                f"descriptor format_version {d['format_version']} != {DESCRIPTOR_FORMAT_VERSION}")
        kwargs = {k: _norm_separators(d[k]) for k in sorted(field_names)}
        cfg = cls(**kwargs)
        _validate_config_engine(cfg, "ref")
        return cfg

    @property
    def config_hash(self) -> str:
        return config_hash(self)


def _validate_config_engine(cfg: Any, interface: str) -> None:
    spec = get_engine(cfg.engine)
    if spec.interface != interface:
        raise ValueError(
            f"engine '{cfg.engine}' 注册为 '{spec.interface}' 引擎，与配置接口 '{interface}' 不符")


def with_override(config: TransferConfig | RefConfig, override: Mapping[str, Any] | None):
    """研究人员显式覆盖（不改变用户接口）；未知键一律 ValueError，覆盖后重新校验 engine。"""
    if not override:
        return config
    field_names = {f.name for f in dataclasses.fields(config)}
    unknown = sorted(set(override.keys()) - field_names)
    if unknown:
        raise ValueError(f"override unknown keys {unknown}; allowed: {sorted(field_names)}")
    new = dataclasses.replace(config, **dict(override))
    _validate_config_engine(new, get_engine(config.engine).interface)
    return new


# --------------------------------------------------------------------------- #
# 配置 hash / 台账 / 执行种子（复用 artifact_registry.param_hash，单一约定）
# --------------------------------------------------------------------------- #
def config_hash(config: TransferConfig | RefConfig) -> str:
    """配置确定性 hash = ``param_hash(config.to_descriptor())``。

    与 B03 注册表 / A02 solver_hash / exp.v22.config_hash 完全同一序列化约定
    （json sort_keys + float repr 全精度）；键序无关、跨平台稳定。
    """
    return param_hash(config.to_descriptor())


def config_depends(config: TransferConfig | RefConfig) -> dict[str, str]:
    """供 B03 ArtifactRegistry.register(depends=...) 的依赖键（params_hash 对齐）。"""
    return {"params_hash": config_hash(config)}


def describe(config: TransferConfig | RefConfig) -> dict:
    """台账 dict：hash + descriptor + engine 元数据 + strength 映射摘要。

    从 ``hash + registry 记录`` 恢复历史参数的完整载荷即 ``desc["descriptor"]``
    （``from_descriptor`` 可逐键重建）。
    """
    spec = get_engine(config.engine)
    interface = spec.interface
    summary = strength_mapping_summary(spec.family) if spec.family is not None else None
    out = {
        "descriptor": config.to_descriptor(),
        "config_hash": config_hash(config),
        "format_version": DESCRIPTOR_FORMAT_VERSION,
        "engine": {
            "name": spec.name,
            "interface": spec.interface,
            "family": spec.family,
            "knob": spec.knob,
            "direction": spec.direction,
            "status": spec.status,
            "capabilities": list(spec.capabilities),
            "ckpt_relpath": spec.ckpt_relpath,
            "knob_default": spec.knob_default,
            "description": spec.description,
            "pending_reason": spec.pending_reason,
        },
        "strength_mapping": summary,
        "frozen_defaults_note": (
            "默认冻结点: DEFAULT_CORE_ENGINE / DEFAULT_REF_ENGINE / ENGINE_REGISTRY[*].knob_default; "
            "总控切换默认 engine 只改这两个常量"),
    }
    return out


def execution_seed(config: TransferConfig | RefConfig) -> int:
    """执行 eps 种子（strength 无关 → 同 config 不同 strength 共享噪声路径）。

    - ``config.seed`` 非 None（冻结默认 20261185 = EVAL_SEED）：直接采用 —— Round1/3
      全臂共用同一 eps_750，是 a04 曲线配对插值的方法学前提，也是历史臂精确恢复条件；
    - ``config.seed`` 为 None：由 config descriptor hash 派生 32-bit 种子（仍确定性）。
    """
    if config.seed is not None:
        return int(config.seed)
    payload = {"purpose": "prism_v25.transfer_eps", "config": config.to_descriptor()}
    return int(param_hash(payload, n=8), 16)


# --------------------------------------------------------------------------- #
# strength → 旋钮映射（只消费 a04 JSON；缺失 → WARNING + 恒等退化）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KnobReading:
    """一次 strength 解析结果（含来源与 clamp 标注，供日志/台账）。"""

    knob: str | None
    value: float
    clamped: bool
    source: str  # "a04:<family>" | "identity_fallback" | "fixed"


def a04_audit_path() -> Path:
    return REPO_ROOT / A04_AUDIT_RELPATH


_MAPPING_CACHE: dict[tuple[str, str], dict | None] = {}


def clear_strength_mapping_cache() -> None:
    """清空映射缓存（测试 / a04 文件热更新后使用）。"""
    _MAPPING_CACHE.clear()


def load_strength_mapping(family: str, *, path: str | Path | None = None,
                          use_cache: bool = True) -> dict | None:
    """加载 a04 某映射族的 strength→knob 节点表；文件缺失/损坏 → None + RuntimeWarning。

    返回结构（数值一律来自 JSON，本模块不重算）：``{family, knob, direction, knots
    (按 strength 升序，含 0.0/1.0 实测端点), clamp_policy, notes, beyond_measured,
    descriptive_only_policy, source_path, a04_sha256_12}``。
    """
    p = Path(path) if path is not None else a04_audit_path()
    key = (family, str(p))
    if use_cache and key in _MAPPING_CACHE:
        return _MAPPING_CACHE[key]
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"[B05] strength 映射不可用（{type(exc).__name__}: {exc}; path={p.name}）；"
            f"PRISM strength 退化为恒等映射 knob=clamp(strength,0,1)。"
            f"请恢复 {A04_AUDIT_RELPATH}（A04 交付物）。",
            RuntimeWarning, stacklevel=2)
        _MAPPING_CACHE[key] = None
        return None
    fam = (data.get("families") or {}).get(family)
    if not isinstance(fam, dict) or not isinstance(fam.get("strength_mapping"), dict):
        raise ValueError(
            f"a04 audit 缺少 family '{family}'（或其 strength_mapping）；"
            f"available={sorted((data.get('families') or {}).keys())}, path={p.name}")
    sm = fam["strength_mapping"]
    knots_raw = sm.get("knots_measured") or []
    if len(knots_raw) < 2:
        raise ValueError(f"a04 family '{family}' knots_measured < 2，无法构建映射")
    knots = [{"arm": str(k["arm"]), "knob_value": float(k["knob_value"]),
              "strength": float(k["strength"])} for k in knots_raw]
    strengths = [k["strength"] for k in knots]
    if any(b <= a for a, b in zip(strengths, strengths[1:])):
        raise ValueError(f"a04 family '{family}' knots strength 非严格递增: {strengths}")
    if abs(strengths[0]) > 1e-9 or abs(strengths[-1] - 1.0) > 1e-9:
        raise ValueError(
            f"a04 family '{family}' 实测端点未覆盖 s=0/1: [{strengths[0]}, {strengths[-1]}]；"
            f"clamp 策略要求两端点为实测臂")
    sha = file_sha256(p)
    mapping = {
        "family": family,
        "knob": str(fam["knob"]),
        "direction": str(fam["direction"]),
        "engine": str(fam.get("engine", "")),
        "knots": knots,
        "clamp_policy": str(sm.get("clamp_policy", "")),
        "notes": str(sm.get("notes", "")),
        "beyond_measured": list(sm.get("beyond_measured") or []),
        "descriptive_only_policy": str(data.get("descriptive_only_policy", "")),
        "source_path": A04_AUDIT_RELPATH,
        "a04_sha256_12": None if sha == UNKNOWN else sha[:12],
    }
    _MAPPING_CACHE[key] = mapping
    return mapping


def strength_mapping_summary(family: str, *, path: str | Path | None = None) -> dict | None:
    """describe() 用的映射摘要（失败时如实标注 identity_fallback 可用，不抛错）。"""
    try:
        m = load_strength_mapping(family, path=path)
    except ValueError as exc:
        return {"status": "invalid", "family": family, "error": str(exc),
                "source_path": A04_AUDIT_RELPATH}
    if m is None:
        return {"status": "unavailable_identity_fallback", "family": family,
                "source_path": A04_AUDIT_RELPATH, "a04_sha256_12": None}
    knob_values = [k["knob_value"] for k in m["knots"]]
    return {
        "status": "ok",
        "family": family,
        "knob": m["knob"],
        "direction": m["direction"],
        "measured_strength_range": [m["knots"][0]["strength"], m["knots"][-1]["strength"]],
        "measured_knob_range": [min(knob_values), max(knob_values)],
        "n_knots": len(m["knots"]),
        "knot_arms": [k["arm"] for k in m["knots"]],
        "clamp_policy": m["clamp_policy"],
        "notes": m["notes"],
        "beyond_measured": m["beyond_measured"],
        "descriptive_only_policy": m["descriptive_only_policy"],
        "source_path": m["source_path"],
        "a04_sha256_12": m["a04_sha256_12"],
    }


def _identity_knob(knob: str | None, strength: float) -> KnobReading:
    v = min(1.0, max(0.0, strength))
    return KnobReading(knob=knob, value=v, clamped=(v != strength), source="identity_fallback")


def strength_to_knob(config: TransferConfig | RefConfig, strength: float, *,
                     mapping: dict | None = None) -> KnobReading:
    """strength s∈ℝ → 旋钮值（消费 a04 knots 分段线性插值；UNMEASURED 区间 clamp）。

    - ``knob_resolution == "fixed"``：恒用 ``knob_default``（历史臂精确恢复）；
    - 否则 s 先按 a04 ``clamp_policy`` 收进 [0,1]（s<0 → 最弱实测端点；s>1 → 最强实测
      端点；两区间为 UNMEASURED，``clamped=True`` 如实标注），再在相邻实测节点间线性插值；
    - 映射不可用（文件缺失）→ 恒等退化 ``knob=clamp(s,0,1)``（与 load 的 WARNING 配套）。
    """
    s = float(strength)
    if math.isnan(s) or math.isinf(s):
        raise ValueError(f"strength must be finite, got {strength!r}")
    spec = get_engine(config.engine)
    if config.knob_resolution == "fixed":
        if config.knob_default is None:
            raise ValueError(f"engine '{config.engine}' knob_resolution='fixed' 但 knob_default=None")
        return KnobReading(knob=spec.knob, value=float(config.knob_default), clamped=False, source="fixed")
    if spec.family is None:  # pending / 无实测曲线引擎：恒等退化（transfer 层另有 pending 闸门）
        return _identity_knob(spec.knob, s)
    m = mapping if mapping is not None else load_strength_mapping(spec.family)
    if m is None:
        return _identity_knob(spec.knob, s)
    knots = m["knots"]
    clamped = s < knots[0]["strength"] or s > knots[-1]["strength"]
    sc = min(knots[-1]["strength"], max(knots[0]["strength"], s))
    lo, hi = knots[0], knots[-1]
    for a, b in zip(knots, knots[1:]):
        if a["strength"] <= sc <= b["strength"]:
            lo, hi = a, b
            break
    span = hi["strength"] - lo["strength"]
    value = lo["knob_value"] if span <= 0 else (
        lo["knob_value"] + (sc - lo["strength"]) / span * (hi["knob_value"] - lo["knob_value"]))
    return KnobReading(knob=m["knob"], value=float(value), clamped=clamped,
                       source=f"a04:{m['family']}")


# --------------------------------------------------------------------------- #
# V25 历史臂精确恢复（04 文档 / a04 JSON / mechanism_a 口径为准）
# --------------------------------------------------------------------------- #
def _historical_arm_descriptors() -> dict[str, dict]:
    """canonical 历史臂 descriptor（构造自冻结常量；provenance 见各注释）。"""
    out: dict[str, dict] = {}
    # B 曲线 τ-shift 臂（04 文档曲线 B / a04 d6_tau_min 族；r1: dict(D6_SOLVER, tau_min=t, tau_hi=0.78),
    # branches="full", T4 锚, original Gate）—— tau_min 是 strength 旋钮（反向）。
    for arm, tmin in (("B_d6_t050", 0.50), ("B_d6_t072", 0.72)):
        out[arm] = TransferConfig.for_engine(
            "d6_flow", branches="full", solver=dict(D6H_SOLVER_FROZEN, tau_min=tmin),
            knob_resolution="fixed", knob_default=tmin).to_descriptor()
    # Hybrid γ 臂（04 文档曲线 C / a04 hybrid_gamma 族；z=(1-γ)·P1(λ=1.0)+γ·z_d6，
    # d6 分支=V24 基座缓存口径 D6_SOLVER+full+original+T4）。
    for arm, gamma in (("C_hyb_g70", 0.70), ("C_hyb_g85", 0.85)):
        out[arm] = TransferConfig.for_engine(
            "hybrid", knob_resolution="fixed", knob_default=gamma).to_descriptor()
    # 机制 A 臂（exp.v25.mechanism_a 评测口径：教师三元组 CANONICAL_TEACHER_SPEC，
    # γ=0.55 = TRAIN_GAMMA_DEFAULT；A1 κ=0.5 为 --kappa 网格首档）。
    out["MechA_A0_g55"] = TransferConfig.for_engine(
        "mecha_a", mecha_arm="A0_global", knob_resolution="fixed", knob_default=0.55).to_descriptor()
    out["MechA_A1_k50_g55"] = TransferConfig.for_engine(
        "mecha_a", mecha_arm="A1_heuristic", mecha_kappa=0.5,
        knob_resolution="fixed", knob_default=0.55).to_descriptor()
    return out


HISTORICAL_ARM_DESCRIPTORS: dict[str, dict] = _historical_arm_descriptors()
HISTORICAL_ARM_NAMES: tuple = tuple(sorted(HISTORICAL_ARM_DESCRIPTORS))


def restore_historical_arm(name: str) -> TransferConfig:
    """按名恢复 V25 历史臂评测配置（未知臂名列出可用项）。"""
    desc = HISTORICAL_ARM_DESCRIPTORS.get(name)
    if desc is None:
        raise ValueError(f"unknown historical arm '{name}'; available: {list(HISTORICAL_ARM_NAMES)}")
    return TransferConfig.from_descriptor(desc)
