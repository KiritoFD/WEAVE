"""B05: Minimal PRISM 双接口骨架（PRISMCore / PRISMRef）— agent2.md §五 B05。

用户接口（验收铁律：``transfer`` 签名只有 content / style|reference / strength）::

    PRISMCore.transfer(content, style, strength=0.5)
    PRISMRef.transfer(content, reference, strength=0.5)

设计铁律：

- **lazy import（B02 手法）**：本模块顶层不 import torch / numpy / exp.*；接口层纯逻辑
  （knob 映射、执行种子、engine 注册、pending 判定、runtime 注入）全部 CPU 可测。
  ``import src.model.prism_v25`` 在无 GPU 环境（``torch.cuda.device_count()==0``）不崩，
  且不拉起任何 exp 重模块（tests 有子进程断言）。
- **执行体两类**（验收 §六）：(a) 已有可复用路径做成薄封装 —— Core 的
  ``run_core_flow`` / Hybrid γ 混合 / PRISM-1 单步 / MechA 教师口径臂、Ref 的 O3 推理
  （10 步 RK2 + wj 引导，逐式同 eval_o_tau_list）；(b) 未就绪路径（机制 B 学生 ckpt）
  ``engine="student"`` → ``NotImplementedError("mechanism B pending")``，注册表
  ``status="pending"`` —— 禁止虚构能力。
- **能力边界如实标注**：当前全部 available 引擎的可复用路径均为 **test750 协议批形态**
  （N=750，与 exp.v18.tasks_and_sources() 对齐；单样本 / 显式 style 覆盖 / 任意单参考图
  编码路径待总控集成，runtime 层显式 NotImplementedError，注册表 capabilities 如实记录）。
  协议批模式下 ``style``/``reference`` 传 None 即采用协议目标（same150 恒等 bypass）。
- **确定性**：执行 eps 种子 = ``execution_seed(config)``（strength 无关、config 确定）；
  同 config 不同 strength 共享噪声路径（Round1/3 同 eps 跨臂方法学），同 strength
  重复调用逐位一致。
- **S10 契约**：解码一律 ``src.utils.decode_s10``（内部 /0.18215 + 自带 post_quant_conv，
  恰好应用一次），绝不调 AutoencoderKL.decode（AGENTS.md 铁律）。

runtime 注入协议：``runtime.run(**kwargs) -> dict``（生产实现 CoreRuntime/RefRuntime，
测试可注入 fake runtime 验证 seed/knob 路径）。返回 dict 含
``engine / interface / knob / knob_value / strength / seed / z（缩放 latent）/ rgb01``。
"""
from __future__ import annotations

from typing import Any, Mapping

from src.v25.artifact_registry import REPO_ROOT
from src.v25.config import (
    DEFAULT_CORE_ENGINE,
    DEFAULT_REF_ENGINE,
    DEFAULT_STRENGTH,
    RefConfig,
    TransferConfig,
    config_hash,
    describe as describe_config,
    execution_seed,
    get_engine,
    strength_to_knob,
    with_override,
)

__all__ = [
    "CoreRuntime",
    "PRISMCore",
    "PRISMRef",
    "RefRuntime",
    "STUDENT_PENDING_MESSAGE",
    "build_core_runtime",
    "build_ref_runtime",
]

STUDENT_PENDING_MESSAGE = "mechanism B pending"

#: 生产 runtime.run 的统一返回键（协议批：z (750,4,64,64) 缩放 latent + rgb01 (750,3,512,512)）
RESULT_KEYS = ("engine", "interface", "knob", "knob_value", "strength", "seed", "z", "rgb01")


# --------------------------------------------------------------------------- #
# 接口层（纯逻辑；CPU 可测；不触碰 torch/exp）
# --------------------------------------------------------------------------- #
class _PRISMBase:
    """共享骨架：config 冻结 / override 校验 / engine 接口一致性 / hash / describe。"""

    _config_cls = None
    _interface: str = ""
    _default_engine: str = ""

    def __init__(self, config: Any = None, *,
                 runtime: Any = None, resources: Any = None,
                 override: Mapping[str, Any] | None = None):
        if config is None:
            config = self._config_cls()
        if override:
            config = with_override(config, override)
        spec = get_engine(config.engine)
        if spec.interface != self._interface:
            raise ValueError(
                f"engine '{config.engine}' 注册为 '{spec.interface}' 引擎，"
                f"与 {type(self).__name__} 接口 '{self._interface}' 不符")
        self._config = config
        self._runtime = runtime
        self._resources = resources

    # ---- 只读视图 ----
    @property
    def config(self):
        return self._config

    @property
    def engine(self) -> str:
        return self._config.engine

    @property
    def config_hash(self) -> str:
        return config_hash(self._config)

    def describe(self) -> dict:
        """台账 dict（hash + descriptor + engine 元数据 + strength 映射摘要）。"""
        return describe_config(self._config)

    # ---- 执行公共路径 ----
    def _execute(self, strength: float, payload: Mapping[str, Any]) -> dict:
        cfg = self._config
        spec = get_engine(cfg.engine)
        if spec.status == "pending":
            raise NotImplementedError(
                f"{STUDENT_PENDING_MESSAGE} (engine='{cfg.engine}': {spec.pending_reason or spec.description})")
        knob = strength_to_knob(cfg, strength)
        if self._runtime is None:
            raise RuntimeError(
                f"{type(self).__name__} runtime 未准备：available engine '{cfg.engine}' 的执行体需要 GPU "
                f"资源（exp 路径 lazy import，接口层不加载任何 checkpoint）；"
                f"先调用 prepare(pool=..., device=...) 或注入测试 runtime")
        return self._runtime.run(
            engine=cfg.engine, interface=self._interface, strength=float(strength),
            knob=knob.knob, knob_value=knob.value, seed=execution_seed(cfg),
            config=cfg, resources=self._resources, **payload)


class PRISMCore(_PRISMBase):
    """内容+画派条件核（engine: prism1 / d6_flow / hybrid / mecha_a / student）。

    - ``config``       TransferConfig（None = 冻结默认 hybrid）；``override`` 仅研究人员用；
    - ``runtime``      生产 CoreRuntime 或测试 fake（``run(**kw) -> dict``）；
    - ``transfer``     用户唯一入口；当前可复用路径为 test750 协议批：
                       ``content=None`` + ``style=None`` → 协议 750 任务（源 latent + 协议目标）；
                       显式单样本 / style 覆盖路径 pending（runtime 层明确报错，见 capabilities）。
    """

    _config_cls = TransferConfig
    _interface = "core"
    _default_engine = DEFAULT_CORE_ENGINE

    def transfer(self, content, style, strength: float = DEFAULT_STRENGTH) -> dict:
        """核心迁移入口（签名即接口契约：content / style / strength，无其余用户参数）。

        strength 经 a04 映射解析为当前 engine 的旋钮值（缺失时恒等退化 + WARNING）；
        执行种子由 config 决定（strength 无关）→ 同 config 跨 strength 共享噪声路径。
        """
        return self._execute(strength, {"content": content, "style": style})

    def prepare(self, pool: str | None = None, device: str = "cuda") -> "PRISMCore":
        """GPU 主机上构建生产 runtime + 协议资源（lazy import，不在 import 期执行）。"""
        runtime = CoreRuntime()
        runtime.prepare(self._config, pool=pool, device=device)
        self._runtime = runtime
        self._resources = runtime.resources
        return self


class PRISMRef(_PRISMBase):
    """参考图条件核（engine: o3_ref）。

    - ``transfer``     用户唯一入口；``reference`` 当前实现 = 参考条件选择器：
                       None → 协议目标（SG-5 画派原型，O3 评测同构）；int/(B,) 张量 →
                       目标画派原型索引覆盖；任意单参考图 RGB 编码路径 pending（R5/机制 C
                       集成后开放，runtime 层明确报错，见 capabilities）。
    """

    _config_cls = RefConfig
    _interface = "ref"
    _default_engine = DEFAULT_REF_ENGINE

    def transfer(self, content, reference, strength: float = DEFAULT_STRENGTH) -> dict:
        """参考条件迁移入口（签名即接口契约：content / reference / strength）。"""
        return self._execute(strength, {"content": content, "reference": reference})

    def prepare(self, pool: str | None = None, device: str = "cuda") -> "PRISMRef":
        runtime = RefRuntime()
        runtime.prepare(self._config, pool=pool, device=device)
        self._runtime = runtime
        self._resources = runtime.resources
        return self


def build_core_runtime(config: TransferConfig | None = None, *,
                       pool: str | None = None, device: str = "cuda") -> CoreRuntime:
    """薄封装入口：构建已 prepare 的 CoreRuntime（GPU 主机使用）。"""
    return CoreRuntime().prepare(config or TransferConfig(), pool=pool, device=device)


def build_ref_runtime(config: RefConfig | None = None, *,
                      pool: str | None = None, device: str = "cuda") -> RefRuntime:
    return RefRuntime().prepare(config or RefConfig(), pool=pool, device=device)


# --------------------------------------------------------------------------- #
# 生产 runtime（薄封装既有可复用路径；torch/exp 全部函数体内 lazy import）
# --------------------------------------------------------------------------- #
_DEFAULT_POOL_PARTS = ("pseudo_gt_24h", "orig_mean_fp16")  # 仓库外 sibling pool（r1/r3 同默认）


def _pool_path(pool: str | None):
    if pool is not None:
        return pool
    return REPO_ROOT.parent.joinpath(*_DEFAULT_POOL_PARTS)


class CoreRuntime:
    """PRISMCore 生产执行体：薄封装 exp.v25 / exp.v24 / exp.v22 既有可复用路径。

    - ``d6_flow``  → exp.v25.r1_inference_ablation.run_core_flow（同函数、同参数语义，
      仅 tau_min 由 strength 旋钮替换；τ 分布如实返回）；
    - ``hybrid``   → compute_prism1_latents(λ=p1_lambda) + run_core_flow(d6 分支) 后
      线性混合 (1-γ)·z_p1 + γ·z_d6（r1 曲线 C 逐式同构）；
    - ``prism1``   → compute_prism1_latents(λ=knob)（r1 曲线 A 同函数）；
    - ``mecha_a``  → exp.v25.mechanism_a 教师口径臂（build_teacher_latents_750 +
      arm_gate_field / apply_gate / energy_matched_gamma，逐式同 eval_mechanism_a_arm，
      仅不做 measure 落盘）；
    - ``student``  → NotImplementedError("mechanism B pending")（双重闸门，防绕过接口层）。
    """

    def __init__(self):
        self._resources: dict | None = None
        self._teacher_ctx: dict | None = None
        self._teacher_sig: str | None = None

    @property
    def resources(self) -> dict | None:
        return self._resources

    def prepare(self, config: TransferConfig, pool: str | None = None,
                device: str = "cuda") -> "CoreRuntime":
        import torch  # noqa: F401  (lazy: 保证 torch.cuda.device_count()==0 环境下 import 不崩)
        from exp.v18 import tasks_and_sources
        from exp.v19.fiber_duality import build_or_load_v19_foundation
        from exp.v20.run_v20 import get_test750_y16_and_d16
        from exp.v20.submode_patchswd import build_or_load_core_knn_bank
        from exp.v22.t4 import build_or_load_t4_latents_750
        from src.utils import load_s10_decoder

        bundle = build_or_load_v19_foundation(str(_pool_path(pool)), device=device)
        bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=device)
        tasks, z_src, _ = tasks_and_sources()
        z_t4, _ = build_or_load_t4_latents_750(bundle, device=device)
        y16, d16 = get_test750_y16_and_d16(bundle)
        dec = load_s10_decoder(device)
        self._resources = {
            "bundle": bundle, "bank": bank, "tasks": tasks, "z_src": z_src.float(),
            "anchor": z_t4, "y16": y16, "d16": d16, "dec": dec, "device": device,
        }
        return self

    # ------------------------------------------------------------ 入口 ----
    def run(self, *, engine: str, interface: str, strength: float, knob: str | None,
            knob_value: float, seed: int, config: TransferConfig,
            resources: dict | None = None, content=None, style=None) -> dict:
        if interface != "core":
            raise ValueError("CoreRuntime 只接受 core 接口")
        R = resources if resources is not None else self._resources
        if R is None:
            raise RuntimeError("CoreRuntime 未 prepare；先调用 prepare(config, device=...)")
        if engine == "student":
            raise NotImplementedError(STUDENT_PENDING_MESSAGE)
        z_src, tasks = self._align_protocol(content, style, R, engine)
        extras: dict[str, Any] = {}
        with torch.no_grad():
            if engine == "d6_flow":
                z, tau_stats = self._flow_branch(config, R, seed, solver=dict(config.solver or {}),
                                                 tau_override=float(knob_value),
                                                 branches=config.branches, gate=config.gate)
                extras["tau_stats"] = tau_stats
            elif engine == "hybrid":
                from exp.v24.p0_m0_m2 import compute_prism1_latents
                z_p1 = compute_prism1_latents(R["bundle"], tasks, z_src, R["d16"],
                                              lam=float(config.p1_lambda), device=R["device"])
                z_d6, tau_stats = self._flow_branch(config, R, seed, solver=dict(config.solver or {}),
                                                    tau_override=None, branches=config.branches,
                                                    gate=config.gate)
                gamma = float(knob_value)
                z = (1.0 - gamma) * z_p1 + gamma * z_d6
                extras = {"tau_stats": tau_stats, "z_p1": z_p1, "z_d6": z_d6, "gamma": gamma}
            elif engine == "prism1":
                from exp.v24.p0_m0_m2 import compute_prism1_latents
                z = compute_prism1_latents(R["bundle"], tasks, z_src, R["d16"],
                                           lam=float(knob_value), device=R["device"])
            elif engine == "mecha_a":
                z, extras = self._mecha_arm(config, R, knob_value)
            else:
                raise ValueError(f"unknown core engine '{engine}'")
            rgb01 = self._decode(z, R)
        return {"engine": engine, "interface": "core", "knob": knob,
                "knob_value": float(knob_value), "strength": float(strength),
                "seed": int(seed), "z": z, "rgb01": rgb01, **extras}

    # ------------------------------------------------------------ 内部 ----
    @staticmethod
    def _align_protocol(content, style, R: dict, engine: str):
        """协议批对齐：content/style=None → 协议 750 任务；显式覆盖路径 pending（如实报错）。"""
        import torch
        tasks = R["tasks"]
        z_src = R["z_src"]
        if content is None:
            z = z_src
        elif torch.is_tensor(content):
            if content.dim() == 3:
                content = content.unsqueeze(0)
            if content.dim() != 4 or tuple(content.shape[1:]) != (4, 64, 64):
                raise ValueError(
                    f"content 须为 (N,4,64,64) 缩放 latent，got {tuple(content.shape)}")
            if int(content.shape[0]) != len(tasks):
                raise NotImplementedError(
                    f"单样本/子集执行 pending：当前可复用路径（run_core_flow / "
                    f"compute_prism1_latents）为 test750 协议批形态，仅支持 N={len(tasks)}；"
                    f"capability 见 ENGINE_REGISTRY['{engine}'].capabilities")
            z = content.float()
        else:
            raise TypeError(f"content 须为 None 或 (N,4,64,64) 张量，got {type(content)!r}")
        if style is not None:
            raise NotImplementedError(
                "显式 style 覆盖 pending：协议批条件下目标画派来自 tasks_and_sources()；"
                "单样本 style 条件路径待总控集成（ENGINE_REGISTRY capabilities 如实标注）")
        return z, tasks

    def _flow_branch(self, config: TransferConfig, R: dict, seed: int, *, solver: dict,
                     tau_override: float | None, branches: str | None, gate: str | None):
        """run_core_flow 薄封装（solver 仅替换 strength 旋钮字段，其余逐项沿用冻结配置）。"""
        import torch
        from exp.v25.r1_inference_ablation import run_core_flow
        if config.d6_ckpt is None or branches is None or gate is None or solver is None:
            raise ValueError(f"engine '{config.engine}' 配置缺少 d6 分支冻结参数")
        sv = dict(solver)
        if tau_override is not None:
            if config.engine == "d6_flow":
                if "tau_min" not in sv:
                    raise ValueError("d6_flow 的 solver 冻结 dict 缺少 'tau_min' 旋钮字段")
                sv["tau_min"] = float(tau_override)
            else:
                raise ValueError("tau_override 仅适用于 d6_flow 引擎")
        eps = torch.randn(R["z_src"].shape, generator=torch.Generator().manual_seed(int(seed)))
        ckpt = REPO_ROOT / config.d6_ckpt
        return run_core_flow(R["bundle"], R["bank"], ckpt, R["tasks"], R["z_src"], R["anchor"],
                             R["y16"], R["d16"], eps, branches=branches, gate_mode=gate,
                             fixed_g=float(config.gate_fixed_g), solver=sv, device=R["device"])

    def _mecha_arm(self, config: TransferConfig, R: dict, knob_value: float):
        """机制 A 教师口径臂薄封装（逐式同 exp.v25.mechanism_a.eval_mechanism_a_arm，无 measure）。"""
        import json as _json

        import torch
        from exp.v25.mechanism_a import (
            apply_gate,
            arm_gate_field,
            as_teacher_spec,
            build_teacher_latents_750,
            energy_matched_gamma,
        )
        spec = as_teacher_spec(config.teacher_spec)
        sig = _json.dumps(spec.to_dict(), sort_keys=True)
        if self._teacher_ctx is None or self._teacher_sig != sig:
            self._teacher_ctx = build_teacher_latents_750(R["bundle"], R["bank"], R["device"], spec=spec)
            self._teacher_sig = sig
        ctx = self._teacher_ctx
        z_p1, z_d6 = ctx["z_p1"], ctx["z_d6"]
        dz = (z_d6 - z_p1).float()
        out = torch.zeros_like(z_p1)
        for st in range(0, len(z_p1), 50):
            sl = slice(st, min(st + 50, len(z_p1)))
            c64 = ctx["c64"][sl].to(R["device"])
            chord = ctx["chords"][sl].to(R["device"])
            M16 = arm_gate_field(config.mecha_arm, c64, dz[sl].to(R["device"]), chord,
                                 net=None, kappa=config.mecha_kappa)
            g = (energy_matched_gamma(M16, dz[sl].to(R["device"]), float(knob_value))
                 if config.mecha_energy_match else float(knob_value))
            z_out = apply_gate(z_p1[sl].to(R["device"]), dz[sl].to(R["device"]), M16, g)
            m_same = torch.as_tensor(ctx["same"][sl]).to(R["device"])
            if m_same.any():
                z_out[m_same] = ctx["z_src"][sl].to(R["device"])[m_same]
            out[sl] = z_out.cpu()
        extras = {"teacher_spec": spec.to_dict(), "mecha_arm": config.mecha_arm,
                  "mecha_kappa": config.mecha_kappa, "mecha_energy_match": config.mecha_energy_match}
        return out, extras

    @staticmethod
    def _decode(z, R: dict):
        """S10 契约解码（decode_s10 内部 /0.18215 + 自带 post_quant_conv 恰好一次）。"""
        from src.utils import decode_s10
        return decode_s10(R["dec"], z.to(R["device"]), batch=16).clamp(0.0, 1.0)


class RefRuntime:
    """PRISMRef 生产执行体：薄封装 O3 One-Shot 推理（exp.v25.r3_oneshot_minimal /
    exp.v24.p0_o1_o3_value.eval_o_tau_list 的组件装配与 10 步 RK2+wj 积分环逐式同构）。

    参考条件 = SG-5 画派原型（协议口径，与已发表 O3 读数同构）；任意单参考图 RGB
    编码路径 pending（R5 参考替换一致性 / 机制 C 集成后开放）。
    """

    def __init__(self):
        self._resources: dict | None = None

    @property
    def resources(self) -> dict | None:
        return self._resources

    def prepare(self, config: RefConfig, pool: str | None = None,
                device: str = "cuda") -> "RefRuntime":
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        from exp.v18 import tasks_and_sources
        from exp.v19.fiber_duality import (
            StyleFiberEncoder,
            build_or_load_v19_foundation,
            extract_color_fiber_27d,
            extract_dino_rgb_moments_768d,
        )
        from exp.v20.slf_flow import coarse_y16_lcn
        from exp.v20.submode_patchswd import load_convnext_tiny_offline
        from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot, _sg5_cnx_v16, _sg5_ref_d16
        from exp.v22.ref_adapter import RefAdapter
        from src.utils import FeatureStack, load_s10_decoder

        bundle = build_or_load_v19_foundation(str(_pool_path(pool)), device=device)
        c21 = torch.load(REPO_ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                         map_location="cpu", weights_only=False)
        cnx = load_convnext_tiny_offline(device).eval()
        for p in cnx.parameters():
            p.requires_grad_(False)

        ck = torch.load(REPO_ROOT / config.ckpt, map_location="cpu", weights_only=False)
        v_dim = 384 if config.v_type == "dino" else 192
        fm = SLFUNetV24OneShot(v_dim=v_dim, use_conf_gate=config.use_conf_gate,
                               use_global_fib=config.use_global_fib).to(device).eval()
        fm.load_state_dict(ck["fm"], strict=True)
        ad = RefAdapter(dim=128).to(device).eval()
        ad.load_state_dict(ck["adapter"], strict=True)
        fib_mean, fib_std = ck["fib_mean"].to(device), ck["fib_std"].to(device)

        fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
        fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
        u_src_750 = fiber_enc.encode(bundle["fnd"]["test750_fib"].to(device).float()).cpu()

        d16_sg5 = _sg5_ref_d16(bundle, c21, device=device)
        v16_sg5 = d16_sg5 if config.v_type == "dino" else _sg5_cnx_v16(c21, cnx, device=device)

        if config.use_global_fib:
            stack = FeatureStack(device)
            dino_model = stack.dino
            clean_dir = REPO_ROOT / "datasets/distinct5_512/train_512clean"
            sg_fibs = []
            for st in c21["sg5"]["stems"]:
                p = clean_dir / st.split("__")[0] / f"{st}.jpg"
                if not p.exists():
                    p = list(clean_dir.rglob(f"{st}.*"))[0]
                xr = TF.to_tensor(Image.open(p).convert("RGB").resize(
                    (512, 512), Image.Resampling.BILINEAR)).unsqueeze(0).to(device)
                sg_fibs.append(torch.cat([
                    extract_color_fiber_27d(xr),
                    extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=device).to(device),
                ], dim=-1))
            fib_sg5 = torch.cat(sg_fibs, 0)
            fib_n_sg5 = (fib_sg5 - fib_mean) / fib_std
            u_sg5 = fiber_enc.encode(fib_sg5.float()).cpu()
            del stack
            torch.cuda.empty_cache()
        else:
            u_sg5 = c21["sg5"]["u"].float()
            fib_n_sg5 = None
        del fiber_enc

        w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
        b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
        tasks, z_src, _ = tasks_and_sources()
        y16_750 = coarse_y16_lcn(((z_src.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
        dec = load_s10_decoder(device)
        self._resources = {
            "fm": fm, "ad": ad, "u_sg5": u_sg5, "fib_n_sg5": fib_n_sg5, "u_src": u_src_750,
            "d16_sg5": d16_sg5, "v16_sg5": v16_sg5, "y16": y16_750, "tasks": tasks,
            "z_src": z_src.float(), "d16": bundle["test_struct"]["dino_16"],
            "dec": dec, "device": device,
        }
        return self

    # ------------------------------------------------------------ 入口 ----
    def run(self, *, engine: str, interface: str, strength: float, knob: str | None,
            knob_value: float, seed: int, config: RefConfig,
            resources: dict | None = None, content=None, reference=None) -> dict:
        if interface != "ref":
            raise ValueError("RefRuntime 只接受 ref 接口")
        if engine != "o3_ref":
            raise ValueError(f"unknown ref engine '{engine}'")
        R = resources if resources is not None else self._resources
        if R is None:
            raise RuntimeError("RefRuntime 未 prepare；先调用 prepare(config, device=...)")
        import torch
        from src.utils import decode_s10
        from src.utils.runtime import bf16_autocast as autocast

        tasks = R["tasks"]
        z_src = R["z_src"]
        if content is None:
            zs = z_src
            sel = slice(0, len(tasks))
            sid_t = torch.tensor([t["source_style_id"] for t in tasks])
            tid_t = self._resolve_tids(reference, tasks)
        else:
            if not torch.is_tensor(content) or content.dim() != 4 or tuple(content.shape[1:]) != (4, 64, 64):
                raise ValueError(f"content 须为 None 或 (N,4,64,64) 缩放 latent，got {content!r}")
            if int(content.shape[0]) != len(tasks):
                raise NotImplementedError(
                    f"单样本/子集执行 pending：O3 可复用路径为 test750 协议批形态，仅支持 N={len(tasks)}")
            zs = content.float()
            sel = slice(0, len(tasks))
            sid_t = torch.tensor([t["source_style_id"] for t in tasks])
            tid_t = self._resolve_tids(reference, tasks)

        tau0 = float(knob_value)
        if knob != "tau0":
            raise ValueError(f"o3_ref 旋钮应为 'tau0'，got {knob!r}")
        wj = float(config.wj)
        steps = int(config.steps)
        device = R["device"]
        same = sid_t == tid_t
        B = len(tasks)

        eps = torch.randn(zs.shape, generator=torch.Generator().manual_seed(int(seed)))
        out = torch.zeros_like(zs)
        with torch.no_grad():
            for st in range(0, B, 25):
                sl = slice(st, min(st + 25, B))
                b = sl.stop - sl.start
                zb = zs[sl].to(device)
                tid_d = tid_t[sl].to(device)
                u_style = R["ad"](R["u_sg5"][tid_t[sl].cpu()].to(device), R["u_src"][sl].to(device))
                u_glob = (R["fib_n_sg5"][tid_t[sl].cpu()].to(device)
                          if R["fib_n_sg5"] is not None else u_style)
                d16_r = R["d16_sg5"][tid_t[sl].cpu()].to(device)
                v16_r = R["v16_sg5"][tid_t[sl].cpu()].to(device)
                y16 = R["y16"][sel][sl].to(device).float()
                d16 = R["d16"][sel][sl].to(device).float()
                z = (1.0 - tau0) * eps[sl].to(device) + tau0 * zb

                f0 = torch.zeros(b, dtype=torch.bool, device=device)
                f1 = torch.ones(b, dtype=torch.bool, device=device)
                l1 = torch.ones(b, device=device)
                sid0 = torch.zeros(b, dtype=torch.long, device=device)
                dt = (1.0 - tau0) / float(steps)

                def _v(zz, pr):
                    tv = torch.full((b,), tau0 + pr * (1.0 - tau0), device=device)
                    with autocast(device):
                        v_sc, v_0c = R["fm"](
                            torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                            sid0.repeat(2), u_style.repeat(2, 1), tv.repeat(2),
                            drop_style=torch.cat([f0, f1], 0),
                            drop_content=torch.cat([f0, f0], 0),
                            layout_scale=torch.cat([l1, l1], 0),
                            d16_ref=d16_r.repeat(2, 1, 1, 1), v16_ref=v16_r.repeat(2, 1, 1, 1),
                            u_glob=u_glob.repeat(2, 1)).float().chunk(2, 0)
                    return v_0c + wj * (v_sc - v_0c)

                for i in range(steps):
                    v1 = _v(z, i / float(steps))
                    v2 = _v(z + 0.5 * dt * v1, (i + 0.5) / float(steps))
                    z = z + dt * v2
                if config.same_mask:
                    m_same = same[sl].to(device)
                    if m_same.any():
                        z = torch.where(m_same.view(-1, 1, 1, 1), zb, z)
                out[sl] = z.cpu()

            rgb01 = decode_s10(R["dec"], out.to(device), batch=16).clamp(0.0, 1.0)
        return {"engine": engine, "interface": "ref", "knob": knob,
                "knob_value": tau0, "strength": float(strength), "seed": int(seed),
                "z": out, "rgb01": rgb01, "wj": wj, "steps": steps}

    # ------------------------------------------------------------ 内部 ----
    @staticmethod
    def _resolve_tids(reference, tasks: list[dict]) -> "object":
        """reference → 目标画派原型索引（None=协议目标；int/(B,) 张量=显式覆盖）。"""
        import torch
        default = torch.tensor([t["target_style_id"] for t in tasks])
        if reference is None:
            return default
        if isinstance(reference, int):
            return torch.full((len(tasks),), int(reference), dtype=torch.long)
        if torch.is_tensor(reference) and reference.dim() <= 1:
            ref = reference.long()
            if ref.numel() != len(tasks):
                raise NotImplementedError(
                    f"reference 索引数 {int(ref.numel())} != 协议批 {len(tasks)}；"
                    f"子集执行 pending（capabilities 如实标注）")
            return ref
        raise NotImplementedError(
            "任意单参考图编码路径 pending：当前 O3 可复用路径的参考条件为 SG-5 画派原型 "
            "（协议口径，与已发表读数同构）；R5 参考替换一致性 / 机制 C 集成后开放 "
            "（ENGINE_REGISTRY['o3_ref'].capabilities 如实标注）")
