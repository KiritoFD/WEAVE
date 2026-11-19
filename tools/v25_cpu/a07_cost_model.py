"""A07: 完整 RGB→RGB 性能归因模型（静态可计算 + 代码结构分析，CPU-only）.

任务（docs/19_pull_together/agent2.md §六 A07）::

    T_total = T_encode + T_feature + T_style + T_anchor + T_flow + T_decode + T_IO
    阶段链: Encoder → DINO/ConvNeXt → Style Encoding → Anchor → Flow → S10 → I/O

本模块职责（不跑 GPU、不 ssh、无 CUDA）：

1. **静态结构层**（从代码逐行核实，非估算）——每个引擎的参数量、NFE（velocity
   评估次数）、CFG batch 放大（cs_j=×2 / full=×4）、实际模型前向次数（image-equivalent）。
   参数量优先用 CPU torch 对活体模型类逐参数求和复算，失败退回文档常量。
2. **实测锚点层**——a04_strength_curves.json（引擎描述）、docs/performance/MASTER_LEDGER.md、
   b04_io_report.md（I/O 绝对秒数）、runs/v24/o3_sg25_eval.json 与
   runs/v25/r3_oneshot_minimal.json 的 e2e_ms/peak_vram。每个数值带出处字段。
3. **三档口径**——cold 首次加载 / warm 热缓存稳态 / 同参考复用特征缓存；每档每引擎
   公式化估算（实测值标 measured，无实测标 estimate 并给依据链）。
4. **1.25 s/img 反推**（exp/v24/o3_sg25_eval.py L154-197 计时段）与
   "20 NFE ≠ 端到端速度" 的量化差异来源。
5. **B04 对接**——T_IO 项直接消费 b04_io_report.md 的绝对秒数。
6. **最小 GPU 实测清单**——哪些估算必须被 GPU 实测替换才能进论文。

运行::

    python -m tools.v25_cpu.a07_cost_model

约束：全程 ``CUDA_VISIBLE_DEVICES=""``、``OMP/MKL/OPENBLAS_NUM_THREADS=2``；只读仓库
内 JSON/文档常量，不写任何文件；所有估算项带 ``est`` 标记与依据链，实测项带出处。
"""
from __future__ import annotations

import os

# CPU 隔离（agent2.md §十一）：必须在任何 torch import 之前设置。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 0) 参数量：live 复算（CPU torch 对模型类逐参数求和）+ 文档常量回退
# --------------------------------------------------------------------------- #
#: 回退常量（live 复算失败时使用；live 成功时以 live 为准并在输出中标注）。
FALLBACK_PARAMS_M: dict[str, float] = {
    "prism1_net": 1.280,       # SpatiallyRoutedChordNet(width=96, chord_dim=128, spatial_router)
    "fiber_enc": 0.305,        # StyleFiberEncoder(in_dim=795, dim=128)
    "ref_adapter": 0.066,      # RefAdapter(dim=128)
    "slfunet_d6": 21.95,       # SLFUNet(base_dim=144, u_dim=128)  文档 ~20.8M(旧口径)/21.95M(live)
    "oneshot_o3": 22.23,       # SLFUNetV24OneShot = SLFUNet + RefAttnV24(0.281M)
    "student_b0_total": 1.64,  # P1 基座冻结 + GammaResidualStudent B0(width64,4blk) 0.36M
    "student_b1_total": 5.98,  # P1 基座冻结 + GammaResidualStudent B1(width176,8blk) 4.69M
    "s10_decoder": 10.66,      # S10 剪枝 decoder（含 post_quant_conv；AGENTS.md 钉死）
    "sd0_vae_total": 83.65,    # SD-VAE encoder+decoder 全量（encoder 约 34-37M，见 stage 层 est）
    "dino_v2_s": 21.7,         # DINOv2-small 裁判/特征栈
    "convnext_t": 28.6,        # ConvNeXt-Tiny（O3 只用 cnx[:4] 子集，见 stage 层 est）
    "csd_vit_l": 303.7,        # CSD ViT-L/14（仅评测裁判）
    "clip_b32": 151.3,         # openai/clip-vit-base-patch32 全模型（图像塔约 88M；仅裁判）
    "lpips_alex": 0.6,         # LPIPS(alex)（仅评测裁判）
}


def _live_param_counts() -> tuple[dict[str, float], list[str]]:
    """CPU 活体复算参数量；任何 import/构造失败都降级为回退常量并记录原因。

    live 只覆盖流模型/编码头等可离线构造的类；裁判与冻结 decoder（S10/SD0-VAE 等）
    无轻量构造路径，恒用文档常量。
    """
    out: dict[str, float] = dict(FALLBACK_PARAMS_M)
    notes: list[str] = []
    try:
        import torch  # noqa: PLC0415  (CPU-only)
        from exp.v24.p0_m0_m2 import SpatiallyRoutedChordNet  # noqa: PLC0415
        from exp.v19.fiber_duality import StyleFiberEncoder  # noqa: PLC0415
        from exp.v22.ref_adapter import RefAdapter  # noqa: PLC0415
        from exp.v20.slf_flow import SLFUNet  # noqa: PLC0415
        from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot  # noqa: PLC0415
        from exp.v25.mechanism_b import B_SCALES, GammaResidualStudent  # noqa: PLC0415

        def _n(m) -> float:
            return sum(p.numel() for p in m.parameters()) / 1e6

        out["prism1_net"] = _n(SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True))
        out["fiber_enc"] = _n(StyleFiberEncoder(in_dim=795, dim=128))
        out["ref_adapter"] = _n(RefAdapter(dim=128))
        out["slfunet_d6"] = _n(SLFUNet(base_dim=144, u_dim=128))
        m2 = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True)
        out["oneshot_o3"] = _n(m2)
        out["refattn_head"] = _n(m2.ref_attn16)
        b0 = GammaResidualStudent(**{k: v for k, v in B_SCALES["B0"].items() if k != "updates"})
        b1 = GammaResidualStudent(**{k: v for k, v in B_SCALES["B1"].items() if k != "updates"})
        out["student_b0_residual"] = _n(b0)
        out["student_b1_residual"] = _n(b1)
        out["student_b0_total"] = out["prism1_net"] + out["student_b0_residual"]
        out["student_b1_total"] = out["prism1_net"] + out["student_b1_residual"]
        del torch
        notes.append("params=live(CPU torch 逐参数求和)")
    except Exception as exc:  # noqa: BLE001 —— 降级路径必须有据
        notes.append(f"params=fallback 文档常量（live 复算失败: {type(exc).__name__}: {exc}）")
        out = dict(FALLBACK_PARAMS_M)
    return out, notes  # noqa: RET504


# --------------------------------------------------------------------------- #
# 1) 实测锚点（每个数值带出处；不带出处的数字一律不许出现在本文件）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Measured:
    """一条实测锚点：value + 单位 + 出处（repo 相对路径/文档名）+ 口径说明。"""

    value: float
    unit: str
    source: str
    note: str = ""


MEASURED: dict[str, Measured] = {
    # ---- PRISM-1（纯单步前馈）----
    "p1_ms_4090": Measured(17.0, "ms/img", "docs/performance/MASTER_LEDGER.md (V19_S2)",
                           "58.7 img/s @4090; 协议批形态，latent/特征全缓存"),
    "p1_ms_3060": Measured(29.0, "ms/img", "AGENTS.md 环境备忘 (~29 ms @3060)",
                           "同 checkpoint 跨卡换算，记录值"),
    # ---- V20 D6 家族（8 步 RK2 SLFUNet 流）----
    "v20_d1_ms": Measured(134.1, "ms/img", "docs/performance/MASTER_LEDGER.md (V20.D1_core_slf_hiS)",
                          "V20 D1..D7 区间 76.6-134.1 ms/img；3060 12GB，训练峰值 4.06GB"),
    "v20_d7_ms": Measured(76.6, "ms/img", "AGENTS.md (V20_D1..D7 区间下端)", ""),
    "v21_1shot_ms": Measured(135.4, "ms/img", "docs/performance/MASTER_LEDGER.md (V21.1Shot_SG_exact_hiS)",
                             "对 StyleGallery 30,400 ms 提速 224x"),
    "stylegallery_ms": Measured(30400.0, "ms/img", "docs/performance/MASTER_LEDGER.md", "商业基线"),
    # ---- O3 One-Shot（SG-25 125-pair 单样本计时段）----
    "o3_t40_ms": Measured(1251.6, "ms", "runs/v24/o3_sg25_eval.json sg25[O3_sg25_t40].e2e_ms",
                          "3060 batch=1，torch.cuda.synchronize 计时段=生成段（编码→流→S10 解码）"),
    "o3_t50_ms": Measured(1245.4, "ms", "runs/v24/o3_sg25_eval.json sg25[O3_sg25_t50].e2e_ms", ""),
    "o3_vram_mb": Measured(2111.8, "MB", "runs/v24/o3_sg25_eval.json sg25[*].peak_vram_mb",
                           "torch.cuda.max_memory_allocated，batch=1"),
    "o3lf_t40_ms": Measured(1672.4, "ms", "runs/v25/r3_oneshot_minimal.json label_free_blind.sg25[0].e2e_ms",
                            "O3_label_free（同架构、不同学生权重）；docs/v25/GPU3060_RESULTS.md 记 1.53-1.67 s/img"),
    "o3lf_t50_ms": Measured(1529.5, "ms", "runs/v25/r3_oneshot_minimal.json label_free_blind.sg25[1].e2e_ms", ""),
    "o3lf_vram_mb": Measured(2135.0, "MB", "runs/v25/r3_oneshot_minimal.json label_free_blind.sg25[*].peak_vram_mb", ""),
    # ---- a04 曲线（质量侧实测臂，供引擎族口径对齐；无延迟字段）----
    "a04_arms": Measured(19.0, "arms", "docs/v25/audits/a04_strength_curves.json inputs.n_arms",
                         "families: prism1_lambda / d6_tau_min / hybrid_gamma / o3_tau0；JSON 无 NFE/延迟字段"),
    # ---- B04 I/O 绝对秒数（Windows 本机实测，warm 中位）----
    "io_infer_save_s": Measured(1.26, "s/臂", "docs/v25/audits/b04_io_report.md §8",
                                "原始 batch torch.save ~738MB/臂（188 个 .pt；src/infer/run.py L75-77）"),
    "io_measure_load_s": Measured(0.4, "s/臂", "docs/v25/audits/b04_io_report.md §8",
                                  "measure 读回 738MB（src/eval/measure.py L62）"),
    "io_eval_json_s": Measured(0.023, "s/臂", "docs/v25/audits/b04_io_report.md §8",
                               "eval.json ~0.9MB dumps+写盘（measure.py L109-112）"),
    "io_features_s": Measured(0.01, "s/臂", "docs/v25/audits/b04_io_report.md §8", "features.pt ~5.9MB"),
    "io_png750_s": Measured(3.23, "s/臂", "docs/v25/audits/b04_io_report.md §2/§8",
                            "750 张 l1+8 线程，噪声上界；仅 write_png=True（runner.run_measure_subprocess 默认 False）"),
    "io_sha30mb_s": Measured(0.015, "s/臂", "docs/v25/audits/b04_io_report.md §3/§8",
                             "30MB checkpoint sha256，仅 B03 hash 复核开启"),
    "io_register_s": Measured(0.0392, "s/次", "docs/v25/audits/b04_io_report.md §4",
                              "registry register 含 git rev-parse；设 WEAVE_GIT_SHA 后 1.814ms"),
    "io_arm_wall_s": Measured(50.0, "s/臂", "docs/v25/audits/b04_io_report.md §8（总控给定基线）",
                              "单臂 750 任务评测墙钟；CPU/I/O 占 9.8%(PNG 开)/3.4%(PNG 关)"),
    # ---- b04 M7 batch 载荷往返 ----
    "io_torch_load_354mb_ms": Measured(1.926, "ms", "docs/v25/audits/b04_io_report.md §7",
                                       "3.54MB batch=4 载荷 torch.load @585-1838 MB/s"),
}


# --------------------------------------------------------------------------- #
# 2) 静态结构层：引擎注册（逐项由代码行核实）
# --------------------------------------------------------------------------- #
@dataclass
class Engine:
    name: str
    interface: str                       # core | ref | pending
    params_keys: tuple                   # FALLBACK_PARAMS_M / live keys
    nfe_vel_evals: int                   # velocity 评估次数（NFE 语义=流模型评估次数）
    cfg_mult: int                        # 每次 velocity 评估的 batch 放大（cs_j=2, full=4, 无CFG=1）
    flow_img_equiv: int                  # = nfe_vel_evals * cfg_mult（image-equivalent 前向）
    fresh_nonflow: dict                  # fresh RGB→RGB 每图非流前向（encode/feature/style/anchor/decode）
    protocol_nonflow: dict               # 协议批缓存路径每图非流前向（特征/锚/教师 latent 全缓存）
    protocol_flow: int | None = None     # 协议批路径流前向（None=fresh 同款; hybrid/mecha_a=0 缓存混合）
    measured_keys: tuple = ()            # MEASURED 键
    code_basis: str = ""                 # 结构依据（文件+行）
    notes: str = ""

    @property
    def params_m(self) -> float:
        return round(sum(P[_k] for _k in self.params_keys), 3)

    @property
    def forwards_per_img_fresh(self) -> int:
        """fresh RGB→RGB 单图全部 NN 前向（image-equivalent；一次性锚管线另见 notes）。"""
        return self.flow_img_equiv + sum(self.fresh_nonflow.values())

    @property
    def forwards_per_img_protocol(self) -> int:
        """协议批（test750 臂, 缓存齐备）每图 NN 前向。"""
        pf = self.flow_img_equiv if self.protocol_flow is None else self.protocol_flow
        return pf + sum(self.protocol_nonflow.values())


# fresh 阶段前向（单图口径）：
#   encode = SD0 VAE 编码器; feature = DINO/ConvNeXt 提取; style = fiber/adapter/bank;
#   anchor = 锚 latent 构造（chroma-shift/T4/τ0 桥, 张量算子或一次性管线）; decode = S10。
O3_FRESH = {"encode": 1, "feature": 4, "style": 1, "anchor": 0, "decode": 1}   # DINO x4=whitened(cx,ref)+moments(cx,ref)
O3_PROTO = {"encode": 0, "feature": 0, "style": 1, "anchor": 0, "decode": 1}   # 协议: z_src/d16/fib/ref 原型全缓存, adapter/fiber tiny 仍跑
CORE_FRESH = {"encode": 1, "feature": 1, "style": 0, "anchor": 0, "decode": 1}  # d16 DINO x1 + VAE enc; y16=latent LCN 算子
CORE_PROTO = {"encode": 0, "feature": 0, "style": 0, "anchor": 0, "decode": 1}  # 协议 style=core8 bank 查表(einsum, 无 NN 前向)
P1_FRESH = {"encode": 1, "feature": 2, "style": 1, "anchor": 0, "decode": 1}    # d16 DINO + 795d fiber moments DINO
P1_PROTO = {"encode": 0, "feature": 0, "style": 1, "anchor": 0, "decode": 1}    # 协议 fiber_enc tiny 前向仍在

ENGINES: list[Engine] = [
    Engine(
        name="prism1", interface="core", params_keys=("prism1_net",),
        nfe_vel_evals=1, cfg_mult=1, flow_img_equiv=1,
        fresh_nonflow=dict(P1_FRESH), protocol_nonflow=dict(P1_PROTO),
        measured_keys=("p1_ms_4090", "p1_ms_3060"),
        code_basis="exp/v24/p0_m0_m2.py L31-63 compute_prism1_latents（单步 stu(...) 前向 + chroma-shift 锚, "
                   "无 CFG 无迭代）; src/v25/config.py L161-167 (knob=lambda, steps=1)",
        notes="style=fiber_enc.encode(0.305M) tiny; anchor=latent chroma-shift alpha=0.20*lambda 张量算子",
    ),
    Engine(
        name="d6_flow", interface="core", params_keys=("slfunet_d6",),
        nfe_vel_evals=16, cfg_mult=2, flow_img_equiv=32,
        fresh_nonflow=dict(CORE_FRESH), protocol_nonflow=dict(CORE_PROTO),
        measured_keys=("v20_d1_ms", "v20_d7_ms"),
        code_basis="exp/v22/core.py L28-32 D6_SOLVER(steps=8) / L100-101 D6H_SOLVER(steps=8) / L185-199 "
                   "rk2_integrate（每步 2 次 vel_fn） / L136-151 cs_j 分支 torch.cat([z,z],0) = x2 batch; "
                   "src/v25/config.py L168-174 (d6_flow 默认 D6H+cs_j)",
        notes="agent2.md 任务的 'd6_flow(cs_j 20NFE)' 与代码不符: 实际 16 vel evals（8 步 x RK2 x2）; "
              "'20 NFE' 恰是 o3_ref 的 10 步 x 2; 32 image-equivalent 才是真实张量工作量; "
              "fresh 锚=T4 管线(P1 前向+canonicalizer)为一次性批量构建, 不入每图前向",
    ),
    Engine(
        name="hybrid", interface="core", params_keys=("slfunet_d6", "prism1_net"),
        nfe_vel_evals=16, cfg_mult=4, flow_img_equiv=65,
        fresh_nonflow=dict(CORE_FRESH), protocol_nonflow={"encode": 0, "feature": 0, "style": 0, "anchor": 0, "decode": 1},
        protocol_flow=0,
        measured_keys=(),
        code_basis="exp/v22/core.py L153-180 full 分支 z_cur.repeat(4,...) = x4 batch; "
                   "exp/v25/r1_inference_ablation.py L156-158 C_hyb_*: z=(1-gamma)*z_p1_100+gamma*z_d6 "
                   "(V24 基座缓存口径, 协议评测仅 decode); fresh 路径 = P1(1) + D6 full(16x4=64) = 65 流前向",
        notes="无直接延迟实测臂; eval 缓存路径 1 前向/图(decode), fresh 68 前向/图",
    ),
    Engine(
        name="mecha_a", interface="core", params_keys=("slfunet_d6", "prism1_net"),
        nfe_vel_evals=16, cfg_mult=2, flow_img_equiv=33,
        fresh_nonflow=dict(CORE_FRESH), protocol_nonflow={"encode": 0, "feature": 0, "style": 0, "anchor": 0, "decode": 1},
        protocol_flow=0,
        measured_keys=(),
        code_basis="exp/v25/mechanism_a.py L560-612 eval_mechanism_a_arm（教师三元组 D6H+cs_j 的 z_p1/z_d6 "
                   "缓存 + apply_gate 张量算子; A0/A1 门控 0 NN 前向, A2/A3 门控网 tiny）; "
                   "teacher_spec=TEACHER_SPEC_CANONICAL (src/v25/config.py L105-113)",
        notes="教师口径 fresh 流前向 = D6 cs_j(32) + P1(1) = 33; 协议评测仅 decode(教师 latent 缓存再生)",
    ),
    Engine(
        name="o3_ref", interface="ref", params_keys=("oneshot_o3", "ref_adapter", "fiber_enc"),
        nfe_vel_evals=20, cfg_mult=2, flow_img_equiv=40,
        fresh_nonflow=dict(O3_FRESH), protocol_nonflow=dict(O3_PROTO),
        measured_keys=("o3_t40_ms", "o3_t50_ms", "o3_vram_mb", "o3lf_t40_ms", "o3lf_t50_ms", "o3lf_vram_mb"),
        code_basis="exp/v24/o3_sg25_eval.py L154-197 计时段: L159 VAE enc / L161-162 whitened dino x2 / "
                   "L163 cnx(ref) / L164-165 moments dino x2 / L166 adapter+fiber(tiny) / L172-184 10 步 RK2 "
                   "x2 evals, 每个 eval torch.cat([zz,zz],0)=x2 batch / L185 S10 decode; "
                   "src/v25/config.py L198-205, L374 (steps=10, wj=2.2)",
        notes="唯一 fresh RGB→RGB 引擎; batch=1 单样本计时段实测 1.25 s/img; SG-25 输出另付未计时 PNG+裁判",
    ),
    Engine(
        name="student", interface="pending", params_keys=("student_b0_total",),
        nfe_vel_evals=1, cfg_mult=1, flow_img_equiv=1,
        fresh_nonflow=dict(P1_FRESH), protocol_nonflow=dict(P1_PROTO),
        measured_keys=(),
        code_basis="exp/v25/mechanism_b.py L165-169 B_SCALES(B0 width64x4blk / B1 width176x8blk), "
                   "L107-142 GammaResidualStudent 单步残差; checkpoint 未产出 (src/v25/config.py L191-195 pending)",
        notes="B0 残差 0.36M + P1 基座 1.28M = 1.64M; B1 = 5.98M; 前向结构与 prism1 同（单步）",
    ),
]


# --------------------------------------------------------------------------- #
# 3) 阶段单价系数（@3060 batch=1, autocast fp16；除标注外均为 estimate + 依据链）
# --------------------------------------------------------------------------- #
# 每项: (low_ms, high_ms, kind, 依据链)
STAGE_UNIT_COSTS: dict[str, tuple[float, float, str, str]] = {
    "vae_enc_512": (30.0, 60.0, "est",
                    "SD0 AutoencoderKL encoder(全量 83.65M 的 encoder 半壁, est 34-37M) 在 512^2 下卷积栈; "
                    "3060 fp16; 无单测, 以同类 SD-VAE 512^2 batch1 经验区间外推"),
    "dino_s_512": (15.0, 25.0, "est",
                   "DINOv2-small 21.7M @512^2(518 patch 32x32) batch1; V14/measure 裁判同型; 无单测"),
    "cnx_s2_256": (8.0, 15.0, "est",
                   "ConvNeXt-T cnx[:4] 子集(stem+前段 stage, est 5-9M/28.6M) @256^2; o3_sg25_eval L71-74"),
    "unet_oneshot_64_b1": (24.0, 30.0, "derived",
                           "由 o3 计时段反解: (1245.4-215ms 固定栈上限) / 40 image-eq = 25.8ms ~ "
                           "(1251.6-128ms 固定栈下限) / 40 = 28.1ms; batch=1 启动开销主导"),
    "unet_oneshot_64_b1_lf": (33.0, 42.0, "derived",
                              "label_free 计时段 1529.5-1672.4ms 反解: (1529.5-215)/40=32.9ms ~ (1672.4-128)/40=38.6ms; "
                              "同架构权重不同即差 ~7-15ms/forward, 列入 must-measure"),
    "unet_d6_img_eq_batched": (1.6, 2.6, "derived",
                               "V20 D1 134.1 ms/img / 64 image-eq(full x4) = 2.1ms 中值; 区间由 D1..D7 "
                               "76.6-134.1 ms/img 除以 64 得 1.20-2.10ms, 加协议批开销余量"),
    "s10_dec_512_b1": (30.0, 50.0, "est",
                       "S10 剪枝 decoder 10.66M(含 post_quant_conv) 512^2 输出 batch1; 无单测"),
    "tiny_mlp": (0.5, 2.0, "est",
                 "fiber_enc 0.305M + RefAdapter 0.066M + 门控/量化算子; 向量级, 微秒~毫秒"),
}


def stage_share_table_o3() -> list[tuple[str, str, str, str, str]]:
    """o3_ref 计时段 1.25 s/img 的阶段占比反推（estimate 区间，依据链入 STAGE_UNIT_COSTS）。"""
    fixed_low = (STAGE_UNIT_COSTS["vae_enc_512"][0] + 4 * STAGE_UNIT_COSTS["dino_s_512"][0]
                 + STAGE_UNIT_COSTS["cnx_s2_256"][0] + STAGE_UNIT_COSTS["s10_dec_512_b1"][0]
                 + 2 * STAGE_UNIT_COSTS["tiny_mlp"][0])
    fixed_high = (STAGE_UNIT_COSTS["vae_enc_512"][1] + 4 * STAGE_UNIT_COSTS["dino_s_512"][1]
                  + STAGE_UNIT_COSTS["cnx_s2_256"][1] + STAGE_UNIT_COSTS["s10_dec_512_b1"][1]
                  + 2 * STAGE_UNIT_COSTS["tiny_mlp"][1])
    flow_low = MEASURED["o3_t50_ms"].value - fixed_high      # 1245.4 - 215 = 1030.4
    flow_high = MEASURED["o3_t40_ms"].value - fixed_low      # 1251.6 - 128 = 1123.6
    t_lo, t_hi = MEASURED["o3_t50_ms"].value, MEASURED["o3_t40_ms"].value

    def pct(x, t):  # 占计时段比例（保守取同端）
        return 100.0 * x / t

    rows = [
        ("T_flow", f"{flow_low:.0f}-{flow_high:.0f} ms",
         f"{pct(flow_low, t_hi):.1f}-{pct(flow_high, t_lo):.1f}%",
         "derived", f"40 img-eq x (24-28.1 ms/forward, batch1 启动开销主导); NFE=20, CFG x2"),
        ("T_decode (S10)", f"{STAGE_UNIT_COSTS['s10_dec_512_b1'][0]:.0f}-{STAGE_UNIT_COSTS['s10_dec_512_b1'][1]:.0f} ms",
         f"{pct(STAGE_UNIT_COSTS['s10_dec_512_b1'][0], t_hi):.1f}-{pct(STAGE_UNIT_COSTS['s10_dec_512_b1'][1], t_lo):.1f}%",
         "est", "10.66M decoder, 512^2 batch1"),
        ("T_encode (SD0 VAE enc)", f"{STAGE_UNIT_COSTS['vae_enc_512'][0]:.0f}-{STAGE_UNIT_COSTS['vae_enc_512'][1]:.0f} ms",
         f"{pct(STAGE_UNIT_COSTS['vae_enc_512'][0], t_hi):.1f}-{pct(STAGE_UNIT_COSTS['vae_enc_512'][1], t_lo):.1f}%",
         "est", "o3_sg25_eval L159; 仅 fresh RGB 路径存在"),
        ("T_feature (4x DINO-S + 1x CNX)", f"{4 * STAGE_UNIT_COSTS['dino_s_512'][0] + STAGE_UNIT_COSTS['cnx_s2_256'][0]:.0f}"
         f"-{4 * STAGE_UNIT_COSTS['dino_s_512'][1] + STAGE_UNIT_COSTS['cnx_s2_256'][1]:.0f} ms",
         f"{pct(4 * STAGE_UNIT_COSTS['dino_s_512'][0] + STAGE_UNIT_COSTS['cnx_s2_256'][0], t_hi):.1f}"
         f"-{pct(4 * STAGE_UNIT_COSTS['dino_s_512'][1] + STAGE_UNIT_COSTS['cnx_s2_256'][1], t_lo):.1f}%",
         "est", "L161/162 whitened x2 + L164/165 moments x2 (DINO) + L163 cnx(ref); 其中 ref 侧 2x DINO + 1x CNX 可缓存"),
        ("T_style (fiber+adapter)", "1-4 ms", "0.1-0.3%", "est", "0.305M+0.066M tiny MLP on 795d"),
        ("T_anchor (tau0 bridge)", "<1 ms", "~0%", "est", "z=(1-tau0)*eps+tau0*z_src 张量算子 (L169)"),
    ]
    return rows, fixed_low, fixed_high


# --------------------------------------------------------------------------- #
# 4) 三档口径：cold / warm / same-ref-cache
# --------------------------------------------------------------------------- #
def tier_model() -> list[dict]:
    """每引擎三档公式化估算；kind: measured / derived / est。"""
    io_per_img_off = 1000.0 * (MEASURED["io_infer_save_s"].value + MEASURED["io_measure_load_s"].value
                               + MEASURED["io_eval_json_s"].value + MEASURED["io_features_s"].value) / 750.0
    io_per_img_on = io_per_img_off + 1000.0 * MEASURED["io_png750_s"].value / 750.0
    rows: list[dict] = []
    for e in ENGINES:
        p1_ms = MEASURED["p1_ms_3060"].value
        d1_ms = MEASURED["v20_d1_ms"].value
        o3_ms = MEASURED["o3_t40_ms"].value
        row: dict = {"engine": e.name, "tiers": {}}
        if e.name == "prism1":
            row["tiers"] = {
                "cold": ("est", "P1 ckpt ~5.1MB(fp32)+bundle(fiber foundation/test_struct/z0 cache) 读盘+反序列化"
                                "+CUDA ctx; b04 M3 hash 吞吐 2.0GB/s 下 5MB 可忽略, 主项是 torch/CUDA 初始化, 无实测"),
                "warm": ("measured", f"17.0 ms/img @4090 (58.7 img/s); ~29 ms/img @3060; "
                                     f"+T_IO {io_per_img_off:.1f}-{io_per_img_on:.1f} ms/img(协议臂摊销)"),
                "refcache": ("measured", "同 warm（协议路径风格弦/锚/特征全缓存, 无参考侧增量）"),
            }
        elif e.name == "d6_flow":
            row["tiers"] = {
                "cold": ("est", "SLFUNet ckpt ~88MB(fp32) + T4 锚 750 latent 一次性构建"
                                "(P1 前向 x750 + canonicalizer) + y16/d16 缓存构建; 一次性, 无实测"),
                "warm": ("derived", f"cs_j=32 img-eq, 为 V20 D1 full 口径(64 img-eq)一半: est 45-80 ms/img"
                                    f"(32x1.2-2.1 + decode 5-10ms + 开销; 单价由 D1 134.1/64 = 2.10ms/img-eq 反推); "
                                    f"实测族锚 76.6-134.1 ms/img(D1..D7, full 口径); "
                                    f"+T_IO {io_per_img_off:.1f}-{io_per_img_on:.1f} ms/img"),
                "refcache": ("measured", "同 warm：core 引擎协议路径本就消费 kNN bank/原型缓存（无 per-ref 前向）"),
            }
        elif e.name == "hybrid":
            row["tiers"] = {
                "cold": ("est", "同 d6_flow + P1 ckpt; 协议评测另需 V24 基座缓存 z_p1_100/z_d6(p0_base_latents_750.pt ~738MB 读入)"),
                "warm": ("est", "eval 侧 1 前向/图(仅 decode, 缓存线性混合); fresh 路径 = 68 img-eq 总前向"
                                "(流 65 = 2x d6_flow(cs_j) 流工作量): est 85-150 ms/img@3060 协议批; 无实测臂"),
                "refcache": ("measured", "同 warm（缓存线性混合, 无参考侧增量）"),
            }
        elif e.name == "mecha_a":
            row["tiers"] = {
                "cold": ("est", "同 d6_flow 教师口径 + 门控网(A2/A3 tiny); 教师 latent 缓存再生一次性"),
                "warm": ("est", "A0/A1: eval 0 流前向（缓存教师差 dz + apply_gate 张量算子）; "
                                "A2/A3 + 门控网 tiny 前向; fresh 教师口径 = 33 img-eq ~ d6_flow 同级"),
                "refcache": ("measured", "同 warm"),
            }
        elif e.name == "o3_ref":
            rf_save = 2 * STAGE_UNIT_COSTS["dino_s_512"][0] + STAGE_UNIT_COSTS["cnx_s2_256"][0]
            rf_hi = 2 * STAGE_UNIT_COSTS["dino_s_512"][1] + STAGE_UNIT_COSTS["cnx_s2_256"][1]
            row["tiers"] = {
                "cold": ("est", "权重读盘 OneShot ~89MB + S10 ~43MB + SD0-VAE ~335MB + DINO ~88MB + CNX ~110MB"
                                "(fp32 估) + CUDA ctx + 首图 warmup; b04 仅给 hash 吞吐 2.0GB/s -> 读+hash ~0.3s, "
                                "反序列化/初始化无实测, est 8-25s"),
                "warm": ("measured", f"1251.6 (t40) / 1245.4 (t50) ms 计时段 @3060 batch=1; "
                                     f"label_free 变体 1672.4/1529.5 ms (+23%~+34%, 同架构不同学生权重, 差异未定因); "
                                     f"峰值显存 2111.8 / 2135.0 MB"),
                "refcache": ("est", f"参考侧缓存(L162/L163/L165: 2x DINO + 1x CNX)后 est 省 {rf_save:.0f}-{rf_hi:.0f} ms/img "
                                    f"(~3-5%), 1180-1214 ms/img; B02 ReferenceFeatureCache(feature_cache='reference_v2') "
                                    f"即此契约; SG-25 计时段未计 PNG+裁判(未计时项见 §5)"),
            }
        else:  # student
            row["tiers"] = {
                "cold": ("est", "同 prism1（B0/B1 ckpt 未产出, mechanism_b pending）"),
                "warm": ("est", "单步残差 = P1 前向 + 残差分支(width64/176): est 与 prism1 同级 17-35 ms/img@4090/3060; "
                                "ckpt 未产出, 禁止写论文"),
                "refcache": ("est", "同 prism1"),
            }
        row["io_per_img_ms_off"] = round(io_per_img_off, 1)
        row["io_per_img_ms_on"] = round(io_per_img_on, 1)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 5) NFE ≠ 端到端速度：量化差异来源
# --------------------------------------------------------------------------- #
def nfe_gap_table() -> list[tuple[str, str, str]]:
    """NFE(velocity 评估) → image-equivalent 前向 → 实测/估算延迟 的逐级放大。"""
    p1_3060 = MEASURED["p1_ms_3060"].value
    o3_ms = MEASURED["o3_t40_ms"].value
    d1_ms = MEASURED["v20_d1_ms"].value
    return [
        # engine, NFE->forwards, latency
        ("prism1", "NFE 1 x1(无CFG) = 1 img-eq", f"17.0 ms @4090 / ~{p1_3060:.0f} ms @3060 (measured)"),
        ("d6_flow", "NFE 16 x2(cs_j) = 32 img-eq; agent2.md '20NFE' 系笔误(实为 o3 的 10 步 x2)",
         f"族实测 76.6-134.1 ms/img @3060 协议批(measured)"),
        ("hybrid", "NFE 16 x4(full) + P1 1 = 65 流 img-eq (fresh 总 68); x4 放大使张量工作量 4x 于 NFE 直觉",
         "eval 缓存路径 1 前向/图(仅 decode); fresh est 85-150 ms/img"),
        ("mecha_a", "NFE 16 x2(cs_j) + P1 1 = 33 img-eq(教师口径 fresh)",
         "eval 缓存路径 1 前向/图(仅 decode) + 门控算子; 无延迟臂"),
        ("o3_ref", "NFE 20 x2(CFG cat[zz,zz]) = 40 img-eq + 非 flow 7 前向(VAE enc/4x DINO/CNX/S10) = 47",
         f"{o3_ms:.1f} ms 计时段 @3060 batch=1 (measured)"),
    ], o3_ms, p1_3060, d1_ms


# --------------------------------------------------------------------------- #
# 6) 最小 GPU 实测清单（MEASURE/G 队列）
# --------------------------------------------------------------------------- #
MUST_MEASURE = [
    ("M1", "阶段分解计时: o3_ref 单样本计时段内部再分段(VAE enc / 4x DINO+CNX / 20 vel evals / S10 dec), "
           "torch.cuda.synchronize 包段; 把 §4 的 est 区间换成实测", "exp/v24/o3_sg25_eval.py 计时段加分段时钟"),
    ("M2", "d6_flow(cs_j, D6H) 协议批延迟: 750 任务 batch=25 墙钟(ms/img) + 峰值显存 —— 主表引擎唯一缺的直接锚点", "r1 臂重跑带 timing"),
    ("M3", "hybrid fresh 路径延迟: 65 img-eq 口径实测(缓存路径已有, fresh 未测), 验证 est 90-150 ms/img", "R7 端到端"),
    ("M4", "PRISM-1 @3060 正式 batch 批口径重测(现值 ~29 ms 为换算记录), 同卡同协议与 o3 1.25 s 对齐", "SG-25 协议批化"),
    ("M5", "cold 首图延迟(全引擎): 权重读盘+反序列化+CUDA ctx+首图 warmup 一次性秒数; b04 只测了 hash 吞吐", "部署文档需要"),
    ("M6", "同参考复用收益(refcache): o3 去掉 ref 侧 2x DINO+1x CNX 的实测差, 验证 est 38-65 ms (~3-5%)", "B02 cache 落地依据"),
    ("M7", "label_free vs O3 计时段差 +0.28-0.42 s/img 定因(同架构不同权重不应差 23-34%): 反解 forward 单价 "
           "33.9-38.6ms vs 25.8-28.1ms", "R7 前必须解释"),
    ("M8", "评测裁判墙钟: measure.py 4 裁判(CSD/DINO/CLIP/LPIPS) 750 图秒数(现含在 ~50s 臂墙钟内未分离)", "分离推理/评测口径"),
    ("M9", "峰值显存: prism1/d6_flow/hybrid 协议批(batch25/50)推理显存(现有只有 o3 batch=1 2.1GB 与 V20 训练 4.06GB)", "12GB 3060 部署边界"),
]


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def _fmt_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()


def main() -> int:
    global P
    P, notes = _live_param_counts()
    print("=" * 96)
    print("A07 完整 RGB->RGB 性能归因模型 (静态可计算, CPU-only; T_total = T_encode+T_feature+T_style"
          "+T_anchor+T_flow+T_decode+T_IO)")
    print("=" * 96)
    print(f"[params] {'; '.join(notes)}")

    # ---- 表 1: 参数量 ----
    print("\n## 表 1 参数量 (M)")
    w = [12, 16, 14, 46]
    print(_fmt_row(["engine", "params(M)", "口径", "组成"], w))
    rows1 = [
        ("prism1", P["prism1_net"], "live/const", "SpatiallyRoutedChordNet(width96) 单步 net"),
        ("d6_flow", P["slfunet_d6"], "live/const", "SLFUNet(base144) 8 步 RK2 流; 文档旧口径 ~20.8M, ledger 记 22M"),
        ("hybrid", P["slfunet_d6"] + P["prism1_net"], "sum", "SLFUNet + P1 (双模型串行, 权重独立)"),
        ("mecha_a", P["slfunet_d6"] + P["prism1_net"], "sum", "同 hybrid 教师口径 + 门控(A0/A1 零参数)"),
        ("o3_ref", P["oneshot_o3"] + P["ref_adapter"] + P["fiber_enc"], "live/const",
         f"OneShot(SLFUNet+RefAttn {P.get('refattn_head', 0.281):.3f}M) + Adapter + FiberEnc"),
        ("student", P["student_b0_total"], "live/const",
         f"P1 基座 + B0 残差 {P.get('student_b0_residual', 0.362):.3f}M (B1: {P.get('student_b1_total', 5.98):.2f}M, pending)"),
        ("共享栈", P["s10_decoder"], "const", "S10 decoder 10.66M(含 post_quant_conv) + SD0-VAE/DINO/CNX 特征栈"),
        ("评测裁判", P["csd_vit_l"] + P["dino_v2_s"] + P["clip_b32"] + P["lpips_alex"], "const",
         "CSD ViT-L + DINOv2-S + CLIP-B32 + LPIPS-alex; 仅 measure 阶段, 不入部署 T_total"),
    ]
    for r in rows1:
        print(_fmt_row([str(r[0]), f"{r[1]:.3f}", str(r[2]), str(r[3])], w))

    # ---- 表 2: NFE / 实际前向 ----
    print("\n## 表 2 NFE vs 实际模型前向 (单图 image-equivalent)")
    w2 = [12, 7, 8, 10, 12, 12, 44]
    print(_fmt_row(["engine", "NFE", "CFG放大", "flow前向", "fresh总", "协议批总", "结构依据(代码)"], w2))
    for e in ENGINES:
        print(_fmt_row([e.name, str(e.nfe_vel_evals), f"x{e.cfg_mult}", str(e.flow_img_equiv),
                        str(e.forwards_per_img_fresh), str(e.forwards_per_img_protocol),
                        e.code_basis[:44] + "..."], w2))
    print(_fmt_row(["fresh", "prism1/student: enc1+feat2(DINO x2)+style1+flow1+dec1 = 6;", "", "", "", "",
                    "d6/mecha: enc1+feat1+flow+dec1 (hybrid 流65已含P1); o3: enc1+feat4+style1+flow40+dec1"], w2))
    print(_fmt_row(["协议批", "test750 臂缓存齐备口径: prism1/student=3(style tiny+flow+dec); d6=33(bank 查表无 NN);", "", "", "", "",
                    "hybrid/mecha=1(教师 latent 缓存线性混合/门控, 仅 decode); o3=42(flow40+style tiny+dec)"], w2))
    print(_fmt_row(["说明", "NFE=velocity 评估次数; CFG 放大 = 每次 eval 的 batch 维(cs_j cat2 / full repeat4)", "", "", "", "", ""], w2))

    # ---- 表 3: 实测锚点 ----
    print("\n## 表 3 实测锚点 (全部带出处)")
    w3 = [22, 14, 10, 48]
    print(_fmt_row(["key", "value", "unit", "source (note)"], w3))
    for k, m in MEASURED.items():
        if k == "a04_arms":
            continue
        print(_fmt_row([k, f"{m.value:g}", m.unit, f"{m.source} ({m.note})"][:4], [22, 14, 10, 48]))

    # ---- 表 4: o3 计时段 1.25 s/img 阶段占比反推 ----
    rows4, fixed_lo, fixed_hi = stage_share_table_o3()
    print("\n## 表 4 o3_ref 计时段 1245.4-1251.6 ms 的阶段占比反推 (est/derived, @3060 batch=1)")
    w4 = [30, 18, 14, 10, 22]
    print(_fmt_row(["stage", "区间", "占比", "kind", "依据"], w4))
    for r in rows4:
        print(_fmt_row([r[0], r[1], r[2], r[3], r[4][:22] + "..."], w4))
    print(f"  固定栈(非 flow)合计 = {fixed_lo:.0f}-{fixed_hi:.0f} ms -> **flow 占计时段 ~83-90%**; "
          f"feature 提取不是大头, flow 的 40 次 image-eq 前向才是")
    print(f"  未计时项(计时段外, o3_sg25_eval L189-197): PIL PNG save + 裁判 stack.extract(3图: DINO/CSD/CLIP) "
          f"+ lpips + VGG gram x2; 墙钟 > 计时段")

    # ---- 表 5: NFE != 端到端 量化 ----
    rows5, o3_ms, p1_3060, d1_ms = nfe_gap_table()
    print("\n## 表 5 '20 NFE != 端到端速度' 逐级放大 (量化差异来源)")
    w5 = [12, 62, 40]
    print(_fmt_row(["engine", "NFE -> 前向放大", "延迟 (出处)"], w5))
    for r in rows5:
        print(_fmt_row([r[0], r[1][:62], r[2][:40]], w5))
    print(f"  差异来源四层: (1) NFE 只数 velocity 评估, CFG batch 放大 x2/x4 后 o3=40 img-eq; "
          f"(2) 非 flow 固定栈 7 前向(VAE enc/4xDINO/CNX/S10) ~128-215 ms; "
          f"(3) batch=1 启动开销主导(单 forward 24-28ms) vs 协议批 ~2.1ms/img-eq 差 ~12x; "
          f"(4) 计时段外 I/O+裁判。")
    print(f"  综合比值: o3 vs prism1 @3060 = {o3_ms:.0f}/{p1_3060:.0f} = {o3_ms / p1_3060:.0f}x 延迟, "
          f"NFE 比 = 20x, 流前向比 = 40x, 全栈前向比 = 47/6 ≈ 8x —— NFE 单独既高估全栈比又低估流比")
    print(f"  交叉验证: V20 D1(full 口径) {d1_ms:.1f} ms/img / 64 img-eq = {d1_ms / 64:.2f} ms/img-eq(协议批), "
          f"是 o3 batch=1 反解单价(24-28.1ms)的 ~1/{(24.0 + 28.1) / 2 / (d1_ms / 64):.0f} —— 批化摊销是第二放大源")

    # ---- 表 6: 三档口径 ----
    print("\n## 表 6 三档口径 (cold / warm / ref-cache; 每档标 measured|derived|est)")
    for row in tier_model():
        print(f"  [{row['engine']}]")
        for tier, (kind, desc) in row["tiers"].items():
            print(f"    {tier:10s} ({kind:8s}) {desc}")

    # ---- 表 7: T_IO 并入 (B04 对接) ----
    print("\n## 表 7 T_IO (b04 实测绝对秒数, 协议 750 臂口径)")
    io_items = [
        ("infer torch.save ~738MB", MEASURED["io_infer_save_s"], "src/infer/run.py L75-77, 每臂生成期一次"),
        ("measure torch.load ~738MB", MEASURED["io_measure_load_s"], "src/eval/measure.py L62"),
        ("eval.json dumps+写", MEASURED["io_eval_json_s"], "measure.py L109-112"),
        ("features.pt", MEASURED["io_features_s"], "measure.py L113-115"),
        ("PNG 750 张 (write_png=True 才有)", MEASURED["io_png750_s"], "runner 默认 --no-png"),
        ("sha256 30MB (B03 hash 开启才有)", MEASURED["io_sha30mb_s"], "runtime.sha256_file"),
        ("registry register (B03 接入才有)", MEASURED["io_register_s"], "设 WEAVE_GIT_SHA 可降 20x"),
    ]
    tot_off = (MEASURED["io_infer_save_s"].value + MEASURED["io_measure_load_s"].value
               + MEASURED["io_eval_json_s"].value + MEASURED["io_features_s"].value)
    for name, m, note in io_items:
        print(_fmt_row([name, f"{m.value:g} {m.unit}", note], [36, 14, 44]))
    print(f"  T_IO 合计/臂 = {tot_off:.2f} s (PNG 关, 必付) ~ {tot_off + MEASURED['io_png750_s'].value:.2f} s (PNG 开); "
          f"占 ~{MEASURED['io_arm_wall_s'].value:.0f}s 臂墙钟 3.4%/9.8% (b04 §8)")
    print(f"  单图摊销 = {1000 * tot_off / 750:.1f} ms (PNG 关) / {1000 * (tot_off + MEASURED['io_png750_s'].value) / 750:.1f} ms (PNG 开)")
    print("  结论: T_IO 与模型快慢无关, 改更强 GPU 不缩短; batch .pt 分片 ~4MB 粒度合理, 逐任务小文件是反模式(b04 §8.6)")

    # ---- 表 8: 最小 GPU 实测清单 ----
    print("\n## 表 8 必须实测的最小清单 (MEASURE/G 队列; 估算未替换前不得写入论文)")
    for mid, desc, where in MUST_MEASURE:
        print(f"  {mid}: {desc}  [{where}]")

    print("\nA07_COST_MODEL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
