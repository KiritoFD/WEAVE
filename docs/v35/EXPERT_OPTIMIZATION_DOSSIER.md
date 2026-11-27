# CLEAVE Final 架构整理与推理优化专家简报（Expert Optimization Dossier）

> **目标读者**：推理性能与系统架构优化专家（目标硬件：**RTX 3060 12GB** 本地端/Windows 测试与 **RTX 4090 24GB** 吞吐上限，512×512 Cold-Start `RGB → RGB` 完整链路）。  
> **基准报告**：[`V35_FINAL_REPORT.md`](V35_FINAL_REPORT.md) · **实测数据**：[`p8_final_report.json`](data/p8_final_report.json) / [`p9_paper_audit_report.json`](data/p9_paper_audit_report.json) · **Pareto 总图**：[`cleave_pareto_triptych.svg`](../performance/figures/cleave_pareto_triptych.svg)

---

## 一、系统定位与“双 Core”交付资产清单

CLEAVE（**Zero-DINO** 风格迁移系统）由两条互补分支构成联合 Pareto 前沿：
1. **E-Core Original（1-NFE 单步高速路径）**：`11.13M` `UnifiedHybridUNet`，4090 `b16` 冷启 **`33.1 img/s`**，`l=0.25..1.00` 覆盖 `DINO-C 0.9166→0.5675` / `CSD Top-1 2.33%→66.83%`。
2. **P-Core Final（44M 高保真/强风格精修路径）**：`E-Core` + `A3` 解析起点 + `44.54M` `PCoreSLFUNet`（`base_dim=208, cond_dim=384`），4090 `b16` 冷启 **`14.3 img/s` (`69.82 ms/img`)**，`s=0.60..1.50` 覆盖 `DINO-C 0.7813→0.3435` / `CSD Top-1 38.83%→91.17%`。

### 本地已拉齐并校验的可部署权重 / 运行期必需资产

> 以下二进制文件因 `.gitignore` 规则不入 Git 库，均已从 `4090` 拉回本地并校验 SHA-256。

| 角色 | 本地路径（相对仓库根） | 大小 | 参数量 | SHA-256 / 备注 |
|---|---|---:|---:|---|
| **P-Core Final（P8 锁定）** | `runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt` | `170.0 MB` | `44,540,532` | `50BCBB4FBC3DAF527A0349D2E68B604914C216FAE3AC9746B2A49EFB3C36AD4C`（`ema`, step 3500） |
| **E-Core Original（1-NFE）** | `runs/v32_nodino/ecore/ECore_NoDINO/ECore_NoDINO_final.pt` | `42.5 MB` | `11,131,268` | `9E226F6F8D1FC82F5AAF4D0F74954447008355F58B50080171FC89F4283535DC`（内含 `raw_stats` 与 `priors`） |
| **P3 Stiefel 消融对照权重** | `runs/v35_final/p3/P3_Q3_stiefel_nopc/P3_Q3_stiefel_nopc_final.pt` | `170.0 MB` | `44,540,532` | `9E12B2C0718A244B2CAAA8586A3E88CBE472101DD02984FC79F8372EA09A3D1B` |
| **A0 SD-VAE Encoder** | `runs/a0/a0_encoder_only.pt` | `130.4 MB` | `34,163,664` | `AutoencoderKL` 冻结编码器（4ch latent, `×0.18215`） |
| **S10B Decoder（主用）** | `deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt` | `40.7 MB` | `10,657,943` | `AC0BF25007D28B2234AA6AFCFEAB3DD35242F5AEF69F9ED49194A9A2BA83A58D`（含 `post_quant_conv`） |
| **S10B Standalone 包装** | `deliver/checkpoints/s10b_standalone/decoder.pt` | `40.7 MB` | `10,657,943` | 与 `s10_p5_lowlr` 张量同源；独立加载器见 [`decoder.py`](../../deliver/checkpoints/s10b_standalone/decoder.py) |
| **ConvNeXt-Tiny Stage3** | `~/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth` | `109.1 MB` | `12,348,000` | 仅取 `features[:6]`（`256px → (B,384,16,16)`） |
| **K=8 子画派原型库** | `runs/v32_nodino/cache/k8_cnx417_bank.pt` | `5.3 MB` | — | `sub_scene (5,8,6144)` + `sub_raw417 (5,8,417)` |
| **A3 解析先验** | `runs/v34_frontier/cache/v34_analytic_priors.pt` | `3.1 KB` | — | `P_Y, P_C, u_y, mu_low_1d, std_low_1d, mu_c, std_c` |
| **Stiefel 投影基 (`W_stiefel_cnx`)** | `runs/v32_nodino/cache/cnx_stiefel_cache.pt` 内 `(384, 32)` 张量 | `48 KB`（全缓存 `1.03 GB`） | — | 推理仅需其中的 `W_stiefel_cnx` |
| **Y16 LCN 亮度投影 (`luma_proj`)** | 来自 `runs/v18/cache/ccf_pool_cache.pt` | `20 B` | — | `w=[0.1297, 0.1152, 0.0413, -0.1495], b=0.4600`（训练缓存 1.4GB） |

---

## 二、端到端推理调用链与代码入口

完整冷启测量入口位于 [`exp/v35_final/common.py`](../../exp/v35_final/common.py#L521-L615) 的 `measure_e2e_speed_and_params`：

```text
rgb_in (B, 3, 512, 512), fp32/bf16 [0,1]
  │
  ├─[Stage 1: 编码与结构提取]──────────────────────────────────────────────
  │   ├─ A0 VAE Encoder (bf16, 34.16M): load_a0_encoder()  [exp/v25/r14_e2e_benchmark.py]
  │   │    z_s = a0_enc.encode(rgb_in.bfloat16()).latent_dist.mean * 0.18215  -> (B, 4, 64, 64)
  │   ├─ rgb_to_structure_64 + local_contrast_norm          [src/utils/structure.py]
  │   │    c64 -> (B, 4, 64, 64)  (供 E-Core 使用)
  │   └─ project_latent_y16_lcn(z_s, w_luma, b_luma)        [exp/v20/slf_flow.py]
  │        y16 -> (B, 1, 16, 16) (供 P-Core bottleneck 使用)
  │
  ├─[Stage 2: Zero-DINO 语义布局与子画派匹配]─────────────────────────────
  │   ├─ ConvNeXtTinyStage3Extractor.extract_gray_layout()  [exp/v32_nodino/cnx_layout.py]
  │   │    256px gray3 -> features[:6] (12.35M) -> 空间白化+球面归一 -> g16: (B, 384, 16, 16)
  │   ├─ extract_cnx_q32(g16, w32_st) -> stiefel_polar_quotient [exp/v21/stiefel_ref_cache.py]
  │   │    einsum(384->32) -> float32 torch.linalg.svd(B, 32, 256) -> U@Vh*16 -> q16: (B, 32, 16, 16)
  │   └─ match_cnx_k8_submode(g16, tid, cnx_bank) -> u: (B, 417)
  │
  ├─[Stage 3: E-Core (1-NFE) + A3 解析起点]─────────────────────────────────
  │   ├─ UnifiedHybridUNet.forward_latent_opt(...)          [exp/v28_unified/student_models.py]
  │   │    1-NFE UNet (11.13M, fp32/channels_last inp) -> ze: (B, 4, 64, 64)
  │   └─ build_pcore_startpoint(ze, tid, priors_a, 1.0, 1.0) -> reconstruct_analytic_draft (mode="A3")
  │        19x19 可分离高斯低通 + L_low/P_C 矩重校准 -> ze_s: (B, 4, 64, 64) [exp/v34_frontier/stage_a_startpoint.py]
  │
  ├─[Stage 4: P-Core 多步流积分 (Dual-CFG)]────────────────────────────────
  │   └─ integrate_v33_trajectory(pcore, ze_s, eps, y16, q16, tid, u, steps=6, solver="dpm2m", **kw)
  │        PCoreSLFUNet (44.54M, base_dim=208, cond_dim=384) [exp/v29_biglittle/pcore_model.py]
  │        z_init = (1 - tau0)*eps + tau0*ze_s               [exp/v33_semantic_control/models.py]
  │        每步调用 _vel(): cat 4 分支 [sc, s0, 0c, 00] -> batch 扩为 4*B -> bf16_autocast 前向
  │        -> zo: (B, 4, 64, 64)
  │
  └─[Stage 5: S10B 轻量解码]──────────────────────────────────────────────
      └─ decode_s10(dec, zo, batch=bs)  [src/utils/vae.py + StudentDecoder(10.66M, fp16)]
           -> rgb_out: (B, 3, 512, 512) [0, 1]
```

### 推理参数调度 (`make_v35_schedule` + `P8` 校准偏移)

P8 锁定校准：[`p8_final_report.json`](data/p8_final_report.json) 中 `ws_mul=1.0`, `wc_mul=1.2`, `dtau0=-0.05`, `steps=6`, `alpha_y=1.0, alpha_c=1.0`。核心档位见 [`exp/v35_final/common.py#L70-L103`](../../exp/v35_final/common.py#L70-L103) 与 [`exp/v34_frontier/train_common.py#L215-L240`](../../exp/v34_frontier/train_common.py#L215-L240)。

---

## 三、已测延迟分解、参数与显存瓶颈分析

### 3.1 RTX 4090（PyTorch Eager，未启用 `compile` / CUDA Graph / TensorRT）实测

数据源：[`p8_final_report.json`](data/p8_final_report.json) `perf_4090`：

| 指标 | Batch = 1 | Batch = 16 | 占比 (`b16`) | 核心代码位置 |
|---|---:|---:|---:|---|
| **端到端冷启耗时 (`RGB→RGB`)** | **`108.7 ms/img`** (`9.2 img/s`) | **`69.82 ms/img`** (`14.3 img/s`) | **100.0%** | [`common.py#L521-L615`](../../exp/v35_final/common.py#L521-L615) |
| **推理峰值显存** | **`0.83 GB`** | **`6.52 GB`** | — | — |
| `Stage 1: A0 Encode + Struct` | — | **`15.79 ms/img`** | **22.6%** | `a0_enc` (34.16M `bf16`) |
| `Stage 2: ConvNeXt + Stiefel` | — | **`1.00 ms/img`** | **1.4%** | `ConvNeXt-S3` (12.35M) + `linalg.svd` |
| `Stage 3: E-Core + A3 Analytic` | — | **`1.05 ms/img`** | **1.5%** | `UnifiedHybridUNet` (11.13M) + `A3` |
| `Stage 4: P-Core ODE (Dual-CFG)` | — | **`39.49 ms/img`** | **56.6%** | `PCoreSLFUNet` (44.54M `bf16`) |
| `Stage 5: S10B Decode` | — | **`12.49 ms/img`** | **17.9%** | `S10` (10.66M `fp16`) |

> 历史 RTX 3060 同拓扑解码对照（见 [`docs/vae/08_STAGED_PROFILE_AND_TRT.md`](../vae/08_STAGED_PROFILE_AND_TRT.md)）：`PyTorch B8` 纯 Decoder `68.83 ms/img` → **`TensorRT FP16 B8` `27.08 ms/img`（`2.54×` 加速）**。

---

## 四、源码审查直接暴露的 6 项高 ROI 优化切入点（供专家评估）

以下均直接对照源码行级定位，未做未经实测的加速数值承诺：

1. **求解器首步双算 (`steps=6` 实际触发 7 次 `_vel` = 28 次前向)**：
   - 位置：[`exp/v33_semantic_control/models.py#L546-L561`](../../exp/v33_semantic_control/models.py#L546-L561)。
   - 现象：`i=0` 时 `prev_v is None` 走 `else` 中点法（调用 `_vel` 2 次：`v1` 与 `v2`），随后 `i=1..5` 各调 1 次，导致名义 `steps=6` 实际跑了 **7 次 4 分支 `_vel`（28 NFE）**；此外 `pcore_ode_ms` 可优先验证 `steps=6→3/4`、首步单评估 Euler 启动及自适应步长表。
2. **4 分支 Dual-CFG 批内放大 (`4×B`) 与条件冗余**：
   - 位置：[`exp/v33_semantic_control/models.py#L512-L542`](../../exp/v33_semantic_control/models.py#L512-L542)。
   - 现象：每步将 batch 复制为 `[v_sc, v_s0, v_0c, v_00]`（`4B` 吞吐）。其中 `v_00`（无条件）与 `v_0c`（纯内容条件）不依赖 `style_id / u_style`；此外可评估：**风格/内容联合 CFG 蒸馏**（`4B → 1B`，理论省去 `75%` P-Core FLOPs）或尾部步剪除 `v_s0` / 复用早期无条件预测。
3. **`PCoreSLFUNet` 缺少 `channels_last`、上采样写法与算子编译**：
   - 位置：[`exp/v29_biglittle/pcore_model.py#L167-L177`](../../exp/v29_biglittle/pcore_model.py#L167-L177)。
   - 现象：仅 `UnifiedHybridUNet` 对输入调用了 `.contiguous(memory_format=torch.channels_last)`，而占 **56.6%** 耗时的 `PCoreSLFUNet` 与 `S10B`/`A0` 均以默认 `NCHW` 运行；上采样采用 `.repeat_interleave(2, -2).repeat_interleave(2, -1)` 替代 `F.interpolate(..., mode="nearest")`；且整条推理链尚未启用 `torch.compile(mode="reduce-overhead")` / `CUDA Graph`（固定 `512×512` 静态 shape 极适合 CUDA Graph 消除 `GroupNorm+SiLU+FiLM` 小 kernel launch 开销）。
4. **`A0 VAE Encoder` 占端到端 `22.6%` (`15.79 ms/img @ 4090`)**：
   - 位置：[`exp/v25/r14_e2e_benchmark.py#L73-L80`](../../exp/v25/r14_e2e_benchmark.py#L73-L80) 与 [`exp/v35_final/common.py#L559`](../../exp/v35_final/common.py#L559)。
   - 现象：解码端已由 `49.49M` 剪枝至 `10.66M S10B`，编码端仍保留未剪枝的 `34.16M` SD-VAE `Encoder`（且 `a0_enc.encode(...)` 构造 `DiagonalGaussianDistribution` 后只取 `.mean`）。在 `3060 12GB` 上，结构化剪枝/蒸馏对齐 latent 空间的轻量 Encoder + `TensorRT FP16`（同时编译 `A0 Encoder`、`P-Core`、`S10B Decoder`）预期空间极大。
5. **全图端到端仅 `0.83 GB` (`b1`) / `6.52 GB` (`b16`) 的 3060 内存与高精度类型混杂**：
   - 位置：[`exp/v35_final/common.py#L558-L578`](../../exp/v35_final/common.py#L558-L578)。
   - 现象：`ECore` 主干、ConvNeXt、`stiefel_polar_quotient`（`svd(32, 256)`）和 `y16/c64` 当前都在 `fp32` 下串行运行且各阶段在 `torch.cuda.synchronize()` 断点同步；实际除 `svd`（可测试极坐标牛顿-舒尔茨迭代 `Newton-Schulz` 替代 CPU/GPU 同步多的 `linalg.svd`）外，`ConvNeXt-S3`、`E-Core`、`P-Core`、`A0`、`S10B` 均可统一低精度（`fp16` on 3060 Turing/Ampere / `bf16` or TensorRT `FP16`/`INT8`）并流水线化。
6. **推理资产去重打包（脱离训练巨型缓存）**：
   - 现状：[`common.py#L525-L536`](../../exp/v35_final/common.py#L525-L536) 测速函数仍从 `1.03 GB` `cnx_stiefel_cache.pt` 仅读取 `(384, 32)` `W_stiefel_cnx`、从 `1.4 GB` `ccf_pool_cache.pt` 仅读取 5 个标量 `luma_proj`。建议打包独立的轻量 `deploy_bundle_v35` 仅加载必需张量（`~400 MB` 全链路），降低冷启 I/O 与 3060 部署复杂度。
