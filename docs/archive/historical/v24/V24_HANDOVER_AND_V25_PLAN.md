# WEAVE 阶段性实验总结、交接文档与 V25 下一阶段计划

> **日期**：2026-11-19｜**覆盖阶段**：V22（受控消融与 `Peak1500`）→ V23（双分支纠偏、高 `DINO-C` 前沿与跨风格 One-Shot）→ V24（内容优先方向修正、`M0–M6` 与 `O1–O4`）  
> **数据台账**：`docs/performance/metrics.csv`（共 **449 条**有效实验臂与基线，100% `S10` 解码 × `test750` 进程隔离四裁判）  
> **测试守卫**：`pytest tests/test_no_absolute_paths.py tests/test_markdown_links.py tests/v22/test_core.py` **25 passed**

---

## 一、全阶段核心 Pareto 全景（WEAVE Cross600）

按 **`DINO-C`（内容保持，横轴）从高到低** 排列三个清晰档位，便于快速定位当前全仓最高水平：

### 1. 极高内容 & 内容优先主目标档（`DINO-C ≥ 0.70`, `LPIPS ≤ 0.52`）
| 臂 / 模型 | 阶段 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Aff ↑ | Top-1 ↑ | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| `M0_prism1_l075` | V19/V24 | **0.7715** | 0.4371 | **0.4848** | 0.5611 | 37.67% | 极高保真基座 (`17 ms`) |
| `M0_prism1_l100` | V19/V24 | **0.7440** | 0.4430 | 0.5057 | 0.5655 | 44.33% | 标准 `PRISM-1` 基座 (`17 ms`) |
| **`M1_lin_g40`** (`0.60 P1 + 0.40 D6`) | **V24** | **0.7467** | **0.4600** | **0.5006** | **0.5791** | **50.17%** | **四项全面严格支配 `PRISM-1`** |
| **`M1_p1_dswd_g35`** (`0.65 P1 + 0.35 DualSWD`) | **V24** | **0.7394** | **0.4664** | **0.5053** | **0.5840** | **49.50%** | 半 NFE 双分支混合 |
| **`M1_lin_g55`** (`0.45 P1 + 0.55 D6`) | **V24** | **0.7321** | **0.4745** | **0.5053** | **0.5892** | **52.67%** | **达到 `DINO-C ≥ 0.73` 下 `DINO-S +0.0315` 验收线** |
| **`M1_p1_dswd_g50`** (`0.50 P1 + 0.50 DualSWD`) | **V24** | **0.7167** | **0.4907** | 0.5137 | **0.6007** | **51.83%** | `DINO-S` 突破 `0.49`（视觉干净） |
| `M2_dwt_ll25_hf85` (Haar 分带混合) | V24 | 0.6994 | 0.4915 | 0.5219 | 0.6008 | **56.17%** | 验证 LL 低频不可硬冻结 |

### 2. 均衡高内容档（`DINO-C ∈ [0.60, 0.67]`, 对标 `V20 D5/D6` 与 `StyleGallery`）
| 臂 / 模型 | 阶段 | NFE | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Top-1 ↑ | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| `M0_v20_d6` | V20/V24 | 32 | **0.6688** | 0.5107 | **0.5374** | 55.67% | V20 高内容对照 |
| **`HiC_JPk_csj65_t4`** | **V23** | **20** | **0.6299** | 0.5330 | 0.5649 | **58.83%** | 同 `DINO-C` 击败 V20 D5 (`+3.7pp`), NFE 减半 |
| **`Joint_T1_C1AB_p1_bal`** | **V23** | 40 | **0.6258** | 0.5243 | 0.5703 | 58.00% | **免 T4 (`PRISM-1` 锚点, ~225ms)** |
| **`DualSWD_pure_hiT_ws52`** | **V23** | **20** | **0.6164** | 0.5467 | 0.5674 | 51.17% | 内容超 SG, 视觉审计通过 |
| **`F3_dual_fixed_orig`** | **V23** | **20** | 0.6096 | **0.5522** | 0.5723 | 49.50% | 全仓最近 SG 前沿 (`-0.008 D-S`) |
| `StyleGallery` (外部扩散) | Baseline | 50×opt | 0.6113 | **0.5604** | 0.5576 | 52.50% | `30.4 s/img`（慢 ~150×） |

### 3. 强风格档（`DINO-C ∈ [0.48, 0.55]`, `Top-1 ≥ 69%`）
| 臂 / 模型 | 阶段 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Top-1 ↑ | 备注 |
|---|---|---:|---:|---:|---:|---|
| `T4Pk_wc15` (`Peak1500`) | V22 | 0.5319 | 0.5610 | 0.6049 | **69.00%** | Bootstrap 零内容代价严格支配 `T0` (`+8.33pp SIG`) |
| `T4Pg45_wc10` (`Peak1500`) | V22 | 0.4779 | **0.5781** | 0.6268 | **76.83%** | 视觉审计通过的高风格甜点 (`206 ms/img` B1) |
| `Mid_C1_hiS` (+ConvNeXt 中尺度) | V23 | 0.4843 | 0.5750 | 0.6251 | **78.33%** | 相对 `Peak1500` 风格/内容同步外移 |

### 4. One-Shot 任意参考演进（3060 旁路，全程 `onehot=0`、无测试目标类别 ID）
| 阶段 / 臂 | 关键变化 | WEAVE `DINO-C` ↑ | WEAVE `DINO-S` ↑ | WEAVE `LPIPS` ↓ | WEAVE `Top-1` ↑ | SG-25 OOD (`125-pair`) |
|---|---|---:|---:|---:|---:|---|
| V22 `P2A_1shot_cs_j` | 冻结主干 + 有 bug `cs_j` | 0.6623 | 0.4482 | 0.5295 | 13.50% | — |
| V23 `R4/R5/R6` (全自配对) | 解冻 + `16×16` RefAttn | 0.6206 | 0.4997 | 0.5173 | 19.67% | — |
| V23 `M1/M2` (+50% 跨风格对) | 跨风格配对监督 | 0.5350 | **0.5326** | 0.5688 | 31.83% | `D-C 0.216 / D-S 0.321` |
| V23 `M3_ab_cross50_srcA0` | **+源图→跨风格 Anchor-Bridge** | 0.5489 | 0.5270 | 0.4458 | **38.00%** | `D-C 0.418 / D-S 0.263` (`1.63s`) |
| V24 `O4C_t60` (`M3 @ τ=0.60`) | 高内容起点扫描 | **0.7320** | 0.4867 | **0.3084** | 19.50% | — |
| V24 `O1_gray_dino_t45/t55` | `V` = 灰度 DINO (对照) | 0.5850 / 0.6781 | 0.5169 / 0.4995 | 0.4202 / 0.3502 | 30.67% / 22.33% | — |
| V24 `O2_rgb_convnext_t45/t55` | **`V` = RGB ConvNeXt s2** | 0.5829 / 0.6793 | 0.5104 / 0.4949 | 0.4198 / 0.3488 | 33.50% / 24.00% | — |
| **V24 `O3_rgb_cnx_gated_t45/t55`** | **`+ 全局 Oklab 置信度门控`** | **0.5845 / 0.6783** | **0.5245 / 0.5025** | 0.4213 / 0.3494 | **35.33% / 24.00%** | **`t50`: `D-C 0.5867 / ΔS +0.157` (`1.25s`)** |

---

## 二、已修复的关键工程 Bug 与确立的方法论铁律

### 1. 已修复并加单测锁死的底层 Bug
1. **双分支 `cs_j` 内容条件清零 Bug** (`exp/v22/core.py:145`, `exp/v22/ref_adapter.py:305`)：此前第二分支 `v_0c` 误传 `layout_scale=0`，修复为 `l2=[lscale, lscale]` 后，V20 Core-SLF 双分支暴涨 `+8.0pp`，V22 Dual 双分支暴涨 `+14.8pp`（`49.50% / DINO-S 0.5522 / DINO-C 0.6096`），彻底推翻“双分支必然崩塌”的错误旧结论。
2. **Haar IDWT 棋盘格错位 Bug** (`src/utils/structure.py:160`)：`idwt2_haar` 在 `stack` 后缺少 `.permute(0, 1, 2, 4, 3, 5)`，导致图像被切成 2×2 交错碎片（`same_top1` 暴跌至 41%）；修复后 `max_diff = 2.83e-7`，已写入 `test_haar_dwt_idwt_exact_inverse`。
3. **A0 Encoder + S10 端到端往返验证**：在 3060 实测 `A0 Encoder` + `S10 Decoder` 往返 **`DINO-C = 0.9710`（`MAE = 0.0189`）**，正式打通任意 RGB→RGB 推理（`1.25 s/img`, `2.1 GB VRAM`）。

### 2. 诚实判负机制与深层归因（避免 V25 重复踩坑）
| 判负实验 | 结果 | 深层根因（已定位） |
|---|---|---|
| `S32Zero` (`32×32` 零初始化残差) | 与同预算 `S32Ctrl` 差 `±0.17pp` | Coarse-16 瓶颈已足够，32×32 亮度结构对语义内容保持无增量 |
| `M3 / M4` (0.99M 1步 SWD 残差网) | `DINO-C 0.706 / DINO-S 0.462 / LPIPS 0.543 / Top1 32.7%` | ① 训练期用 `3×3` 模糊近似 `z0` 与测试 `n2c_z.pt` 分布不匹配；② 1 步常规卷积直接优化单图 `Patch-SWD` 注入了高频噪声（LPIPS 恶化 `+0.04`） |
| `M5` (0.99M 1步洁净 `D6` 蒸馏) | `DINO-C 0.732 / DINO-S 0.436 / Top1 41.5%` | 1 步 1M 卷积无法拟合 22M 模型 8 步 ODE 的非线性风格残差，平均化导致模糊与画风稀释 |
| `M6_flow4` (1.06M 4步确定性残差流) | `DINO-C 0.661 / DINO-S 0.423 / LPIPS 0.639 / Top1 22.7%` | 无 `dino_16` 条件 + 无噪声平滑确定性桥 + `z0` 训练-测试分布 gap → 4步 ODE 积分发散至 Impressionism 坍缩（91.7%） |
| `O4D_self / O4D_teacher` (1.57M 1步 One-Shot 残差) | `DINO-C 0.91~0.94 / Top1 1.0~1.8%` | ① `O2-Teacher` 内容权重 `2.2` 过于保守（`mid_cos=0.965` 几乎贴原图）；② 1 步网络易退化为恒等输出 |

---

## 三、双机当前状态与关键资产交接清单

### 1. 机器实时状态（2026-11-19 停机收束时刻）
- **4090 (`ssh 4090`, `/root/Workspace/WEAVE`)**：**GPU 空闲**（无活跃实验 tmux；仅保留系统自带 `cpolar`/`infsrv`）。
- **3060 (`ssh 3060`, `I:\Latent_Style\WEAVE`)**：**1 个收尾任务 `O3_label_free` (`PID=24468`) 正在后台自然收尾**（已跑过 `800/1200` 步，日志 `I:\Latent_Style\WEAVE\runs\v24\p2_o3_lf.log`，完成后自动写入 `runs/v24/p2_o3_labelfree.json`）。下一会话接手时第一条命令可查看结果：
  ```powershell
  ssh 3060 "powershell -NoProfile -Command \"Get-Content I:\Latent_Style\WEAVE\runs\v24\p2_o3_labelfree.json\""
  ```

### 2. 关键可复用资产位置
- **本地已救援权重**：`runs/v22/{rescue_slf_T4Ssnap1500.pt, rescue_slf_T4Ssnap1200.pt, tmatrix_rescue_slf_T4.pt}`；本地评测 JSON 缓存：`runs/v22/`, `runs/v23/`, `runs/v24/`。
- **本地视觉审计拼图**：
  - `archives/v23_traj_diag/`（Step 0 内容破坏定位轨迹图）
  - `archives/v23_frontier_audit/v23_frontier_grid.jpg`（`DualSWD` / `HiC_JPk` 高 DINO-C 前沿图）
  - `archives/v23_sg25_audit/sg25_audit_grid.jpg`（SG-25 25类 OOD 视觉诊断图）
  - `archives/v24_p0_audit/v24_p0_audit_grid.jpg`（`M0_prism1` vs `M1_lin_g55` 高内容对比图）
- **4090 关键缓存与模型**：
  - `runs/v24/cache/p0_base_latents_750.pt`（750 条 `PRISM-1 l075/l100` + `V20 D6` 潜空间缓存）
  - `runs/v24/cache/ref_teacher_pairs_2000.pt`（2000 对无类别真参考优化对，已同步至 3060 `runs/v24/cache/`）
  - `runs/v23/dual_swd/slf_DualSWD_pure.pt`, `runs/v23/joint_c1ab/slf_Joint_{T1,Pk}_C1AB.pt`
- **3060 关键模型与基准**：
  - `runs/v24/oneshot_val/O3_rgb_cnx_gated.pt`（当前最强 One-Shot 模型）
  - `runs/v24/sg_25_manifest_125.json` + `I:\Latent_Style\StyleGallery\`（25 风格家族官方目录）
  - `runs/a0/a0_encoder_only.pt` + `deliver/sd0_s10_profile/sd0_vae/`（已验证 `0.971` 往返 DINO-C 的 A0 Encoder）

---

## 四、V25 下一阶段实验计划方案（基于 V24 教训校准）

针对 V24 暴露出的核心瓶颈，建议 V25 聚焦三条高胜率、低工程浪费的精准路径：

### 方案 A（4090 主线）：消除 `z0` gap + 单引擎蒸馏 `M1` 混合前沿，冲击主目标 `DINO-C ≥ 0.73, DINO-S ≥ 0.49–0.51, LPIPS ≤ 0.50`
- **核心依据**：V24 `M1_lin_g55` (`0.7321 / 0.4745 / 0.5053 / 52.67%`) 与 `M1_p1_dswd_g50` (`0.7167 / 0.4907 / 0.5137 / 51.83%`) 证明高内容区间向右推移完全可行；而 `M3–M6` 失败是因为 **外挂 1M 小网容量不足 + 训练期用 `avg_pool` 假冒 `n2c_z.pt` 造成了严重分布偏差**。
- **执行步骤**：
  1. **修复训练-测试起点一致性**：用 4090 上现成的 `B4` (`P1_B4_distill50.pt`) + `canonicalizer` 为 4400 张训练集一次性生成真实的 `z0_tr`（耗时仅 ~1 分钟），彻底消除 `z_p1` 的 train-test gap。
  2. **两条低风险单引擎路线二选一/并行**：
     - **A1（单步增强 `PRISM-1+`, 1.5M–3M, `~20 ms/img`）**：直接解冻并热启动 `SpatiallyRoutedChordNet` (`PRISM-1` 自身权重，绝不从零外挂！)，以 **`0.45 * z_p1 + 0.55 * z_d6` / `0.50 * z_p1 + 0.50 * z_dswd`**（在训练集真 `z0_tr` 上生成的 4400 条无噪声混合目标！）做直接潜空间 + `ConvNeXt` 中尺度预算精调。既有 `PRISM-1` 原生 `0.744` 内容保真和 `~20 ms` 极速，又把 `DINO-S` 直接拉向 `0.49–0.50`！
     - **A2（22M 双分支 `SLFUNet` 4步快速高保真编辑, `~60 ms/img`）**：用已验证最强高内容架构 `SLFUNet (DualSWD)` 从真实 `z_p1` 出发做 **4 步 Rectified Flow Bridge**（完整保留 `dino_16` 与 `cs_j` 双分支 CFG），目标达到 **`DINO-C ≥ 0.73, DINO-S ≥ 0.50–0.51, LPIPS ≤ 0.50, Top-1 ≥ 55%`**。

### 方案 B（3060 One-Shot 线）：修正无标签教师强度 + 扩充 25+ 多风格池，攻克真正 OOD 泛化
- **核心依据**：V24 证实 `O3_rgb_cnx_gated`（灰度 DINO `Q/K` + RGB ConvNeXt `V` + 全局 Oklab 置信度门控）显著领先，且在 SG-25 上已达 `DINO-C=0.5867`、`1.25s/img`；主要短板是 ① `O2-Teacher` 内容权重 `2.2` 过于保守（`mid_cos=0.965` 变动微弱），② 训练集仍仅含 WikiArt 5 类油画，缺少动漫/水墨/线稿/抽象/赛博朋克覆盖。
- **执行步骤**：
  1. **回收 3060 正在跑的 `O3_label_free` 结果**，并与带 5 类偏置的 `O3_rgb_cnx_gated` 做 WEAVE test750 + SG-25 对比，量化保守教师 (`mid_cos=0.965`) 的真实风格迁移幅度。
  2. **重标定无标签教师 `O2-Teacher-v2`**：将内容权重从 `2.2` 降至 `0.7~0.9`（使筛选后 `mid_cos` 落在 **`0.80–0.86`**，既完整保住主体五官构图，又有肉眼明显的色彩与笔触替换），并在 4090 上生产 3000 对高质量无标签跨风格样本。
  3. **扩充多风格参考池（§O3 落地）**：从公开画作/插画/自然图库（严格排除 StyleGallery 官方 25 类测试图库）引入 **20–30 个风格家族（水墨/版画/动漫/粉彩/素描/抽象等）+ 自然照片内容图（COCO/FFHQ 风格）**，用 `O3_rgb_cnx_gated`（`onehot=0`, `srcA0 Anchor-Bridge`）联合微调。
  4. **严格同台基准对比**：在固定多内容×多参考的 StyleGallery 官方配对清单上，同台运行官方 `StyleGallery` 代码与 `V25 O3`，报告 `DINO-C`, `ΔS`, `LPIPS`, `VGG Gram`, `ArtFID` 与端到端延迟。

### 方案 C（文档、图表与论文 `PRISM` 收束）
- 若 V25-A 在 `DINO-C ≥ 0.73` 下稳定拿到 `DINO-S ≥ 0.49` 且单模型延迟 `< 60 ms`，即刻冻结最终双模型交付族：
  1. **`PRISM-Core-Fast`（主线单模型）**：高内容 (`DINO-C 0.73–0.75`)、低失真 (`LPIPS ≤ 0.50`)、极速 (`20–60 ms/img`)；
  2. **`PRISM-Ref`（One-Shot 单图参考版）**：无类别 ID、RGB ConvNeXt 外观交叉注意力、`1.25 s/img` 端到端。
- 同步更新 `docs/performance/MASTER_LEDGER.md`、重新生成 `docs/performance/figures/*.svg` 帕累托曲线，并修订 `docs/icme2027/paper.tex` 中方法叙事与消融表格（删除已淘汰的多余变体，聚焦上述最简双模型主线）。
