# 04 指标与结果

> 核心警告：本仓库存在**两套不可混用的评测口径**（750 all-pairs July / 600+150 Nov）。同名列数字不能直接比较。

## 1 指标定义

| 指标 | 方向 | 定义（当前实现） | 代码/证据 |
|---|---|---|---|
| DINO-S | ↑ | 生成图与目标风格参考图的 DINOv2 CLS 余弦相似度；历史实现取 `max cos(gen, refs)` | `archives/legacy-root-research-scripts-20260715/compute_dino_flat_results.py#L7-L10` |
| DINO-C | ↑ | 生成图与内容源图的 DINOv2 CLS 余弦 | 同上 L8 |
| DINO-structure | ↓ | patch self-similarity MSE | 同上 L10 |
| CLIP-S | ↑ | 生成图与目标风格参考图的 CLIP 相似度 | `utils/run_evaluation.py#L1331-L1354`（`clip_style`） |
| LPIPS | ↓ | 生成图 vs 内容图，AlexNet 感知距离 | `utils/run_evaluation.py#L592-L650` |

注意：`utils/run_evaluation.py` **不计算 DINO**；DINO 由 `scripts/_repro_dino_eval.py` 调外部脚本产出 `dino_summary.json`（[L39-L63](file:///g:/GitHub/Latent_Style/WEAVE/scripts/_repro_dino_eval.py#L39-L63)）。DINO 主干在 July 脚本里是 **dinov2-small**，Nov 缓存改用 **dinov2-base**（`Exp.md#L22`）——这本身就是口径差异，且文件未明确声明迁移日期（**未验证**）。

## 2 口径 A：论文主表（750 all-pairs，2026-07）

- 协议：150 内容图 × 5 目标风格 = 750 对（含 150 同风格对角线）；`train endpoint_adain_scale=1.0`，评测 `inference.json` 覆盖为 `2.0`；oriented-HF 配置。
- 论文报告值（WEAVE，D5-512）：**DINO-S 0.4916 / CLIP-S 0.7128 / LPIPS 0.2593 / DINO-C 0.8103**。
  - 来源 [config.json#L3-L12](file:///g:/GitHub/Latent_Style/WEAVE/config.json#L3-L12)；交叉核对 [verify_main_table.py#L77-L85](file:///g:/GitHub/Latent_Style/WEAVE/verify_main_table.py#L77-L85)。
- 对应 checkpoint：`runs/submission/canonical_oriented_epoch4/epoch_0004.pt`（`config.json#L8`）。

同族对比（`docs/713/SUBMISSION_HANDOFF_2026-07-15.md#L147-L151`）：

| 点 | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|
| Baseline epoch 6 | 0.4867 | 0.7075 | **0.2508** | **0.8281** |
| Oriented route epoch 4（论文） | **0.4915** | **0.7126** | 0.2596 | 0.8103 |
| Oriented route epoch 6 | 0.4878 | 0.7101 | 0.2563 | 0.8215 |

> 论文值 0.4916/0.7128/0.2593 与 handoff 的 0.4915/0.7126/0.2596 差 0.0001-0.0003，属四舍五入/复算误差，不是新结果。

## 3 口径 B：600 跨 + 150 同（2026-11，当前唯一有效主表）

- 协议：150 内容图 × 4 个非本风格目标 = **600 跨风格主表**；对角线 150 = **Identity 补充**；参考图固定 `first_image_of_target_style_test_dir`（`Exp.md#L3-L4`）。
- 同口径基线 `eval_base_ep10`（论文模型 epoch 10，L/14 CLIP + 统一 split）：

| 项目 | DINO-S | CLIP-S | DINO-C | LPIPS | 来源 |
|---|---:|---:|---:|---:|---|
| 600 跨风格（主表） | 0.3977 | 0.6425 | 0.7939 | 0.3053 | `Exp.md#L23` |
| 150 同风格（Identity） | 0.8263 | — | — | — | `Exp.md#L23` |

- 路线 A 试跑（**不进结论**，500/1000 步）：

| run | 600 DINO-S | 600 CLIP-S | 600 DINO-C | 600 LPIPS | 150 DINO-S |
|---|---:|---:|---:|---:|---:|
| routeA_smoke_ep12（500 步） | 0.3938 | 0.6461 | 0.8258 | 0.2722 | 0.8526 |
| a_ls005_lc01_s2_e1_s0（1000 步） | 0.3912 | 0.6454 | 0.8359 | 0.2667 | 0.8572 |
| a_ls01_lc01_s2_e1_s0（1000 步） | 0.3926 | 0.6454 | 0.8342 | 0.2670 | 0.8558 |

- 历史参考（旧 CLIP base-patch32、复用旧产出，**不是同管线**）：600 = 0.4080/0.7007/0.7667/0.3412；150 = 0.8264/0.8467/0.8240/0.2568（`Exp.md#L30`）。

## 4 架构探针（750 all-pairs，fine-tune 6ep，2026-07）

来源 `docs/713/EXPERIMENT_SUMMARY_FOR_METHOD_AND_NEXT_PLAN.md#L30-L56`；列为 DINO-S / DINO-C / CLIP-S / LPIPS / Off-DINO-S。

| 路线 | DINO-S | DINO-C | CLIP-S | LPIPS | 结论 |
|---|---:|---:|---:|---:|---|
| HF delta（AdaIN 1.5） | 0.4850 | 0.7966 | 0.7176 | 0.2929 | 路径连通，小幅 |
| HF delta strong | 0.4870 | 0.7991 | 0.7176 | 0.2955 | 首个可用改进 |
| **subband（当前架构）** | **0.4886** | 0.7981 | 0.7209 | 0.2966 | 安全最优 |
| spatial raw HF | 0.4901 | **0.4043** | 0.7483 | **0.5382** | 内容崩，拒绝 |
| subband + texture stats | 0.4884 | 0.7988 | 0.7194 | 0.2960 | 保守备选 |
| subband 去 style memory | 0.4849 | 0.7948 | 0.7167 | 0.2943 | 拒绝（memory 有用） |
| subband residual ×1.5 | 0.4875 | 0.7798 | 0.7211 | 0.3054 | 风格偏置，内容代价高 |

## 5 基线主表（`docs/79/main_table_v4_remote.csv`）

该 CSV 只有 `clip_s/lpips/musiq`，**没有 DINO-S/DINO-C**，因此不能与第 2、4 节的 DINO 数字直接比较。代表行（D5-512）：

| 方法 | CLIP-S | LPIPS | 参数 | 训练 |
|---|---:|---:|---|---|
| Identity | 0.693 | 0.000 | 0 | free |
| AdaIN | 0.668 | 0.743 | 0+144M | free |
| WCT (VGG19) | 0.706 | 0.635 | 0+144M | free |
| SD-Turbo | 0.693 | 0.003 | 0+1.1B | free |
| IP-Adapter | 0.829 | 0.636 | 0+1.4B | free |
| CUT | 0.714 | 0.374 | 7.0M | 322.6 min |
| SaMam | 0.582 | 0.243 | 8.8M | 436.0 min |
| Seedream 4.5 | 0.720 | 0.477 | API | API |
| **Ours (WEAVE)** | **0.715** | **0.382** | 903K+84M | 3.08 min |

（行号 `main_table_v4_remote.csv#L2-L40`。）

## 6 矛盾与待澄清（必须知道）

1. **CSV 的 WEAVE 行（0.715 / 0.382）与 `config.json` 主表（0.7128 / 0.2593）不一致**，且 CSV 参数写 903K，而当前论文模型约 1.04M。结论：CSV 是较早 v4 快照，**不要当作最新主表**（`docs/79/baseline_result_sources.md#L19` 也只说它是"snapshot, not proof"）。
2. **0.4916 / 0.4915 / 0.4886 / 0.3977 不是同一东西**：
   - 0.4916 = 论文 ckpt、750 all-pairs（≈0.4915 复算）；
   - 0.4886 = subband fine-tune 6ep 探针，**不是论文 ckpt**；
   - 0.3977 = 600 跨风格、新 split、DINOv2-base 基线。
3. **0.4080/0.7007/0.7667/0.3412** 是旧 CLIP base-patch32 管线，只作历史参考。
4. `docs/79` 的 RMSE/WFI/CLIP 旧表与 DINO 表来自不同时期，不要拼接成一张总表。
5. 基线缺 DINO 两列，论文若声称 DINO 优势需补测或明确只比 CLIP/LPIPS。
