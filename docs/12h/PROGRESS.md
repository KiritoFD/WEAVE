# WEAVE 12 小时实验进度（PROGRESS）

方案原文：[PLAN_12H.md](PLAN_12H.md)（逐字保存，冲突时以方案末尾"补充执行要点"为准）。

## 0. 固定基线（不得中途更换）

| 项 | 值 |
|---|---|
| 评测协议 | `repro_paper`：DINOv2-small + CLIP ViT-B/32 + 750 all-pairs + `ref_policy=first_image_of_target_style_test_dir` + 8-step ODE + AdaIN 2.0 + LPIPS(alex) + fp32 |
| 推理 batch | 固定 32（生成 / metric / decode 均 32；不得中途切换） |
| 复现基线（04:02） | dino_s 0.491502 / dino_c 0.810384 / clip_s 0.712828 / lpips 0.259390 |
| 论文目标 | dino_s 0.4916 / dino_c 0.8103 / clip_s 0.7128 / lpips 0.2593 |
| 参考 `fast_paper_b32` | dino_s 0.491226 / dino_c 0.811184 / clip_s 0.714236 / lpips 0.259333；clip_s 偏 +1.44e-3，**不并入常量** |
| canonical ckpt | `WEAVE/archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt`（禁止移动/覆盖） |
| 判定口径 | 750 四指标 + 600 跨风格 / 150 同风格拆分；FM loss 下降不算成功 |
| 早停 | 诊断/扫描 run 一律 `internal_early_stop_enabled=false`；正式候选 `min_steps>=2000` |
| 代码 SHA | 本地 HEAD `8b4a76a7929e`（远端 WEAVE 副本无 `.git`，注册表用本地 SHA） |

## 1. 阶段总览

| 阶段 | 时间窗 | 内容 | 状态 |
|---|---|---|---|
| P0 | 0:00–0:30 | 诊断代码三修 + 严格 ckpt + schema 两键 + 环境锁定 | ✅ 代码完成（CPU 自测通过），GPU 复跑待空卡 |
| P1 | 0:30–2:00 | A1–A4 小集过拟合；A5–A9 机制诊断 | ⏳ 等 GPU |
| P2 | 2:00–5:00 | T0–T8 训练目标 | ⏳ |
| P3 | 5:00–8:30 | M0–M6 容量；S0–S3 Style Memory；S4–S5 条件注入 | ⏳ |
| P4 | 8:30–10:30 | R1–R6 组合与真实轨迹 | ⏳ |
| P5 | 10:30–12:00 | seed 7/123 复验、完整评测、决策报告 | ⏳ |

## 2. P0 明细

| 项 | 要求 | 结果 | 证据 |
|---|---|---|---|
| A3 监督 | 用 `FlowMatchingObjective._build_training_target()` 的真实 endpoint | ✅ 已改：endpoint / x_t / 速度目标全部来自真实目标构造 | `tools/diagnose_saturation.py::analysis_a3` |
| A2 标签方差 | 改 `t=0`、固定风格条件，测多个候选目标的标签方差 | ✅ 已改：`t=0`（x_t==content，预测只算一次），固定 `style_id`+`style_latent`，Top1/4/8 报 cos 均值/标准差/极值 + 标签两两 cosine | `analysis_a2` |
| A4 真实轨迹 | 调真实 `model.integrate_transport()`，不用系数假 AdaIN | ✅ 已改：4 个模式（net_only / real_ode / last_step_adain / every_step_adain）走真实推理路径 | `analysis_a4` + `_temporary_cfg` |
| 严格 ckpt | 打印 expected/missing/unexpected 三列表，非空即失败 | ✅ `strict=False` 仅用于取三列表，任一非空 `SystemExit`；形状不匹配也失败 | `_load_checkpoint` |
| schema | `stop_after_global_steps` / `save_step_milestones` 声明 + 未知键不进 extra | ✅ 默认 `0` / `[]`；未知键抛 `ValueError`；`start_epoch` 历史键白名单忽略 | `config_schema.py::TrainingConfig` |
| 自测 | smoke + 守护测试 | ✅ smoke（CPU, 526,312 参数）跑通；守护测试全绿 | `archives/20261104/p0_guard_checks.py`、`archives/20261104/diagnose_smoke.json` |

P0 代码改动文件（2 个，均未 commit）：

- `tools/diagnose_saturation.py`：A2/A3/A4 三修 + 严格 ckpt（重写，314 → 约 390 行）。
- `config_schema.py`：`TrainingConfig` 新增两字段、未知键校验、里程碑归一化。

一次性验证脚本（非源码，归档目录）：`archives/20261104/p0_guard_checks.py`、`archives/20261104/scan_training_keys.py`。

## 3. P1-A 受控过拟合（A1–A4）

| ID | 数据规模 | 配对 | 预算 | 状态 | 早停 | 750 四指标 | 600/150 拆分 | 与基线差 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 32 | 固定 | 2,000 steps | ⏳ | — | — | — | — | — |
| A2 | 128 | 固定 | 2,000 steps | ⏳ | — | — | — | — | — |
| A3 | 512 | 固定 | 2,000 steps | ⏳ | — | — | — | — | — |
| A4 | 128 | 每 epoch 重采样 | 2,000 steps | ⏳ | — | — | — | — | — |

约束：microbatch 不得大于子集长度；不得 `drop_last` 导致有效步为空；每 run 记录 L_LL/L_LH/L_HL、cos(v,u)、‖v‖/‖u‖（训练集 + 固定验证样本）。

## 4. P1-B 不训练机制诊断（A5–A9）

| ID | 测量 | 状态 | 结论 |
|---|---|---|---|
| A5 | Top1/4/8 目标速度条件方差 | ⏳ | — |
| A6 | 五个时间分桶 velocity error | ⏳ | — |
| A7 | Teacher-forcing vs 自回归 ODE | ⏳ | — |
| A8 | 固定内容、交换 style ID | ⏳ | — |
| A9 | 真实 ODE / 仅网络 / 不同 AdaIN 位置 | ⏳ | — |

## 5. P2 训练目标（T0–T8）

| ID | 方案 | 600 步快筛 | 2,000 步复验 | 750 四指标 | 判定 |
|---|---|---|---|---|---|
| T0 | 原始 FM（对照） | ⏳ | — | — | — |
| T1 | Cross-only（identity_prob=0） | ⏳ | — | — | — |
| T2 | 20% 真 Identity | ⏳ | — | — | — |
| T3 | Top1 配对 | ⏳ | — | — | — |
| T4 | Top4 配对 | ⏳ | — | — | — |
| T5 | 无桥噪声（bridge_sigma=0） | ⏳ | — | — | — |
| T6 | 早期偏置采样（Logit-Normal μ=-1，须关 legacy compat） | ⏳ | — | — | — |
| T7 | Huber | ⏳ | — | — | — |
| T8 | 内容锚定 HF 端点（α=0.5 / 1.0） | ⏳ | — | — | — |

T8 判据：显著降低训练难度 **且** 提升 DINO-S 才算有效；只降 FM loss 判"目标有问题"。

## 6. P3 容量与条件结构

### P3-A 容量扫描（先 T0 监督，600 步快筛）

| ID | Width | Depth | 状态 | 750 四指标 | 判定 |
|---|---|---|---|---|---|
| M0 | 64 | 4 | ⏳ | — | — |
| M1 | 64 | 8 | ⏳ | — | — |
| M2 | 48 | 12 | ⏳ | — | — |
| M3 | 96 | 4 | ⏳ | — | — |
| M4 | 96 | 8 | ⏳ | — | — |
| M5 | 128 | 4 | ⏳ | — | — |
| M6 | 96 | 6 | ⏳ | — | — |

大模型降低 microbatch、增加梯度累积，保持有效 batch 与总曝光量一致。

### P3-B Style Memory 容量（先把 token 数参数化）

| ID | Memory tokens | 主干 | 状态 | 750 四指标 | 判定 |
|---|---|---|---|---|---|
| S0 | 256 | 64×4 | ⏳ | — | — |
| S1 | 64 | 64×4 | ⏳ | — | — |
| S2 | 32 | 64×4 | ⏳ | — | — |
| S3 | 64 | 96×4 | ⏳ | — | — |

### P3-C 条件注入

| ID | 改动 | 状态 | 判定 |
|---|---|---|---|
| S4 | Style-AdaLN 替代单纯 Cross-Attention | ⏳ | — |
| S5 | Style-conditioned HF velocity head | ⏳ | — |

## 7. P4 组合与真实轨迹（R1–R6）

| ID | 实验 | 启动条件 | 状态 | 判定 |
|---|---|---|---|---|
| R1 | 最佳目标 + 最佳主干 | P2/P3 出现独立收益 | ⏳ | — |
| R2 | HF residual 读取 `h.detach()` | 共享主干梯度竞争明显 | ⏳ | — |
| R3 | epoch 4 后冻结主干，只训风格分支 | 风格早期峰值后退化 | ⏳ | — |
| R4 | 2-step 可微 ODE rollout + 局部 HF 分布损失 | TF/ODE 轨迹差异明显 | ⏳ | — |
| R5 | 4-step rollout 对照 | R4 有效 | ⏳ | — |
| R6 | Route B 独立伪目标质量测试 | T8 证明内容锚定端点有价值 | ⏳ | — |

## 8. P5 最终复验

| 项 | 内容 | 状态 |
|---|---|---|
| seed 复验 | seed 7、123（约 35 min） | ⏳ |
| 完整评测 | 固定 eval 协议 750（约 30 min） | ⏳ |
| 逐风格分析 | 误差归因（约 15 min） | ⏳ |
| 报告 | 200→2,000 steps 指标曲线 + 决策表（约 10 min） | ⏳ |

## 9. 风险与约束

- GPU 当前被评测线 `eval_fast.py`（PID 372854，batch 32，canonical ckpt）占用；空出后立刻进 P1，不等 30 分钟。
- 磁盘事务已移交独立会话；本线不再做任何清理/搬迁。`I:` 可用约 138G。
- 每个 run 必须落：`config.json`、配置 SHA、代码 SHA、seed、有效 batch、曝光量、四指标、`results.json`，并登记 `runs/registry.jsonl`。
- 所有新开关默认关闭、默认数值行为不变；不 commit（git 线统一提交）。
