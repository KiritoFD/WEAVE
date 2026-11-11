# WEAVE V7 阶段文档导航

> 来源：`E:\writing-block.md`（用户 2026-11-11 拍板的下一阶段方案）  
> 设备：单卡 RTX 3060 12GB  
> 时间预算：16 小时  
> 研究对象：WEAVE 主模型；**VAE 保持冻结**  
> 评测：所有候选完整运行 `tasks_750.json` = Cross600 + Same150

## 本阶段定位

V6/W/P0–P3 的结论已经明确：

- 换 A0 latent 池不解决跨风格迁移；
- 直接蒸馏 Oracle 或拟合匹配画作 latent 都不形成竞争力；
- R1 推理期梯度有效，但优化目标是五类分类边界，不等于真实画派分布；
- 修复 batch 共享掩码后，TopK 的真实 DINO-S 增益为 +0.0187 / +0.0185，**不再达到 +0.020 主门槛**，不能作为主研究结论。

因此 V7 不再围绕 “怎样沿 R1 梯度走得更聪明” 做扫描，而转向：

> **WEAVE：面向无配对艺术风格迁移的轻量化、分布驱动、测试时自适应 latent 变换。**

核心定义：**Amortized Style Transport + Instance-Adaptive Refinement**，即「摊销式风格变换 + 实例自适应校正」。

1. Predictor 学习源→目标风格的基本变换，给出内容保持较好的初始结果；
2. Corrector 用真实目标画派分布 + 当前源图结构做少步 latent 优化；
3. 小模型允许多次梯度更新，不要求一次前向决定所有局部编辑。

## 文档清单

本次按 `E:\writing-block.md` 新增/细化的核心文档：

| 文档 | 作用 |
|---|---|
| `V7_IMPLEMENTATION_PLAN.md` | P0–P5 细化实施步骤、输入/产物、脚本、预算、判定 |
| `V7_EVAL_PROTOCOL.md` | 评估协议、指标定义、P4.3 门槛、统计与报告要求 |
| `V7_BOUNDARIES.md` | 冻结项 / 可变项 / 禁止项 / 失败处理 / 研发集限制 |
| `V7_EXPERIMENT_MATRIX.md` | 实验矩阵、命名规范、Go/No-Go、梯度范数审计、开销记录 |

### 历史存档（不属于本轮 writing-block 方案）

以下文件在本轮之前已存在于 `docs/v7/`，属于早期 V7/ScoreNet/RDMD/DMD 路线记录，**不是本次 16h 方案的执行依据**，也不得与本轮 P0–P5 编号混淆：

- `P0_TWEEDIE_SCAN_REPORT.md`
- `P1_PURE_MSE_SCORENET_REPORT.md`
- `V7_DMD_FIX_EVALUATION_REPORT.md`
- `V7_DMD_FIX_ROUTE_REPORT.md`
- `V7_HYBRID_EVALUATION_REPORT.md`
- `V7_RDMD_EVALUATION_REPORT.md`
- `V7_ROUTE_REPORT.md`
- `V7_STABLE_EVALUATION_REPORT.md`

本轮只以 `V7_IMPLEMENTATION_PLAN.md`、`V7_EVAL_PROTOCOL.md`、`V7_BOUNDARIES.md`、`V7_EXPERIMENT_MATRIX.md` 四份文档为准。

## 阶段预算

| 阶段 | 预算 | 核心研究 |
|---|---:|---|
| P0 | 2h | 验证跨风格条件响应与相对残差参数化（不训练） |
| P1 | 3h | 从真实画作建立风格分布能量 S1/S2 |
| P2 | 4h | 测试时真实分布引导优化 T0–T5 |
| P3 | 4h | 训练显式源→目标风格 Predictor M0/M1/M2 |
| P4 | 1.5h | Predictor 与 Corrector 联合验证 C0–C4 |
| P5 | 1.5h | 统一统计、机制诊断、报告 |
| **合计** | **16h** | |

## 本轮必须回答的三个问题

- **问题 A**：真实风格分布能否提供比 R1 分类边界更好的优化方向？
- **问题 B**：显式源→目标残差建模能否减少原有条件捷径？
- **问题 C**：Predictor 和 Corrector 是否真正互补？

回答方式见 `V7_EXPERIMENT_MATRIX.md` 的 Go/No-Go 决策树。三个问题的答案都必须来自完整 750 任务 + 配对 bootstrap，不得来自 loss 曲线或缩小子集。

## 与 V6 成果的接口

| 冻结基线 / 资产 | 路径 |
|---|---|
| B4 起点 | `runs/v6/P1_B4_distill50/P1_B4_distill50.pt` |
| Q2a 起点 | `runs/v6/Q2a_embed_style/Q2a_embed_style.pt` |
| R1 critic | `runs/v6_next/probes/r1_r1.pt` |
| 统一评测器 | `tools/v6_next/r2_eval.py` |
| 配对比较 | `tools/v6_next/w_paired_compare.py` |
| 修复后 TopK 基线 | `docs/v6/TOPK_FIX_RETEST.md`、`runs/v6_w/eval_topk_fix/` |
| 评估任务 | `eval_protocol/tasks_750.json` |
| R2 封存只读 | `runs/v6_r2/`、`docs/v6/R2_12H_REPORT.md` |

V7 产物统一写入 `runs/v7/` 与 `docs/v7/`，不覆盖 `runs/v6_w/` 或 `runs/v6_r2/` 任何既有结果。

## 当前状态

**仅完成方案细化与边界规定，未启动任何 GPU 任务。** 启动入口与预检要求见 `V7_IMPLEMENTATION_PLAN.md` 的「单卡串行入口」。
