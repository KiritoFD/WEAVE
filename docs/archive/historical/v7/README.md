# WEAVE V7 / V7-R 阶段文档导航

> **当前主报告**：[V7R_NEXT_BREAKTHROUGH.md](<V7R_NEXT_BREAKTHROUGH.md>)。
> V7-R 已完成历史运行；CPT/NEXT 尚处 P0 可信度审计，不能依据旧总控 FINISHED 宣称完成。
> 保留 R3c/R4c 的测试时编辑结果；N1/N2 学习式 Corrector 失败。目标特异性新证据与历史结论修正在主报告中。
> 原始全量结果/配对统计/源码归档：[实验证据索引](<../../deliver/v7r_snapshot/manifest.json>)。
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

本目录只保留本次按 `E:\writing-block.md` 新增/细化的 V7 方案文档：

| 文档 | 作用 |
|---|---|
| `README.md` | 导航、定位、预算、与历史资产接口 |
| `V7_IMPLEMENTATION_PLAN.md` | P0–P5 细化实施步骤、输入/产物、脚本、预算、判定 |
| `V7_EVAL_PROTOCOL.md` | 评估协议、指标定义、P4.3 门槛、统计与报告要求 |
| `V7_BOUNDARIES.md` | 冻结项 / 可变项 / 禁止项 / 失败处理 / 研发集限制 |
| `V7_EXPERIMENT_MATRIX.md` | 实验矩阵、命名规范、Go/No-Go、梯度范数审计、开销记录 |

V7 只以这五份文档为准；早期 V7/ScoreNet/RDMD/DMD 路线报告已按要求清理，不保留在 `docs/v7/`。

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

**V7 全部阶段已完成（2026-11-12 00:31–01:36，3060 单卡 tmux 串行，实际耗时约 65 分钟）。**

- 串行入口：`sbflow/scripts/run_v7_16h.sh`（已实现；`--dry-run` 预检通过：编译 + 19/19 CPU 单测 + 冻结输入检查；全程 resume-aware）
- 子脚本：`tools/v7/`（v7_common / p0_causal_audit / p1_energy / p1_validity / p2_corrector / p3_train / p4_combine / p5_report / p5_visual_grid / v7_tests）
- `r2_eval.py` 仅追加 `--gen relative` 与 `--gen v7` 两个 generator mode，既有模式输出不变

### 最终结论（详见 `V7_REPORT.md` / `runs/v7/final_verdict.json`）

| 阶段 | 判定 |
|---|---|
| P0 相对参数化 | **NO_RELATIVE**（B4 relative Top-1 +10.7pp 但 DINO-C −0.038、DINO-S −0.009） |
| P1 真实风格能量 | **GO_ENERGY**，`best_energy = S2_sd`（AUROC 0.983；S1 未过均衡性/稳定性，作对照） |
| P2 测试时优化 | **NO_CORRECTOR**（T2–T4 内容大幅改善 DINO-C 0.85 vs T0 0.745，但 DINO-S 未超过 T0；T5 无正信号未跑） |
| P3 Predictor | **NO_PREDICTOR**（M1 恒等坍缩 Top-1 0.022；M2 风格响应增强 Top-1 0.268 但内容崩坏 DINO-C 0.50） |
| P4 组合 | C2/C3/C4 按规则跳过（`p4_skipped.json`）；C0/C1 锚点复用 |

**问题 A = 否**（真实分布能量有效可分但不能在等 DINO-C 下胜过 R1 边界的 DINO-S）；**问题 B = 否**（相对残差未减少条件捷径，下一阶段应转向风格不变内容表示）；**问题 C = 无结论**（C4 未运行）。最佳臂为 T1（修复后 TopK25，PARTIAL，与 V6 结论一致）。

### 运行记录

- 运行日志：`runs/v7/logs/`；阶段标志：`runs/v7/state/*.done`（6 阶段全部完成）
- 预定义 scale 策略（P2，协议 §6）：分量梯度按 step-0 实测 RMS 定标 + 组合目标回溯线搜索（α ∈ {1,…,1/32}）；smoke 实测 S2 能量 7.07→3.39、margin 上升、内容无损
- 开销：单次 750 评测 88–180 s；M1/M2 训练 736/856 s（有效 batch 240，峰值显存 1.8/2.8 GiB）
