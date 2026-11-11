# WEAVE V7 评估协议与成功门槛

> 本文件定义 V7 的**唯一评测口径**。任何候选进入报告或对比表，都必须满足本协议。  
> 本协议不改变历史 R2/W/P0–P3 结果，只约束 `runs/v7/` 新实验。

## 1. 固定协议

| 项 | 规定 |
|---|---|
| 任务集 | `eval_protocol/tasks_750.json` |
| 规模 | Cross600 + Same150；**不得使用 Val200/Test200/子集筛选** |
| 参考特征 | `eval_protocol/cache/references_v1_features.pt` |
| 源特征缓存 | `runs/v6_r2/cache/tasks750_source_features.pt` |
| 原始 RGB 零点 | `paper_rgb_v1` / raw RGB source zero point |
| Decoder | 统一 TRT fp16 SD-VAE 解码；VAE 冻结 |
| Evaluator | `tools/v6_next/r2_eval.py` |
| 配对统计 | `tools/v6_next/w_paired_compare.py` / 150 source clusters |
| 基线 anchor | B3 `runs/v6_r2/eval_B3_distill20/eval.json` |
| B4 对照 | `runs/v6_r2/eval_B4_distill50/eval.json` |
| 修复后 P2 对照 | `runs/v6_w/eval_topk_fix/full_b8/eval.json`、`fixed_topk25_b8/eval.json`、`fixed_topk50_b8/eval.json`、`fixed_topk25_s8_b8/eval.json` |
| R2 归档 | 只读，不修改 |

每个候选输出目录必须包含：

- `eval.json`
- `features.pt`
- 若涉及测试时优化，还应有 `optim_trace.json` 与 `grad_norm.json`
- 若涉及训练，还应有训练 log 与 checkpoint metadata

## 2. 主指标

### 2.1 Cross600

| 指标 | 定义 / 用途 |
|---|---|
| Top-1 | CSD 五类目标风格分类准确率；辅助指标，不是单独成功标准 |
| Margin | CSD target-vs-other margin |
| CSD ΔT | 相对原始源图的目标风格分数变化 |
| DINO-S | 与目标风格参考池的 DINOv2 CLS 相似度；主风格指标 |
| DINO-C | 与源图 DINOv2 CLS 相似度；主内容指标 |
| LPIPS | 与源图的感知距离；主内容/局部损伤指标 |
| MinOff | Off-target Minimalism attractor rate；风格覆盖缺陷 |

### 2.2 Same150

| 指标 | 定义 / 用途 |
|---|---|
| Same Top-1 | 同风格身份保持 |
| Same Margin | 同风格分类 margin |
| Same DINO-C | 同风格内容一致性 |
| Same LPIPS | 同风格重建/漂移 |
| Same dS | 同风格风格偏移 |

### 2.3 分风格与方向覆盖

所有最终候选必须报告：

- 五目标风格逐类 Top-1 / DINO-S / DINO-C / LPIPS；
- 20 个 directed routes 的 Top-1 / DINO-S / DINO-C / LPIPS；
- Minimalism、Ukiyo_e、Early_Renaissance 单独检查，防止只靠 Impressionism/Minimalism 拉平均；
- 源风格残留比例（固定分层 40 张审计，且必须针对当前候选，不得复用 Q2a 结果代替 B4/M2）。

## 3. 配对统计

所有相对 B3 的提升必须使用 paired clustered bootstrap：

- cluster = 150 个 source clusters；
- 输出 95% CI；
- 对 Top-1、DINO-S、DINO-C、LPIPS 分别报告 Δ 与 CI；
- 每个 eval 的 `by_id` 必须与 B3 task-id 集合一致；
- 不允许只看均值差，不允许改变 task-id 对齐；
- 若 task 集合不一致，该臂不得进入主对比表，只能列为 protocol-mismatch。

## 4. P4.3 成功门槛（沿用 V6/W）

| 指标 | 要求 |
|---|---|
| DINO-S | 相对 B3 ≥ +0.020，且配对 CI 下界 > 0 |
| DINO-C | ≥ 0.8125 |
| LPIPS | ≤ 0.4712 |
| CSD Top-1 | 不低于原设容忍范围（相对 B3 ΔTop-1 ≥ -0.02） |
| 风格覆盖 | 不允许只改善 Impressionism / Minimalism；20 routes 不得出现系统性崩坏 |
| Same150 | 不得出现明显额外退化；需与 B3/B4 和修复后 Full 对比 |
| 统计 | 研发集内配对显著性；不宣称盲测 |

### 4.1 判定等级

| 等级 | 条件 |
|---|---|
| PASS | 所有主门槛同时满足 |
| PARTIAL | DINO-S CI 下界 > 0 但未达到 +0.020，或 DINO-C/LPIPS/Top-1 有一项不足 |
| NO | 任一主要指标明显退化，或 DINO-S 无显著提升 |

> 修复后 TopK25/50 的正确判定是 PARTIAL / 未通过主门槛；此结论作为 V7 的历史输入，不得被重新包装成 PASS。

## 5. 等 DINO-C 档位比较

V7 P2 的核心问题不是 Top-1，而是：

> 在相同内容保持水平下，真实分布目标是否比 R1 分类边界更接近目标风格分布？

因此 P2 必须额外报告 **DINO-C 档位表**：

| 档位 | DINO-C 范围 | 候选 | DINO-S | LPIPS |
|---|---|---|---|---|
| C0 | [0.74, 0.76) | T0/T1/T2/T3/T4/T5 | best / mean | best / mean |
| C1 | [0.76, 0.78) | ... | ... | ... |
| C2 | [0.78, 0.80) | ... | ... | ... |
| C3 | [0.80, 0.82) | ... | ... | ... |

同档位比较必须按同一 source task 配对；若某个臂在档位内没有任务，不得跨档位补平均。

## 6. 梯度与目标审计

凡涉及可微优化目标（P1/P2/P3），必须记录：

- `grad_style_norm`
- `grad_content_norm`
- `grad_anchor_norm`
- `grad_r1_norm`
- 各 loss 的初始尺度与每 100 step / 每步优化的统计；
- 若最大梯度/最小梯度 > 100，必须标为 `imbalanced`，该臂不得作为最终 PASS 候选，除非有预定义归一化并复测通过；
- 禁止把 CSD / DINO / LPIPS 指标直接作为可微优化目标。

## 7. 推理开销

所有最终候选必须记录 3060 实测：

| 项目 | 要求 |
|---|---|
| 750 任务生成墙钟 | r2_eval log / profiler |
| 峰值显存 | `torch.cuda.max_memory_allocated` 或 nvidia-smi 采样 |
| 每任务耗时 | wall / 750 |
| 优化步数 | 明确 T/P/C 臂 |
| 解码方式 | 统一 TRT fp16 SD-VAE |
| 模型参数量 | Predictor / Corrector / auxiliary stats |

## 8. 视觉案例

固定 10 张源图（每风格 2 张）× 5 个目标 Style-ID 输出一张网格，用于检查：

- 内容结构是否保留；
- 是否出现目标画派局部纹理/笔触；
- 是否产生 classifier-hacking 式不自然纹理；
- Minimalism attractor 与 Ukiyo_e 漏检。

视觉案例**只作机制诊断**，不得替代数值门槛。

## 9. 研发集限制

- 750 任务长期用于研发，所有显著性为研发集内配对结论；
- 不宣称独立盲测；
- 论文级结论需未来作者隔离盲集；
- 文档必须明确写出：`Val200 ⊂ Test750 / Test750 repeated use`。

## 10. 报告最低内容

`docs/v7/V7_REPORT.md` 至少包含：

1. P0–P5 各阶段 Go/No-Go；
2. 所有候选完整 Cross600 / Same150 表；
3. 相对 B3 的配对 bootstrap；
4. 五风格 + 20 directed routes；
5. 等 DINO-C 档位表；
6. 梯度范数审计；
7. 推理开销；
8. 视觉案例说明；
9. 问题 A/B/C 的明确回答；
10. 失败实验与排除原因；
11. 研发集限制声明。

## 11. 不允许的口径变化

以下任何行为都会使候选失效：

- 修改 `tasks_750.json`；
- 改 reference features 或 source cache；
- 换 Decoder / VAE；
- 用 CSD/DINO/LPIPS 作为训练或测试时优化的可微目标；
- 只在 40/200 图上筛候选，再把选中臂跑 750；
- 仅引用旧 TopK 未修复结果作为 P2 证据；
- 把 loss 下降当成风格提升；
- 把 Q2a 的源风格残留比例直接推广到 B4/M2；
- 触碰 `runs/v6_r2/` 或 `docs/v6/R2_12H_REPORT.md`。
