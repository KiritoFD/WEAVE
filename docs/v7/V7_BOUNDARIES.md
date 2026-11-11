# WEAVE V7 边界规定：冻结项、可变项、禁止项

> 本文件是 V7 的硬边界。凡与本文件冲突的实验、报告或脚本改动，默认视为协议破坏，结果不得进入最终对比表。

## 1. 冻结项（只读，不得修改）

### 1.1 数据与评测

- `eval_protocol/tasks_750.json`
- `eval_protocol/cache/references_v1_features.pt`
- `runs/v6_r2/cache/tasks750_source_features.pt`
- `paper_rgb_v1` 原始 RGB 零点
- 统一 TRT fp16 SD-VAE 解码
- B3 anchor：`runs/v6_r2/eval_B3_distill20/eval.json`
- R2 只读归档：`runs/v6_r2/`、`docs/v6/R2_12H_REPORT.md`
- 修复后 P2 基线：`runs/v6_w/eval_topk_fix/`

### 1.2 模型 / 资产

- VAE：冻结，不训练、不替换；S10 仅作为历史 decoder 加速成果，不进入 V7 主模型。
- Q2a：`runs/v6/Q2a_embed_style/Q2a_embed_style.pt`
- B4：`runs/v6/P1_B4_distill50/P1_B4_distill50.pt`
- R1 critic：`runs/v6_next/probes/r1_r1.pt`
- canonicalizer：`c1_gentle` 及既有 `runs/v6/cache/canonicalizer_stats.pt`
- 五风格顺序：`Early_Renaissance`, `Impressionism`, `Minimalism`, `Rococo`, `Ukiyo_e`

### 1.3 历史结论

- 修复后 TopK50 / TopK25 S8 的 DINO-S 增益为 +0.0187 / +0.0185，**未达到 +0.020 主门槛**；
- `full`、`hf`、`content`、`style_energy` 不受旧 `topk_spatial_mask()` bug 直接影响；
- V1/V2 A0 latent swap 未超过 B3/B4；
- P3 RealPair FM 同口径评测为 NO；
- Q2a 40 张审计结果不能直接推广为 B4/M2 统计量。

## 2. 可变项（允许探索）

### 2.1 P0：输出参数化

允许比较：

```text
z_direct   = F(c(zs), target)
z_relative = zs + F(c(zs), target) - F(c(zs), source)
```

不得改 `F` 权重、`c`、`zs`、Decoder。

### 2.2 P1：真实风格统计

允许构造：

- S1：原始 latent 的 Haar / patch / 协方差 / SWD 统计；
- S2：冻结 R1 中间特征的空间无关统计；
- SD-VAE 池与 A0 池作为统计对照。

不允许：

- 重新训练 VAE；
- 引入大型视觉教师或 CLIP/CSD/DINO 风格能量；
- 把 CSD/DINO/LPIPS 当作可微优化目标；
- 把 W1 的 Oracle 特征逐位置拟合改名为「真实分布」。

### 2.3 P2：测试时优化

允许优化项：

```text
L_real_style, L_content, L_anchor, weak R1 margin
```

允许固定 B4 起点、4 步 latent 优化；T5 若增加结构约束，必须保持总 RMS / 步数与 T0–T4 可比。

必须对梯度范数做审计；不得靠 loss 数值比例盲设权重。

### 2.4 P3：Predictor 训练

允许训练：

```text
z_pred = zs + A_theta(c(zs), target) - A_theta(c(zs), source)
```

允许从 B4 初始化；允许 M1 使用原 V6 分布损失作结构对照；允许 M2 使用 P1 最佳真实分布能量 + 内容约束 + 弱 R1。

不得：

- 使用 Oracle latent 重建作为主任务；
- 改有效 batch 来规避 3060 显存约束；可以降 microbatch + 增 accumulation；
- 只看 loss 下降判断成功；
- 未做五 Style-ID 输出差异诊断就宣称学到跨风格变换。

### 2.5 P4：组合

允许组合 P2 与 P3 中**各自独立有效**的模块。若 M2 或 Corrector 未通过独立判定，不机械运行所有组合。

## 3. 不允许的扩展（本轮明确暂停）

在 P1/P2 的真实分布目标和内容代理没有被独立验证前，禁止叠加：

- Ratio Gate；
- Proximal / pullback；
- Push–Pull；
- Learned GateNet；
- 展开优化网络；
- 新 VAE 或 decoder 训练；
- 新 CSD/DINO/CLIP 教师；
- TopK 比例/步数继续扫描；
- 针对 TopK 的更多后处理 trick。

原因：TopK batch-mask bug 已经证明，归因不清时叠加模块会污染实验解释。

## 4. 禁止项（协议破坏）

以下行为会使结果无效：

1. 修改 `tasks_750.json` 或任何参考/源特征缓存；
2. 用 Val200/Test200/40 图子集筛选候选，再跑 750 作为验证；
3. 改变 Decoder、VAE、evaluator 的指标定义；
4. 把 CSD/DINO/LPIPS 用作可微优化目标；
5. 并发启动两个 GPU 任务（单卡排他）；
6. 重训或修改 W/R2/VAE-swap 已完成臂；
7. 覆盖 `runs/v6_w/`、`runs/v6_r2/` 结果；
8. 把旧 TopK 未修复结果作为 P2 成功证据；
9. 把 Q2a 审计值直接当作 B4/M2 的源风格残留；
10. 不记录梯度范数就把多 loss 结果写成 「分布引导有效」；
11. 将 loss 下降、自身能量下降视为风格提升；
12. 将研发集显著性写成盲测结论。

## 5. 失败处理

| 情况 | 动作 |
|---|---|
| 任一阶段 `.done` 缺失且无产物 | 允许续跑该阶段 |
| 显存 OOM | 降低 microbatch / 增加 accumulation；不得改有效 batch 或 tasks |
| 梯度范数失衡 >100 | 标记 `imbalanced`，按预定义 scale 复测一次；仍失衡则判负 |
| P1 能量无法带动独立 DINO-S | 不进入 T2/T3/T4；记录为风格描述失败 |
| P2 真实分布目标无效 | 不调几十组系数；保留 T0/T1 对照并写负结果 |
| M2 未超过 M1/B4 | 不换更大网络；测量源风格残留、残差强度、训练梯度 |
| P4 组合无互补 | 分别保留有效部分，不包装协同收益 |
| GitHub 推送失败 | 本地提交 + git bundle，网络恢复后再推；不阻塞本地记录 |

## 6. 单卡与预算边界

- 设备：RTX 3060 12GB；
- 总预算：16h；
- 默认 `V7_BUDGET_SEC=57600`；
- 每阶段必须有 `state/*.done`、`log`、关键产物；
- 超过阶段预算未完成则停止并报告具体阶段，不自动继续；
- 所有 750 评测统一 batch=8，除非记录 OOM 并显式使用 batch=1 对照；不得混用 batch 作为筛选手段。

## 7. 研发集与论文边界

- `tasks_750.json` 已长期参与研发，所有统计是研发集内配对结果；
- 不宣称盲测、泛化或 SOTA；
- 论文级验证需要作者隔离盲集；
- 本阶段最多宣称：**在既有研发协议下，某种真实风格分布目标 / 相对残差参数化 / Predictor-Corrector 组合是否优于固定基线**。

## 8. 文档与代码边界

V7 新增/修改只允许落在：

- `docs/v7/`
- `runs/v7/`
- `sbflow/scripts/run_v7_16h.sh`（及 V7 专用子脚本）
- `tools/v7/`（新增模块）
- 必要时在 `tools/v6_next/r2_eval.py` 增加 V7 generator mode，但不得改既有模式输出

禁止修改历史报告以掩盖错误；只能追加更正说明并注明旧结论作废。
