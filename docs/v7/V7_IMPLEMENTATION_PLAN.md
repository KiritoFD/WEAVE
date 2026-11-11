# WEAVE V7 细化实施方案（P0–P5）

> 约束：单卡 RTX 3060 12GB；16 小时预算；VAE 冻结；所有候选完整运行 Cross600 + Same150。  
> 本文件只规定实施流程，不改评测口径；评估门槛见 `V7_EVAL_PROTOCOL.md`。

## 0. 预检与公共约定

### 0.1 环境

| 项 | 值 |
|---|---|
| 本地工作目录 | `G:\WEAVE` |
| 3060 repo | `/mnt/i/Latent_Style/WEAVE` |
| 产物根目录 | `runs/v7/` |
| 文档根目录 | `docs/v7/` |
| 主评测 | `tools/v6_next/r2_eval.py --tasks eval_protocol/tasks_750.json` |
| 配对统计 | `tools/v6_next/w_paired_compare.py` |
| Python | 3060 WSL `/usr/bin/python3` |
| GPU 排他 | `nvidia-smi --query-compute-apps` 为空才允许启动 |

启动前统一导出：

```bash
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 0.2 冻结资产

以下资产在 V7 中只读：

- `eval_protocol/tasks_750.json`
- `eval_protocol/cache/references_v1_features.pt`
- `runs/v6_r2/cache/tasks750_source_features.pt`
- `runs/v6/P1_B4_distill50/P1_B4_distill50.pt`
- `runs/v6/Q2a_embed_style/Q2a_embed_style.pt`
- `runs/v6_next/probes/r1_r1.pt`
- `runs/v6_r2/eval_B3_distill20/eval.json`（B3 anchor）
- `runs/v6_w/eval_topk_fix/` 下修复后 TopK / Full baseline
- R2 只读归档：`runs/v6_r2/`、`docs/v6/R2_12H_REPORT.md`

### 0.3 单卡串行入口

V7 规划采用一个后台串行入口，阶段完成标志写入 `runs/v7/state/*.done`：

```bash
bash sbflow/scripts/run_v7_16h.sh              # full serial, resume-aware
bash sbflow/scripts/run_v7_16h.sh --dry-run    # compile + unit + input check
```

> 注意：`sbflow/scripts/run_v7_16h.sh` 及其 V7 子脚本目前是**规划路径，尚未创建**。本轮只完成方案细化与边界规定，未启动任何 GPU 任务；脚本实现与 dry-run 预检应在正式启动 V7 时单独完成。

任一阶段失败即停止，不允许自动跳到下一阶段；续跑只重跑缺失/失败阶段，已 `.done` 且产物完整的阶段跳过。阶段预算由 `V7_BUDGET_SEC` 控制，默认 57600 秒。

### 0.4 命名规范

| 类型 | 命名 |
|---|---|
| 相对参数化诊断 | `runs/v7/p0_relative/` |
| 风格能量缓存 | `runs/v7/p1_energy/` |
| 测试时优化臂 | `runs/v7/p2_eval/T0..T5` |
| Predictor 训练臂 | `runs/v7/p3_models/M1..M2` |
| Predictor 评测 | `runs/v7/p3_eval/M1..M2` |
| 联合验证 | `runs/v7/p4_eval/C0..C4` |
| 最终报告 | `docs/v7/V7_REPORT.md` |

每个评测目录必须包含 `eval.json` 与 `features.pt`（若 `r2_eval.py` 正常输出）。

---

## P0：不训练，验证相对风格变换与条件响应

**预算：2h。** 使用现有 Q2a、B4，不修改任何权重。

### P0.1 两种输出

原方案：

```text
z_direct = F(c(zs), target)
```

相对风格方案：

```text
z_relative = zs + F(c(zs), target) - F(c(zs), source)
```

注意：`F` 使用同一个冻结 checkpoint；`source` / `target` 为五类 Style-ID；`c` 固定为 `c1_gentle`；不得为 relative 模式改 canonicalizer 或 Decoder。

### P0.2 完整 750 评测矩阵

| 名称 | 起点 | 输出 |
|---|---|---|
| `p0_q2a_direct` | Q2a | `z_direct` |
| `p0_q2a_relative` | Q2a | `z_relative` |
| `p0_b4_direct` | B4 | `z_direct` |
| `p0_b4_relative` | B4 | `z_relative` |

每组完整 `tasks_750.json`。指标：Cross600 Top-1 / Margin / CSD ΔT / DINO-S / DINO-C / LPIPS / MinOff；Same150 Top-1 / DINO-C / LPIPS。

### P0.3 机制诊断

固定 40 张分层源图（每风格 8 张），对 B4 重做 Style-ID 因果响应：

- 源风格残留比例；
- 五个 Style-ID 输出之间的 latent RMS / CSD 差异；
- 20 个 directed routes 的逐方向 DINO-S / DINO-C；
- 不把 Q2a 的 40 张审计值直接推广到 B4。

产物：

- `runs/v7/p0_relative/*/eval.json`
- `runs/v7/p0_b4_causal_audit.json`
- `docs/v7/V7_P0_NOTE.md`

### P0.4 判定

- 若 relative 在保持 DINO-C/LPIPS 不明显退化时提升目标风格响应：P3 优先采用相对残差参数化。
- 若输出变化太小或无明显收益：记录为「当前 Style-ID 差分不足」，仍进入 P1；不调大网络、不加新 loss。

---

## P1：建立真实画作风格能量

**预算：3h。** 不重新训练 VAE，不引入大型视觉教师，不把 CSD/DINO 用作优化目标。

### P1.1 数据源

- 原始 SD-VAE 训练池：`pseudo_gt_24h/orig_mean_fp16/`
- A0 重编码池：`pseudo_gt_24h/orig_mean_fp16_a0/`
- 五风格严格按现有 `STYLES` 顺序：`Early_Renaissance`, `Impressionism`, `Minimalism`, `Rococo`, `Ukiyo_e`
- 允许把 A0 池作为对照统计量，但**不得改 VAE / evaluator**。

### P1.2 两类风格描述

#### S1：原始 latent 统计

- 多尺度 Haar 分解（至少 1 级，可选 2 级 audit）；
- 每类画派的通道均值、尺度、协方差；
- 局部 patch 统计；
- 可选 sliced Wasserstein 距离。

保留 V6 原有 DWT/patch-SWD 实现作为对照，不能把旧实现包装为新贡献。

#### S2：R1 中间特征统计

- 冻结 `R1FeatureExtractor`；
- 提取 64/32/16 三个尺度；
- 对空间维计算均值、方差、通道相关性；
- 从真实目标画派样本构造分布参照；
- 与 W1 的 Oracle 逐位置特征拟合严格区分：只匹配空间无关的统计量。

产物：

- `runs/v7/p1_energy/S1_sd.pt`, `S1_a0.pt`
- `runs/v7/p1_energy/S2_sd.pt`, `S2_a0.pt`
- `runs/v7/p1_energy/energy_audit.json`

### P1.3 必须先通过的四个测量

1. S1/S2 是否能区分真实五风格；
2. 不同画派间分布距离是否大于同画派内部差异；
3. 对真实图像增加小的内容保持扰动后，风格能量是否稳定；
4. 对生成 latent 做小步优化使能量下降时，独立 DINO-S 是否改善。

产物：`runs/v7/p1_energy/energy_validity.json`。

**Go/No-Go：**

- 若能量只优化自身分数、不能带动独立 DINO-S：P1 判负，不进入 P2 的 T2/T3/T4，不调几十组系数。
- 若至少一个 S1/S2 满足 validity：记录 `best_energy = argmax(独立 DINO-S 带动能力)`，进入 P2。

---

## P2：真实风格分布驱动的测试时优化

**预算：4h。** 固定 B4 为起点，4 步 latent 优化；T5 若增加源图结构约束，步数与总 RMS 预算必须与 T0–T4 保持可比。

### P2.1 实验臂

| 实验 | 风格目标 | 内容保护 |
|---|---|---|
| T0 | R1 margin | 原 Full Gradient |
| T1 | R1 margin | 修复后 TopK25 |
| T2 | 真实风格统计 S1 | Predictor 锚定 |
| T3 | R1 特征统计 S2 | Predictor 锚定 |
| T4 | R1 + P1 最佳真实分布目标 | Predictor 锚定 |
| T5 | T4 | 增加源图结构约束 |

### P2.2 梯度范数审计

每组优化前必须记录各 loss 对 latent 的梯度范数：

```text
grad_style, grad_content, grad_anchor, grad_r1
```

若任一非主导项范数低于主导项 1/100，必须按预定义 scale 调整一次；禁止仅按 loss 数值大小设权重。至少需要产出：

- `runs/v7/p2_eval/T*/grad_norm.json`
- `runs/v7/p2_eval/T*/optim_trace.json`

### P2.3 优化 trace

每个臂记录至少 20 个固定任务（跨五风格均匀）：

- 风格损失；
- 内容损失；
- R1 margin；
- 累计 latent 位移；
- 真实风格特征距离；
- 独立 DINO-S / DINO-C；
- 局部图像损伤 proxy（如 LPIPS 到源图、DWT LL 漂移）。

禁止把 CSD/DINO 作为可微优化目标；它们只用于评测。

### P2.4 完整评测

T0–T5 全部运行完整 `tasks_750.json`，写入 `runs/v7/p2_eval/T*/eval.json`。

成功定义：**在相同 DINO-C 档位下 DINO-S 优于 T0**，且五风格覆盖不退化、Same150 不明显退化。若真实分布目标无效，明确记录其与 V6 旧统计监督的局限，不继续调节大量系数。

---

## P3：训练显式源→目标风格 Predictor

**预算：4h。** 这是本轮最重要训练实验；不再以「重建 Oracle latent」为主要任务。

### P3.1 参数化

```text
z_pred = zs + A_theta(c(zs), target) - A_theta(c(zs), source)
```

- `A_theta` 复用 C3UNet；
- 初始化从 B4 权重出发；
- `source == target` 时由构造保证恒等；
- 训练数据：现有 15,545 张真实画作 latent，五风格均衡采样；
- 有效 batch 约 240，优先 800–1600 steps，bf16；
- 3060 12GB 不够时降低 microbatch、提高 accumulation，不改变有效 batch。

### P3.2 训练目标

```text
L = L_style + λ_c L_content + λ_r L_residual + λ_r1 L_r1_margin
```

- `L_style`：P1 选出的真实分布能量；
- `L_content`：结构一致性代理，必须经过独立 DINO-C / LPIPS / DWT LL 审计；
- `L_residual`：约束 `z_pred - zs` 的 RMS，防止破坏结构换风格统计；
- `L_r1_margin`：弱辅助，避免完全丢失目标类别信息。

训练前记录每个 term 的初始梯度范数；训练中每 100 step 记录一次。

### P3.3 训练矩阵

| 模型 | 参数化 | 跨风格监督 |
|---|---|---|
| M0 | 原 B4 | 冻结，不训练 |
| M1 | 相对风格残差 | 原 V6 分布损失 |
| M2 | 相对风格残差 | P1 最佳真实分布能量 + 内容约束 |

产物：

- `runs/v7/p3_models/M1/M1.pt`
- `runs/v7/p3_models/M2/M2.pt`
- 训练 log、梯度范数 log、checkpoint metadata。

### P3.4 训练后检查

1. 同一内容输入、五种目标 Style-ID 的五个输出差异；
2. 源风格残留比例；
3. 残差 RMS；
4. 训练梯度是否出现某一项主导；
5. 完整 750 评测。

**不能只看 loss 下降。** 若 M2 未超过 M1 或 B4，不换更大网络；转而测量源风格残留、残差强度与训练梯度，记录失败原因。

---

## P4：Predictor 与 Corrector 联合验证

**预算：1.5h。** 只组合前两阶段独立有效的组件。

| 模型 | Predictor | Corrector |
|---|---|---|
| C0 | B4 | 无 |
| C1 | B4 | 原 R1 梯度 |
| C2 | B4 | P2 最佳真实分布校正 |
| C3 | M2 | 无 |
| C4 | M2 | P2 最佳真实分布校正 |

规则：

- 如果 M2 或新 Corrector 未通过各自独立改善判定，不机械运行所有组合；
- 若都有效，C0–C4 全部完整 750；
- 额外记录 C4 相对 C2 的「校正位移是否减少、内容保持是否改善、目标风格匹配是否提高」。

核心问题不是简单串联，而是：

> 更好的初始风格变换，能否减少测试时校正强度，从而同时获得更强风格和更好的内容保持？

---

## P5：统一验收与研究结论

**预算：1.5h。**

### P5.1 统一统计

对所有候选：

- 完整 Cross600 + Same150；
- 配对 clustered bootstrap vs B3；
- 五风格分项、20 个 directed routes；
- 视觉案例网格（固定源图 × 五目标）；
- 3060 实际推理开销（生成时长、峰值显存、每任务耗时）；
- 优化 trace 与梯度范数摘要。

### P5.2 报告

生成：

- `docs/v7/V7_REPORT.md`
- `runs/v7/final_matrix.json`
- `runs/v7/final_verdict.json`
- `runs/v7/final_report_inputs.json`（记录所有被纳入/排除的 eval 路径与原因）

### P5.3 三个问题必须逐条回答

| 问题 | 必须引用 |
|---|---|
| A：真实风格分布能否优于 R1 分类边界 | P1 validity + P2 T0/T2/T3/T4 + 同 DINO-C 档位 DINO-S |
| B：显式源→目标残差是否减少条件捷径 | P0 relative + P3 M1/M2 + 源风格残留 / 五 Style-ID 差异 |
| C：Predictor 与 Corrector 是否互补 | P4 C0–C4 + 校正位移 / 内容 / 风格匹配 |

---

## 启动前检查清单

- [ ] `docs/v7/README.md` 已读；
- [ ] `V7_EVAL_PROTOCOL.md` 门槛确认；
- [ ] `V7_BOUNDARIES.md` 禁止项确认；
- [ ] `runs/v7/state/` 为空或允许续跑；
- [ ] GPU 空闲；
- [ ] `runs/v6_w/eval_topk_fix/` 修复后结果存在；
- [ ] B4 / Q2a / R1 / B3 anchor 路径存在；
- [ ] R2 与 `tasks_750.json` 未修改；
- [ ] 未并发启动其他 GPU 任务。

本轮不启动任何 Ratio Gate、Proximal、Push–Pull 或 Learned GateNet；只有 P1/P2 的真实分布目标和内容代理被独立验证后，后续阶段才允许提出这些扩展。
