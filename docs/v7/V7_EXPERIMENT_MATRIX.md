# WEAVE V7 实验矩阵与决策规则

> 本文件把 `E:\writing-block.md` 的实验设计落成可执行矩阵、Go/No-Go 和记录规范。  
> 所有最终候选必须完整运行 `tasks_750.json`；详见 `V7_EVAL_PROTOCOL.md`。

## 1. 总览

| 阶段 | 输入 | 主要产物 | Go/No-Go |
|---|---|---|---|
| P0 | Q2a / B4 | 相对参数化 750 评测；B4 因果响应 | 决定 P3 是否优先 relative |
| P1 | 15,545 真实画作 latent、R1 | S1/S2 风格能量与 validity | 决定 P2 T2/T3/T4 是否启动 |
| P2 | B4 + P1 best energy | T0–T5 750 评测 + trace | 决定新 Corrector |
| P3 | B4 init + 训练池 | M1/M2 模型与 750 评测 | 决定新 Predictor |
| P4 | P2/P3 有效组件 | C0–C4 组合 | 回答互补性 |
| P5 | 全部 eval | 统计、报告、verdict | 最终结论 |

## 2. P0 矩阵：相对风格变换（不训练）

| ID | 起点 | 输出 | batch | 任务 |
|---|---|---|---:|---|
| `p0_q2a_direct` | Q2a | `F(c(zs), target)` | 8 | 750 |
| `p0_q2a_relative` | Q2a | `zs + F(c(zs), target) - F(c(zs), source)` | 8 | 750 |
| `p0_b4_direct` | B4 | `F(c(zs), target)` | 8 | 750 |
| `p0_b4_relative` | B4 | 同上 relative | 8 | 750 |

### P0 机制诊断

| 输出 | 内容 |
|---|---|
| `p0_b4_causal_audit.json` | 40 张分层源图，B4 五 Style-ID 响应、源风格残留、20 routes 摘要 |
| `p0_style_delta.json` | 同内容五目标输出 latent RMS / CSD 差异 |
| `docs/v7/V7_P0_NOTE.md` | 是否采用 relative 的结论 |

### P0 Go 规则

- **GO_RELATIVE**：relative 在不显著退化 DINO-C / LPIPS 时提升 DINO-S 或目标风格响应；P3 默认采用 relative。
- **NO_RELATIVE**：输出差异很小或无收益；P3 仍可训练 M2，但必须把 relative 作为结构对照，不宣称结构收益。

## 3. P1 矩阵：真实风格能量

| ID | 描述 | 数据源 | 输出 |
|---|---|---|---|
| `S1_sd` | Haar / patch / 协方差 / SWD 统计 | SD-VAE latent 池 | `p1_energy/S1_sd.pt` |
| `S1_a0` | 同 S1，A0 latent 池对照 | A0 latent 池 | `p1_energy/S1_a0.pt` |
| `S2_sd` | R1 64/32/16 特征空间统计 | SD-VAE latent 池 + frozen R1 | `p1_energy/S2_sd.pt` |
| `S2_a0` | 同 S2，A0 latent 池对照 | A0 latent 池 + frozen R1 | `p1_energy/S2_a0.pt` |

### P1 validity

必须对每个候选中至少 200 张真实图像（每风格 40 张）与 40 张生成/扰动样本记录：

| 测量 | 通过条件 |
|---|---|
| 五风格可分性 | 风格间距离 > 同风格内距离；记录 AUROC / margin |
| 内容保持扰动稳定性 | 小扰动后能量变化小于风格间差异 |
| 优化方向一致性 | 沿能量下降做小步 latent 优化时，独立 DINO-S 有提升或显著不退化 |
| 各风格均衡 | 不允许只有 Impressionism / Minimalism 可分 |

产物：`p1_energy/energy_validity.json`，字段 `best_energy`。

### P1 Go/No-Go

- **GO_ENERGY**：至少一个 S1/S2 通过全部 validity；记录 `best_energy`。
- **NO_ENERGY**：能量仅优化自身分数、不能带动 DINO-S；P2 只运行 T0/T1 作为对照，不运行 T2/T3/T4/T5，不调系数。

## 4. P2 矩阵：测试时优化

固定起点 B4；默认 4 步，除 `T5` 外总 RMS 预算一致；所有臂完整 750。

| ID | 风格目标 | 内容保护 | 前置条件 |
|---|---|---|---|
| `T0` | R1 margin | 原 Full Gradient | 必跑 |
| `T1` | R1 margin | 修复后 TopK25 | 必跑；复用/重跑修复实现 |
| `T2` | S1 best | Predictor 锚定 | P1 GO 且 S1 有效 |
| `T3` | S2 best | Predictor 锚定 | P1 GO 且 S2 有效 |
| `T4` | R1 + P1 best | Predictor 锚定 | P1 GO |
| `T5` | T4 | + 源图结构约束 | T4 有正信号 |

若 P1 NO_ENERGY，只保留 T0/T1 并写 `p2_no_energy_verdict.json`。

### 每个 T 臂必须记录

```text
grad_norm.json       # grad_style/content/anchor/r1
optim_trace.json     # 至少 20 个固定任务，逐步 loss/R1/位移/能量
eval.json            # 完整 750
features.pt
run_meta.json        # steps, rms, scale, seed, ckpt, task hash
```

### P2 判定

| 判定 | 条件 |
|---|---|
| `NEW_CORRECTOR` | 在相同 DINO-C 档位下，T2/T3/T4/T5 的 DINO-S 优于 T0，且 Same150 不退化、梯度不失衡 |
| `NO_CORRECTOR` | 真实分布目标不优于 T0，或只提升自身能量/分类器分数 |
| `INCONCLUSIVE` | 梯度失衡 / trace 缺失 / 任务覆盖不足；不得进入 P4 |

## 5. P3 矩阵：Predictor 训练

| ID | 参数化 | 跨风格监督 | 训练 |
|---|---|---|---|
| `M0` | 原 B4 | 冻结基线 | 不训练 |
| `M1` | relative residual | 原 V6 分布损失 | 800–1600 steps |
| `M2` | relative residual | P1 best energy + content + residual + weak R1 | 800–1600 steps |

固定：

```text
init = B4
effective_batch ≈ 240
bf16
c = c1_gentle
styles = 5
same-style identity = structural, not weighted loss
```

### 训练记录

- `train.log`
- `grad_norm.log`（每 100 step）
- `config.json`
- `checkpoint.pt`
- `five_style_probe.json`（同内容五 Style-ID 输出差异）
- `source_leak_probe.json`

### P3 判定

| 判定 | 条件 |
|---|---|
| `NEW_PREDICTOR` | M2 完整 750 超过 B4 与 M1，且五 Style-ID 输出有差异、源风格残留下降、残差 RMS 合理 |
| `STRUCTURE_ONLY` | M1 有收益但 M2 无收益；记录结构对照 |
| `NO_PREDICTOR` | M2 未超过基线；不换更大网络，转查源风格残留、残差强度、梯度 |

## 6. P4 矩阵：联合验证

只在 `NEW_CORRECTOR` 与/或 `NEW_PREDICTOR` 存在时运行必要组合。

| ID | Predictor | Corrector | 运行条件 |
|---|---|---|---|
| `C0` | B4 | 无 | 必跑锚点（可复用 B4 direct） |
| `C1` | B4 | T0 R1 gradient | 必跑锚点 |
| `C2` | B4 | P2 best | `NEW_CORRECTOR` |
| `C3` | M2 | 无 | `NEW_PREDICTOR` |
| `C4` | M2 | P2 best | 两者均存在 |

若 P2 或 P3 NO，不机械运行 C2/C4；写 `p4_skipped.json` 说明原因。

### P4 互补性判据

`C4` 必须相对 `C2` 同时满足：

1. 校正所需累计 latent 位移更小；
2. DINO-C 不下降、LPIPS 不上升；
3. DINO-S 不下降；
4. Same150 不退化。

否则不宣称 Predictor-Corrector 协同，只保留有效单模块。

## 7. P5 统计矩阵

所有 eval 进入统一表：

| 字段 | 来源 |
|---|---|
| `arm_id` | 目录名 |
| `phase` | P0/P2/P3/P4 |
| `eval_path` | `runs/v7/.../eval.json` |
| `cross600` | eval summary |
| `same150` | eval summary |
| `paired_vs_B3` | bootstrap |
| `grad_balance` | `grad_norm.json` |
| `runtime` | log / meta |
| `status` | PASS / PARTIAL / NO / INCONCLUSIVE |
| `include_in_final` | bool + reason |

最终写：

- `runs/v7/final_matrix.json`
- `runs/v7/final_verdict.json`
- `docs/v7/V7_REPORT.md`

## 8. 三问决策树

```text
P1 GO_ENERGY?
 ├─ NO  -> 问题 A = 当前风格描述不足; P2 只做 T0/T1; 不调系数; P4 不做 C2/C4
 └─ YES -> P2 T2/T3/T4/T5
           ├─ NEW_CORRECTOR -> 问题 A = 是; 保留 best corrector
           └─ NO_CORRECTOR  -> 问题 A = 否; 记录负结果

P0/P3 relative / M2?
 ├─ relative 或 M2 降低源风格残留、提高五 Style-ID 差异且 750 指标改善 -> 问题 B = 是
 └─ 否则 -> 问题 B = 否; 下一阶段应转向风格不变内容表示，不继续只调 loss

P4 C4 同时减少校正位移且保持/提升风格内容?
 ├─ 是 -> 问题 C = 是; 形成 Predictor-Corrector 主线
 └─ 否 -> 问题 C = 否; 分别保留有效部分，不包装协同
```

## 9. 风险登记

| 风险 | 触发 | 处理 |
|---|---|---|
| 旧 TopK batch bug 复现 | mask 在 batch 维求均值 | 禁止；使用 `docs/v6/TOPK_FIX_RETEST.md` 修复版 |
| 内容代理不可靠 | R2 `L_ss/L_inv` 类指标与 DINO-C/LPIPS 不一致 | 记录不一致，不宣称内容保护 |
| 梯度失衡 | max/min grad norm > 100 | 预定义 scale 复测一次；仍失衡判 INCONCLUSIVE |
| 缩小集筛选 | 40/200 图上挑候选 | 协议破坏，结果无效 |
| Decoder 混入 | 用 S10 / A0 decoder 评测 | 协议破坏 |
| 研发集过拟合 | 只用 750 反复调参 | 文档声明为研发集内结论，不宣称盲测 |
| 顺序/ batch 依赖 | 改 batch 或任务顺序结果变 | 记录；修复前该臂不得进入最终结果 |
| GitHub 推送失败 | 网络 reset | 本地 commit + bundle；不阻塞实验记录 |

## 10. 最小交付物检查

V7 结束前必须存在：

- `docs/v7/V7_REPORT.md`
- `docs/v7/V7_P0_NOTE.md`
- `runs/v7/final_matrix.json`
- `runs/v7/final_verdict.json`
- `runs/v7/state/*.done`
- 所有最终候选 `eval.json`
- 所有优化臂 `grad_norm.json` 与 `optim_trace.json`
- 所有训练臂 `grad_norm.log` 与 `five_style_probe.json`
- 视觉案例网格：`runs/v7/visual_grid.png`
- 推理开销：`runs/v7/runtime.json`

缺失任一关键产物时，P5 判为 `INCOMPLETE`，不得写 PASS。
