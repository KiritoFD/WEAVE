# V6 下一阶段 · 执行方案

**来源计划**：`E:\V6_NEXT_PHASE_STYLE_LEARNING_PLAN.md`（103 行，用户提供）
**本文档**：把该计划落到本仓库的**具体可执行形式** —— 哪些资产已存在、哪些要新建、每个闸门的判据、预算与取消顺序
**日期**：2026-11-10
**状态**：P0 / P1 已完成，P2 待启动

---

## 0. 起点与边界

### 0.1 已确认的起点事实

| 项 | 值 | 来源 |
|---|---|---|
| Direct 最小模型 | **L11_k1024_dim16，217,824 参数**（A0 的 11.6%） | `V6_CONCLUSIONS.md` |
| Val200 | ΔT +0.0421 / DINO-C 0.9216 / LPIPS 0.3403 / 5/5 | 同上 |
| **Heldout400** | ΔT +0.0244 / **Top-1 0.028** | P0.1 |
| **Same150** | ΔS +0.0209 / **Top-1 0.987** | P0.1 |
| ScoreDiff（历史） | Heldout400 **Top-1 0.367** / ΔT +0.0669 / **DINO-C 0.5687** | P0.2 |

**核心问题不是"V6 能不能做风格迁移"，而是"Direct 当前训练目标无法建立目标风格身份"。**
证据：同一模型在 Same150 上 Top-1 0.987，在 Heldout400 上 0.028 —— 35 倍差距，
差异只在于"是否要求改变风格"。

### 0.2 不在本阶段范围

- 不重训 ScoreDiff（已有 ckpt 作为强风格对照）
- 不改 VAE、不改评测协议、不改五风格/20 方向设定
- 不做两风格模型（严格五风格等权）
- 不以"超过 20% 随机"作为成功判据（见 §3.3）

---

## 1. 已有资产（可直接复用，不重建）

| 类别 | 资产 | 路径 |
|---|---|---|
| Direct 算子 | `C3UNet`（direct / flow 两模式，含 style_keys） | `tools/v6/models_v6.py` |
| Direct 训练/评测 | 全参数化 CLI + eval200 + 条件审计 | `tools/v6/train_eval_v6.py` |
| ScoreDiff | `ConditionalScoreNet` + `edit_with_score_difference` + ckpt | `tools/v6/score_diff.py`、`runs/v6/score_diff/score_diff_net.pt` |
| 规范化器 | `ContentCanonicalizer`（c1 / c1_gentle / c1_soft / c2 / c2_zca / none） | `tools/v6/canonicalize.py` |
| 参考统计 | 1024/风格缓存（**必须用这个，不是 Core64**） | `runs/v6/cache/indomain_style_targets_v2_k1024.pt` |
| 评测层级 | Val200 / Heldout400 / Same150 / Screen50 任务文件 + 统一评测器 | `runs/v6_24h/*_tasks.json`、`tools/v6_24h/eval_tiers.py` |
| 统计工具 | 源图聚类配对 bootstrap | `tools/v6_24h/stats.py` |
| 串行队列 | 排他锁 + 超时 + 状态落盘 + 配置快照 | `tools/v6_24h/queue.py` |
| 无监督诊断 | 流形检查、风格条件效力、可视化网格 | `tools/v6_24h/{csd_manifold_check,style_condition_efficacy,make_grid}.py` |
| **新阶段工具** | 几何方向诊断、R1 critic、梯度闸门 | `tools/v6_next/{eval_direction,probe_style,gate}.py` |

**结论：不需要新建训练框架。** 新阶段主要是**新增损失项**与**新评测口径**。

---

## 2. 阶段划分与预算

| 阶段 | 内容 | 预算 | 训练臂 | 状态 |
|---|---|---|---:|---|
| **P0** | 统一复评 + 几何诊断 | 3 h | 0 | ✅ **完成** |
| **P1** | latent 风格判别器 R1 + 梯度闸门 | 4 h | 1（判别器） | ✅ **完成** |
| **P2** | 生成器监督机制 G0–G6 | 6 h | 3–4 | ⏳ **待启动** |
| **P3** | 乘法旋钮 M1–M3 | 4 h | 0 | 未开始 |
| **P4** | 最终评测 | 5 h | 0 | 未开始 |
| **P5** | QA / 文档 | 2 h | 0 | 未开始 |
| **合计** | | **24 h** | | |

**实测校准**：Direct 单臂训练 10–19 min（dim16 为 12.0 min），评测约 2 min。
**P2 的 3–4 个臂约需 1 小时训练**，预算充裕。

---

## 3. 共同协议

### 3.1 固定不变的部分

| 项 | 值 |
|---|---|
| 风格与方向 | 5 风格 / 20 个有向转换，batch 按目标等权 |
| VAE | SD1.5 VAE（`stabilityai/sd-vae-ft-ema`，scaling 0.18215） |
| 规范化器 | `c1_gentle` |
| 内容注入 | 多尺度（64/32/16 全开） |
| λ_same / λ_s / λ_c | 0.25 / 0.35 / 0.525 |
| 参考统计 | **1024/风格** |
| 种子 | 42（主）+ 43（复核） |

### 3.2 主基准与对照

| 角色 | 模型 | 用途 |
|---|---|---|
| **结构基点** | L11_dim16（218K） | 所有新机制的起点 |
| **独立源图锚点** | L8_dim32（845K） | 已有 Heldout400 结果，作宽度对照 |
| **风格强度参照** | ScoreDiff N=24 | Pareto 前沿的右端点 |
| **历史重现** | A0（1.88M） | 只复评，不重训 |

### 3.3 成功判据（**不是"超过 20% 随机"**）

按计划 §2 的要求：

1. **主指标**：在相近内容档位（DINO-C ≥ 0.85 / 0.75 / 0.65）比较 **目标 Top-1、Target Margin**
2. **ΔT 仅辅助**，不再以 `styles_improved = 5/5` 判定成功
3. **Rococo / Ukiyo / Minimalism 必须单列**，不允许被整体平均掩盖
4. **必须报告 5×5 混淆矩阵的 Minimalism 列** —— P0.2 显示这是主要失败模式
5. 若使用 CSD 蒸馏，最终评价必须加入**独立特征 + 真实风格分布 + 盲图**
6. **同一 CSD 不能既是唯一教师又是唯一裁判**

### 3.4 分层任务

| 层级 | 规模 | 用途 | 使用次数记录 |
|---|---:|---|---|
| Smoke16 | 16 | 数学/代码正确性 | 每次改动 |
| Screen50 | 50 | 粗选旋钮 | 不限 |
| **Val200** | 200 | 参数选择 | 记录 |
| **Heldout400** | 400 | 候选确认 | **记录，禁止反复调参** |
| Same150 | 150 | 自一致 | 记录 |

---

## 4. P0：已完成（详见 `V6_NEXT_RESULTS.md` §1–5）

| 子项 | 结果 |
|---|---|
| P0.1 L11 独立评测 | Heldout400 ΔT +0.0244（五模型最高），Top-1 0.028；Same150 Top-1 0.987 |
| P0.2 ScoreDiff 复现 | Top-1 0.367，但 DINO-C 0.5687、**Minimalism 占 98.8%** |
| P0.3 几何诊断 | `cos_align = +0.127`（伪指标 0.30 高估 2.4 倍），`near_not_crossed` 仅 0.070 |

---

## 5. P1：已完成（详见 `V6_NEXT_RESULTS.md` §8–9）

| 子项 | 结果 |
|---|---|
| R1 判别器 | 816K 参数，**balanced top-1 0.875**，ER/Roc/Uki recall 0.814/0.834/0.937 |
| 对照 | DWT 矩只有 0.555；**无 Minimalism 吸引子**（仅吸收 17/4705） |
| 梯度闸门 | 真实 CSD margin 单调改善 **+0.2489**（缺口 88%），`cos(g, V6) = 0.094` |
| **Pareto** | **严格优于 ScoreDiff**（同 Top-1 下 DINO-C 高 0.11） |

---

## 6. P2：生成器监督机制（待启动）

### 6.1 候选臂（按计划 §5）

| 臂 | 机制 | 依赖 | 优先级 |
|---|---|---|---|
| **G0** | 续训对照（不加新损失，同预算） | 无 | **必做**（保底） |
| **G1** | **margin ranking 约束**：用冻结 R1 critic 的 margin 作训练信号 | R1（已完成） | **必做** |
| G4 | ScoreDiff 蒸馏：把 ScoreDiff 的输出当伪目标 | ScoreDiff（已有） | 备选 |
| G2 | 分布匹配（SWD/MMD）在 latent 空间 | R1 特征 | 低（V11 已证传导失败） |
| G3 | 对比学习（正负样本对） | R1 | 低 |
| G5 | 多任务（分类 + 重构） | R1 | 低 |
| G6 | 风格键增强（S6 的扩展） | 无 | 低（S6 已证收益 0.001–0.002） |

### 6.2 G1 的具体实现

在现有损失上加一项：

$$L_{\rm margin} = -\log\sigma\big(m_t(z_{\rm out}) / \tau\big),
\quad m_t = \text{logit}_t - \operatorname{logsumexp}_{k\neq t}\text{logit}_k$$

其中 critic 冻结、`z_out` 为 Direct 输出、`τ` 为温度。

**关键设计约束**（防 metric hacking）：
- critic **必须冻结**，且其训练集与 Heldout400 的源图不相交
- 训练时**不允许**把 CSD 加进反传
- 每个臂都必须报告 Minimalism 列的混淆矩阵

### 6.3 G1 的准入条件（已在 P1 满足）

| 条件 | 要求 | 实测 |
|---|---|---|
| critic 分类能力 | balanced top-1 ≥ 0.75 | **0.875** ✅ |
| 困难目标 recall | 每类 ≥ 0.50 | **0.814** ✅ |
| **梯度指向真实 margin** | 真实 CSD margin 上升 | **+0.2489** ✅ |
| 与现有方向不冗余 | 与 V6 方向夹角大 | cos **0.094** ✅ |

### 6.4 P2 闸门

| 判据 | 阈值 |
|---|---|
| 相对 G0 的 Top-1 提升 | ≥ +5pp（配对聚类 CI 排除 0） |
| 不出现 2/5 以上目标退化 | 分目标 ΔT 与 margin |
| 内容不跌破档位 | DINO-C ≥ 0.85（保真）/ 0.75（均衡） |
| **Minimalism 列不膨胀** | 非对角 Minimalism 计数不超过 G0 的 1.2 倍 |
| 独立复核 | Heldout400 + Same150 |

---

## 7. P3：乘法旋钮（未开始）

| 旋钮 | 内容 |
|---|---|
| M1 | Direct 的 α 插值（已实现，`m0_alpha.py`，含每目标标定） |
| M2 | ScoreDiff 的 N / lr / anchor（已在 P0.2 跑过 2 个配置） |
| M3 | 内容切向投影（可选） |

---

## 8. 交付物

| 文件 | 状态 |
|---|---|
| `docs/v6/V6_NEXT_STYLE_PLAN.md`（本文件） | ✅ |
| `docs/v6/V6_NEXT_P0_CORRECTIONS.md` | ✅ |
| `docs/v6/V6_NEXT_RESULTS.md` | ✅ |
| `docs/v6/V6_NEXT_DECISIONS.md` | ✅ |
| `runs/v6_next/{probes/,checkpoints/,eval_*/,grids/}` | 部分（probes 已有） |
| `Negative_Results/L4_ablation/v6_next_verified_negatives.md` | 待 P2 后 |

**代码**：`tools/v6_next/{eval_direction,probe_style,gate}.py` ✅
**脚本**：`sbflow/scripts/run_v6_next_{p01,p02,direction,r1,gate}.sh` ✅

---

## 9. 取消顺序（计划 §8）

若时间不足，**按此顺序取消**：
R3 → G2 → G6 → M2。

**G0 与至少一条 G1/G4 路线必须保留。**

---

## 10. 风险与已知限制

| 风险 | 说明 | 缓解 |
|---|---|---|
| **R1 梯度太弱** | 每维 1e-4，需 eps≈3.5 才把 margin 从 −0.28 推到 −0.03 | 训练而非重复单步；或加大 critic 容量的梯度尺度 |
| **单步探针 ≠ 训练** | P1 的 Pareto 优势是单步的，训练后可能不同 | P2 必须与 G0 同预算对照 |
| **Minimalism 吸引子** | ScoreDiff 的 98.8% 集中在 Min | 每个臂报告 Min 列；G1 用 margin 而非 accuracy 作损失 |
| **Heldout400 已多次使用** | 不能称"全新盲测" | 记录使用次数；如有条件建新 source-heldout |
| **参考集来自训练池** | 绝对 ΔT 乐观约 40% | 只做臂间比较，不宣称泛化 |
| **R1 与 Direct 训练集重叠** | R1 用全部训练 latent，Direct 也用它 | critic 冻结 + 源图不相交的评测集，但**存在间接泄漏风险，需在 P2 记录** |
