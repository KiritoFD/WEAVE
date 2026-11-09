# Negative_Results

**本目录的用途**：集中保存**已证伪的方向、失败的机制、使实验失效的实现缺陷、以及
未带来收益的消融**。放在仓库根目录，与 `docs/` 平级，因为它与"什么有效"同等重要。

**维护规则**

1. 任何被放弃的路线必须在此留档，不能只留在聊天记录或临时日志里。
2. 每条记录必须写清：**假设是什么、怎么测的、结果是什么、为什么失败、什么条件下可能复活**。
3. 数字必须能追溯到 `runs/` 下的产物或对应 `docs/` 报告。
4. **不得**把"没测出来"写成"证明无效"。

---

## 分层

| 层 | 含义 | 判定难度 | 复活成本 |
|---|---|---|---|
| **L1 任务建模** | 优化目标本身与真实任务不一致 | 最难，需要数学论证 | 最高，要改问题定义 |
| **L2 机制** | 在正确目标下，该机制仍无效 | 中，需要对照实验 | 中，可换实现或加预算 |
| **L3 实现缺陷** | 代码与其声称的数学目标不一致 | 易，代码审查即可确定 | 低，修完可重跑 |
| **L4 消融** | 某组件被移除后无差异，或加入后有害 | 易，单因素对照 | 低，直接简化 |

**层级越靠前，越不应该靠调参去救。**

---

## 索引

### L1 任务建模

- [`L1_task_formulation/bridge_objective_ignores_content.md`](L1_task_formulation/bridge_objective_ignores_content.md)
  —— 标准 Schrödinger Bridge 只保证 $T_\#p_i=p_j$，**没有内容项**；
  严格内容保持与完整分布匹配**数学上不相容**。
- [`L1_task_formulation/unconditional_generation_unavailable.md`](L1_task_formulation/unconditional_generation_unavailable.md)
  —— 共享模型在 4000 等效步下**无法无条件生成** Early Renaissance / Rococo / Ukiyo-e，
  因此"向这三种风格做转换"从一开始就不可能成功。

### L2 机制

- [`L2_mechanism/v10_and_v11_arms.md`](L2_mechanism/v10_and_v11_arms.md)
  —— V10 六组 + V11 四组共 10 个机制，**ΔT 全部为负**；Rococo / Ukiyo_e 的 Top-1 恒为 0。
- [`L2_mechanism/edm_analytic_term_dominates.md`](L2_mechanism/edm_analytic_term_dominates.md)
  —— 把网络残差置零（纯 EDM 解析统计项）就能复现全部失败形态；
  学习残差只提高**相对排名**，不改善**分布距离**。
- [`L2_mechanism/distance_similarity_anticorrelation.md`](L2_mechanism/distance_similarity_anticorrelation.md)
  —— ΔT 与 SWD 系统性反相关，由**多样性塌缩**解释；只报其中一个会得出相反结论。

### L3 实现缺陷

- [`L3_implementation/six_code_defects.md`](L3_implementation/six_code_defects.md)
  —— 六处使 V10 结论失效的确定性代码错误（含 SWD 梯度被切断、评测口径错误等）。

### L4 消融

- [`L4_ablation/v6_component_ablations.md`](L4_ablation/v6_component_ablations.md)
  —— V6 已有 9 个变体 + 本轮补齐的 8 个单因素消融（进行中）。

---

## 一句话总览：什么被证伪了

| 被证伪的说法 | 证据 | 层 |
|---|---|---|
| "独立随机端点 + Bridge MSE 足以完成风格迁移" | V10/V11 十组 ΔT 全负 | L1 |
| "把源分布运输到目标分布就等于风格迁移" | 内容-分布不相容定理 + 实测 ΔS 大降而 ΔT 不升 | L1 |
| "加大空间分辨率（patch4）能救困难风格" | 01→04 仅 +1.0pp，Rococo/Ukiyo 仍为 0 | L2 |
| "在线 α-DSBM 能修正终点分布" | e1 最保守、ΔT 最差；忠实实现复现同形态 | L2 |
| "分布监督（SWD）能改善风格分布" | d2：ρ=0.80、L_dist 降 65%，CSD-SWD 不降 | L2 |
| "EDM 之外的网络残差是主要贡献" | κ=0 复现全部失败形态 | L2 |
| "指标已接近天花板所以看不出差异" | SWD 地板实测 0.0111，实测 0.027（2.4 倍） | — |
| "Rococo 失败是因为样本少" | Rococo 2,519 张，**第二大类** | — |

## 什么仍然成立

- **V6 D1 直接算子是仓库里唯一 ΔT 为正的路线**：+0.0323（`c1_gentle`），
  DINO-C 0.9167，LPIPS 0.3658，4/5 风格改善。
- V6 的"内容规范化 + 条件生成"与 V11 理论最终指向的方向一致，**不是巧合**。
- 评测口径本身可用：SWD 地板 0.0111、真实图 style Top-1 0.98，都有分辨力。
