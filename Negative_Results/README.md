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
- [`L1_task_formulation/delta_t_does_not_imply_recognizability.md`](L1_task_formulation/delta_t_does_not_imply_recognizability.md)
  —— **ΔT 为正不等于生成图被识别为目标风格。**
  V6 24h 全部臂的目标风格 Top-1 只有 **2.5%–10%**，而随机水平是 **20%–25%**，
  即比随机低 4–8 倍；同时 ΔT 全部为正（+0.017 到 +0.044）。
  ΔT 只衡量"朝目标原型的余弦位移"，不要求最终落在该原型的 Voronoi 区域内。
  **因此 ΔT 绝不能单独报告，必须同时给 Top-1。**

### L2 机制

- [`L2_mechanism/v10_and_v11_arms.md`](L2_mechanism/v10_and_v11_arms.md)
  —— V10 六组 + V11 四组共 10 个机制，**ΔT 全部为负**；Rococo / Ukiyo_e 的 Top-1 恒为 0。
- [`L2_mechanism/edm_analytic_term_dominates.md`](L2_mechanism/edm_analytic_term_dominates.md)
  —— 把网络残差置零（纯 EDM 解析统计项）就能复现全部失败形态；
  学习残差只提高**相对排名**，不改善**分布距离**。
- [`L2_mechanism/distance_similarity_anticorrelation.md`](L2_mechanism/distance_similarity_anticorrelation.md)
  —— ΔT 与 SWD 系统性反相关，由**多样性塌缩**解释；只报其中一个会得出相反结论。
- [`L2_mechanism/operator_is_a_style_remover.md`](L2_mechanism/operator_is_a_style_remover.md)
  —— **V6 的算子是"去风格器"而非"换风格器"。**
  17 臂实测：源风格相似度 ΔS **−0.10 到 −0.15**（实质移动），
  但目标 ΔT 只有 **+0.03**，即 \|ΔT\|/\|ΔS\| ≈ **0.30** ——
  **70% 的位移是"破坏源风格而未获得目标风格"**。
  因此输出仍有 **91.4%** 被识别为源风格、仅 3.1% 为目标风格。
  方向纯度（比值）比位移幅度更能预测 ΔT：N5 位移最大（ΔS −0.154）但纯度最低（0.11），ΔT 也最低。

### L3 实现缺陷

- [`L3_implementation/six_code_defects.md`](L3_implementation/six_code_defects.md)
  —— 六处使 V10 结论失效的确定性代码错误（含 SWD 梯度被切断、评测口径错误等）。

### L4 消融

- [`L4_ablation/v6_component_ablations.md`](L4_ablation/v6_component_ablations.md)
  —— V6 已有 9 个变体 + 本轮补齐的 8 个单因素消融（进行中）。
- [`L4_ablation/v6_24h_cancelled_additions.md`](L4_ablation/v6_24h_cancelled_additions.md)
  —— **S5 HF 结构损失被预实验取消**：与现有 DWT 矩冗余 **0.921**，
  区分力更低（0.503 vs 0.555），且把困难目标拉得更近（1.22 vs 2.58）。
  附带发现：DWT 空间里 **Ukiyo 与 ER 只差 0.60**，解释困难目标的天花板。
- [`L4_ablation/redundant_path_ablation.md`](L4_ablation/redundant_path_ablation.md)
  —— **冗余多路组件的单因素消融不可归因。**
  S1/S2/S3 单层关闭的 ΔT 变化为 −0.0014 / +0.0017 / −0.0018（三层和 −0.0015），
  但全关（A1）为 **+0.0151**，交互残差 **+0.0166**。
  多尺度内容注入是**冗余容错通路**，任一层都够用，单层消融时其余层补偿。

---

## 一句话总览：什么被证伪了

| 被证伪的说法 | 证据 | 层 |
|---|---|---|
| "独立随机端点 + Bridge MSE 足以完成风格迁移" | V10/V11 十组 ΔT 全负 | L1 |
| "把源分布运输到目标分布就等于风格迁移" | 内容-分布不相容定理 + 实测 ΔS 大降而 ΔT 不升 | L1 |
| "加大空间分辨率（patch4）能救困难风格" | 01→04 仅 +1.0pp，Rococo/Ukiyo 仍为 0 | L2 |
| "在线 α-DSBM 能修正终点分布" | e1 最保守、ΔT 最差；忠实实现复现同形态 | L2 |
| "分布监督（SWD）能改善风格分布" | d2：ρ=0.80、L_dist 降 65%，CSD-SWD 不降 | L2 |
| "V6 的变换近似恒等" | ΔS −0.11 是实质移动；真问题是方向纯度 0.30 | L2 |
| "去掉某一层内容注入能定位 A1 的收益" | 三层和 −0.0015 vs 全关 +0.0151，残差 +0.0166（冗余） | L4 |
| "ΔT 是 Top-1 的有效代理" | 秩相关仅 0.214；需 6.6× ΔT 才到随机水平 | L1 |
| "ΔT +0.008 是显著进展" | 小于 VAE 往返漂移 0.0113，属噪声级 | L1 |
| "加 DWT 统计类损失能救困难目标" | HF 特征与 DWT 矩冗余 0.921，且更差 | L4 |
| "λ_c 越大越好" | 0.525→0.75 只多 0.0012（饱和） | L4 |
| "Core64 采样课程有价值" | 去掉后 ΔT 差 −0.0002，自一致完全相同 | L4 |
| "EDM 之外的网络残差是主要贡献" | κ=0 复现全部失败形态 | L2 |
| "指标已接近天花板所以看不出差异" | SWD 地板实测 0.0111，实测 0.027（2.4 倍） | — |
| "Rococo 失败是因为样本少" | Rococo 2,519 张，**第二大类** | — |

## 什么仍然成立

- **V6 D1 直接算子是仓库里唯一 ΔT 为正的路线**：+0.0323（`c1_gentle`），
  DINO-C 0.9167，LPIPS 0.3658，4/5 风格改善。
- V6 的"内容规范化 + 条件生成"与 V11 理论最终指向的方向一致，**不是巧合**。
- 评测口径本身可用：SWD 地板 0.0111、真实图 style Top-1 0.98，都有分辨力。
