# A06：20 条跨风格迁移方向诊断（2026-11-20）

> **TASK**: A06（Slot A）｜**STATUS**: DONE（5 臂逐任务 5×5 路线矩阵 + 1 臂如实降级）
> 工具：[`tools/v25_cpu/a06_route_diagnostics.py`](../../../tools/v25_cpu/a06_route_diagnostics.py)（一键复算：`python -m tools.v25_cpu.a06_route_diagnostics`，CPU-only，`OMP_NUM_THREADS=2`，matplotlib Agg，无 ssh/CUDA）
> 数值：[`a06_route_diagnostics.json`](a06_route_diagnostics.json)（全量：逐臂 20 路线 × top1/margin/dino_s/dT/dino_c/lpips/lock/预测落点分布）；图：[`a06_route_heatmap.png`](a06_route_heatmap.png)
> 口径：750 任务协议，读数 cross600（20 路线 × 30 任务）；`target_top1` = CSD referee argmax == 目标画派；**源锁定率** = cross 任务上 referee argmax == 源画派（`pred_eq_src`，逐任务真值字段，非近似）；`dT` = 输出对目标画派的 CSD 原型分 − 源图基线（`src/eval/measure.py` L100，风格增益）。

## 〇、结论速览（分级）

| # | Finding | 分级 |
|---|---|---|
| F1 | **`Min→Uki` 是全场共同最弱路线**（5 臂全部 bottom-1：top1 0.000–0.033，源锁定 0.90–1.00——输出几乎全部停留在 Minimalism）；反向 `Uki→Min` 反而是最强路线（3 臂 1.000）——**极端方向不对称** | confirmed |
| F2 | 跨臂共性失效呈两条轴：**`*→ER`**（`Imp→ER` 5/5 锁定 0.97、`Roc→ER` 5/5、`Min→ER` 4/5）与 **`Min→*`**（`Min→Uki` 5/5、`Min→Roc` 4/5、`Min→ER` 4/5）。ER 是最难目标（逐目标 top1 .15–.23），Minimalism 是最难源（逐源 top1 .11–.23） | confirmed |
| F3 | **源锁定率与 Top-1 反向绑定**：PRISM-1 lock .487 / Top-1 .443，D6 .342/.557、cs_j .345/.550、B_t066 .353/.550、Hybrid g55 .420/.527——PRISM-1 的 Top-1 劣势主要是"改不动源外观"，不是判别器偏好 | confirmed（观测性） |
| F4 | A05 F4 路线级证实：Ukiyo-e 是**最难目标**（入 Uki 4 路线 top1 .00–.23）但**是好源**（出 Uki 4 路线 .23–1.00，`Uki→Min` 1.0）——"质心孤立"只伤害**生成端**，不伤害条件端 | confirmed（复算一致） |
| F5 | A05 F3 修正为**单向泄漏**：`ER→Roc` 全臂 .77–.87（易），`Roc→ER` 仅 .07–.13 且 40–67% 锁回 Rococo（离开 Rococo 的那部分也多数落入 Impressionism .23–.53 而非 ER）——"最相似对"不是互相混淆，是 Roc→ER 单向拉回 | confirmed（修正性） |
| F6 | A01 Minimalism 守卫触发路线级定位：cs_j vs full 的 Min 目标 −7.5pp 分解为 `Roc→Min` −13.3pp、`Imp→Min` −10.0pp、`ER→Min`/`Uki→Min` 各 −3.3pp——**full 分支的贡献集中在 Roc/Imp 源进入 Minimalism 的判别** | confirmed（复算逐位一致） |
| F7 | `O3_rgb_cnx_gated_t55`（One-Shot）**无逐任务记录**（全仓只有聚合 JSON），路线分解不可能、也不伪造——只按聚合口径入表（csd_top1 .2400 / dino_c .6783 / dino_s .5025 / lpips .3494），图上占位 | data-boundary（如实降级） |

## 一、数据边界（必须先读）

- 逐任务臂（750 行 × `src_style_id`/`tgt_style_id`/`pred_class`/`pred_eq_src` 齐全，20 路线 × 30 任务校验通过）：`M0_prism1_l100`、`M0_v20_d6`、`M1_lin_g55`（`runs/v24/measured/`）；`P02_d6h_csj`、`B_d6_t066`（`runs/v25/measured/`）。
- **`O3_rgb_cnx_gated_t55` 降级**：`runs/v24` 与 `runs/v25` 的 `measured/` 均无该臂目录；唯一的数字在 [`runs/v25/r3_oneshot_minimal.json`](../../../runs/v25/r3_oneshot_minimal.json) `reused` 块（复用 V24 聚合值）与 `runs/v24/p0_o1_o3.json`（聚合列表）。**没有逐任务行，就没有路线矩阵；本报告不构造任何"估计的"源→目标分解。**
- 任务清单原列 `M0_prism1_l100`（v24 重测）与 A05 引用的 v19 `V19_S2_prism1_v2` 是同一模型族的重测口径；两者逐目标 Top-1 最大差 0.87pp（§三），路线级读数可安全用于 V25 决策。

## 二、逐臂最弱路线 Top-5 与总览

| 臂 | cross Top-1 | 源锁定率 | dT 风格增益 | same150 盲 Top-1 | 最弱路线 Top-5（top1） |
|---|---:|---:|---:|---:|---|
| `M0_prism1_l100` | 44.33% | **48.67%** | 0.107 | 96.67% | Min→Uki .000 / Imp→ER .033 / Roc→Uki .067 / Imp→Uki .100 / Roc→ER .133 |
| `M0_v20_d6` | 55.67% | 34.17% | 0.141 | 96.67% | Min→Uki .033 / Imp→ER .033 / Roc→ER .067 / Min→Roc .067 / Min→ER .133 |
| `M1_lin_g55` | 52.67% | 42.00% | 0.150 | 96.67% | Min→Uki .000 / Imp→ER .033 / Roc→ER .100 / Min→Roc .100 / Min→ER .100 |
| `P02_d6h_csj` | 55.00% | 34.50% | 0.130 | 96.67% | Min→Uki .000 / Imp→ER .033 / Roc→ER .100 / Min→Roc .100 / Min→ER .133 |
| `B_d6_t066` | 55.00% | 35.33% | 0.130 | 96.67% | Min→Uki .033 / Imp→ER .033 / Roc→ER .100 / Min→Roc .133 / Min→ER .200 |

- PRISM-1 的最弱 5 条里有两条是 **Uki 目标**（Roc→Uki .067、Imp→Uki .100），D6 系臂的 Uki 目标相对好（cs_j 的 Roc→Uki .533）——PRISM-1 对孤立目标的生成短板比 D6 族更尖锐。
- Hybrid g55 的 dT 风格增益最高（.150）但 Top-1 低于 D6/csj：线性混合提高了风格分但不提升判别命中，与 04 文档曲线读法一致。

## 三、跨臂共性失效路线（bottom-5 重合）

| 路线 | 入 bottom-5 臂数 | 失效形态（预测落点，5 臂范围） |
|---|---|---|
| `Min→Uki` | **5/5**（全部 bottom-1） | 锁定源 .90–1.00，输出停 Minimalism；Uki 目标完全不动 |
| `Imp→ER` | **5/5** | 锁定源 **.97（五臂完全一致）**，输出停 Impressionism |
| `Roc→ER` | **5/5** | 锁回 Rococo .40–.67；离开者多数落入 Impressionism（.23–.53）而非 ER |
| `Min→Roc` | 4/5 | 锁定源 .77–.87 |
| `Min→ER` | 4/5 | 锁定源 .70–.87 |

**两条结构性轴**：
1. **ER 目标轴**（`Imp→ER`、`Roc→ER`、`Min→ER`）：ER 是最难目标（逐目标 top1：ER .15–.23 << Min .78–.93、Imp .60–.89）。ER 早期文艺复兴的外观签名（素描性、低饱和）与源差距最小、referee 判别裕度低，所有引擎都"改不到 ER"。
2. **Minimalism 源轴**（`Min→Uki`、`Min→Roc`、`Min→ER`）：Minimalism 是最难源（逐源 top1：Min .108–.225 << Uki .608–.858）。极简源的强平整外观主导输出，参考风格压不过它；唯一例外 `Min→Imp`（.37–.60）。

**方向不对称范例**：`Uki→Min` 1.000 / `Min→Uki` 0.000（PRISM-1、Hybrid、cs_j）；`Roc→Min` 1.000（D6）/ `Min→Roc` .067（D6）。路线矩阵不是对称的相似度表——**生成端存在强方向性**，与 V10.2 "Separation Margin ≠ Generative Direction" 铁律在路线层面再次印证。

## 四、与已知结论交叉验证

### 4.1 A05 F4（Ukiyo 质心最孤立但 PRISM-1 最差目标）——路线级解释：孤立性只伤害生成端

`M0_prism1_l100` 复算逐目标 Top-1 `[ER .150, Imp .600, Min .783, Roc .583, Uki .100]` vs A05 引用的 v19 实测 `[.142, .600, .792, .583, .100]`——最大差 0.87pp（v24 重测一致性通过）。路线分解：

- **入 Uki**（目标侧）：`Min→Uki` .000、`Roc→Uki` .067、`Imp→Uki` .100、`ER→Uki` .233——一致地差；
- **出 Uki**（源侧）：`Uki→Min` 1.000、`Uki→Roc` .700、`Uki→Imp` .500、`Uki→ER` .233——除→ER 外全部良好。

**解释线索**：Ukiyo-e 的 795D 质心孤立（A05 F4，最大质心余弦 +0.02）意味着没有任何其它画派的输出会"顺路"落进 Uki 半空间——**条件端**读 Ukiyo 源没问题（甚至因独特而好读），但**生成端**要从非 Uki 源造出 Uki 外观需要跨越最大的几何距离，而 PRISM-1 的单步前馈（17ms）没有足够的位移能力。逐目标锁定率佐证：入 Uki 的源锁定 .733（PRISM-1）——输出留在源画派。

### 4.2 A05 F3（ER–Rococo 质心最相似 +0.42）——修正为单向拉回

| 臂 | ER→Roc top1 | Roc→ER top1 | Roc→ER 锁回 Rococo | Roc→ER 落入 Imp |
|---|---:|---:|---:|---:|
| prism1 | .867 | .133 | .57 | .30 |
| d6 | .767 | .067 | .40 | .53 |
| g55 | .867 | .100 | .67 | .23 |
| csj | .867 | .100 | .40 | .50 |
| t066 | .833 | .100 | .40 | .50 |

"最相似对"**不是**互相混淆：ER→Roc 是全矩阵最易路线之一。真正的失效是 **Roc→ER 单向**：输出要么留在 Rococo（质心相似 → 小位移就"够近"），要么过冲到 Impressionism（中间调色地带）。路线级教训：**质心相似度预测的是"双向都难或都易"，实测是"单向易、单向难"——迁移方向性不能由特征空间几何推断**（同 V10.2 铁律）。

### 4.3 A01（Minimalism 守卫：cs_j vs full −7.50pp）——路线级定位

复算逐目标与 A01 报告**逐位一致**：csj `[.225, .892, .800, .533, .300]`、full `[.225, .883, .875, .492, .283]`。Minimalism 目标的 7.5pp 差分解：

| 路线 | cs_j | full | Δ |
|---|---:|---:|---:|
| `ER→Min` | .933 | .967 | −3.3pp |
| `Imp→Min` | .500 | .600 | **−10.0pp** |
| `Roc→Min` | .800 | .933 | **−13.3pp** |
| `Uki→Min` | .967 | 1.000 | −3.3pp |

**解释线索**：full CFG 分支的价值集中在 **Rococo/Impressionism 源 → Minimalism** 的判别裕度（这两条路线 cs_j 只有 .50/.80）。Minimalism 目标整体易（锁定率仅 .05–.175），但装饰性强的源（Roc/Imp）需要更大位移才能被 referee 判为极简——full 分支的额外 NFE 恰好买了这部分位移。若未来想删 full 分支，守卫应针对这两条路线单独设阈值，而不是只看最差类均值。

### 4.4 O3 One-Shot（聚合口径，无路线分解）

`O3_rgb_cnx_gated_t55`：csd_top1 **.2400**、dino_c .6783、dino_s .5025、lpips .3494、style_affinity .6039（`runs/v25/r3_oneshot_minimal.json` reused 块）。聚合 Top-1 .24 低于全部 5 个逐任务臂（.443–.557）；A05 F5 已指出其 SG-5 ER 参考失位（→Min，margin −0.11）。若未来 4090 侧为 O3 臂补 `measure` 逐任务产物，本工具的臂清单加一行即可复算路线矩阵（接口已就绪）。

## 五、热力图

[`a06_route_heatmap.png`](a06_route_heatmap.png)：2×3 分面，5 个逐任务臂各一格 5×5（行=源、列=目标，格=路线 Top-1%，灰对角=same150 盲 Top-1），第 6 格为 O3 占位（明确标注 NO PER-TASK ROWS）。读图要点：第二列（→Imp）与第三列（→Min）几乎全绿，第一列（→ER）与第五列（→Uki）大面积红——两个"难目标"列；Minimalism 行（第三行）是唯一的整行偏红源。

## 六、科学结论

**SCIENTIFIC_CONCLUSION: confirmed（路线层结构成立，且与 A01/A05 交叉验证逐位一致）**——

1. 750 协议下跨风格迁移失败**不是臂特异的，是路线结构性的**：5 臂共享同一组失效路线（`Min→Uki`/`Imp→ER`/`Roc→ER` 5/5，`Min→Roc`/`Min→ER` 4/5），且失效形态高度一致（源锁定 0.9+ 或质心相似方向的单向拉回）。
2. **源锁定率（cross 上 referee argmax==源）是比 dino_s 更直接的"改不动"指标**：它解释了 PRISM-1（lock .487）与 D6 族（lock .34–.35）的 Top-1 差距，且五臂 same150 盲测全部 96.67%——身份保持与风格位移的权衡集中在 cross 侧。
3. A05 两条几何结论在路线层被**精化而非否定**：质心孤立（Uki）= 生成端难、条件端易；质心相似（ER–Roc）= 单向易、单向难。特征空间几何**不能**替代配对路线评测做可编辑性决策——但路线矩阵可以（本轮即为该决策提供了首个全量路线级台账）。

## 七、局限与风险

1. `O3_rgb_cnx_gated_t55` 无逐任务数据，One-Shot 路线的路线级失效模式**未知**（已如实降级，未伪造）；其聚合 Top-1 .24 的路线归因只能等 GPU 侧补 measure 产物。
2. 路线 n=30，Top-1 的二项标准误 ~±9pp（0.5 处）；单条路线的 ±1 个任务差异（3.3pp）**不是**显著差异——跨臂比较只应看 5/5、4/5 这种结构性重合与 >10pp 的差，不要对单格数字过度解读。
3. `M0_prism1_l100`（v24 重测）与 v19 原测逐目标最大差 0.87pp——若与 v19 文档数字对账请以各自 run 的 eval.json 为准，不要混用。
4. `pred_eq_src` 是 CSD referee 的 5 类 argmax 口径的源锁定，不是像素级身份；dino_c 高的臂（PRISM-1 .744）天然更容易被判回源画派，lock 与内容保持部分纠缠（F3 的"反向绑定"是观测性结论，非因果）。
5. 逐路线预测落点分布只覆盖 referee 的 5 类 argmax；referee 内部 score（`csd_scores_5`）未做路线级裕度分析（JSON 已存逐路线 margin 均值，可后续深挖）。

## 八、复算

```bash
OMP_NUM_THREADS=2 python -m tools.v25_cpu.a06_route_diagnostics   # JSON + PNG，~10 s
python -m pytest tests/test_no_absolute_paths.py tests/test_markdown_links.py -q
```

## 九、交付块（agent2.md §十二）

- **TASK**: A06 — 20 个跨风格迁移方向分析
- **STATUS**: DONE（5 臂全量 5×5 路线矩阵 + 共性失效 + 交叉验证；O3 臂如实降级为聚合口径）
- **FILES_CHANGED**: `tools/v25_cpu/a06_route_diagnostics.py`（新增）、`docs/v25/audits/a06_route_diagnostics.json`（新增）、`docs/v25/audits/a06_route_heatmap.png`（新增）、`docs/v25/audits/a06_route_report.md`（新增，本文件）
- **KEY_RESULTS**: 每臂最弱路线均为 `Min→Uki`（0.000–0.033，锁源 0.90–1.00）；共性失效 `Imp→ER`/`Min→Uki`/`Roc→ER`（5/5 臂 bottom-5）+ `Min→ER`/`Min→Roc`（4/5）；PRISM-1 lock .487 vs D6 族 .34–.35 解释其 Top-1 劣势
- **TESTS**: `python -m tools.v25_cpu.a06_route_diagnostics` 退出码 0；护栏 pytest 两文件全绿（见下）
- **SCIENTIFIC_CONCLUSION**: confirmed（见 §六）
- **RISKS**: 见 §七（O3 边界、路线 n=30 噪声、lock↔dino_c 纠缠）
- **NEXT_TASK_SUGGESTION**: A07/A08 若做"最佳候选继续运行"，建议把 **`Roc→Min`/`Imp→Min` 路线裕度**作为 P02 full-vs-csj 决策的守卫指标（§4.3），并在 4090 侧为 O3 臂补逐任务 measure 以解锁 One-Shot 路线矩阵（工具已预留一行式扩展）。
