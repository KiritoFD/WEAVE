# V25 收敛计划（Minimal PRISM）— 修订版（前 2 小时）

> **与初版（原 `docs/19_pull_together.md`）的差异**：本版落实用户 2026-11-19 审查意见的四处修正——
> **C1** `tau_v ≤ 0.78` 截断修正（§3.1）；**C2** 等效判定改为预注册非劣性边界（§2）；
> **C3** `O3` vs `O3_label_free` 非单变量，归因实验后置到机制 C（§4.1）；**C4** 训练配方↔CFG 有效分支对应规则（§3.2 Round 2）。
> 冻结结论表（初版 §一）不变，不再重复。

---

## 1. 已冻结结论（不再重复实验）

继承初版 §一 全表：Core 数据筛选（80%/全池/K8 等效）、Layout 固定策略、**彻底淘汰 32×32 分支**、不拆解三元 Gate、关闭 1 步 SWD 残差网（M3/M4）、暂停 1M 外挂蒸馏（M5/M6/O4D）、混合默认线性（M2 Haar 判负）、锁定 One-Shot `O3` 结构（`Q/K` 灰度 DINO + `V` RGB ConvNeXt + Oklab 门控）。

## 2. 统一判定规则（修正 C2：非劣性，不是"CI 跨零"）

对每个待删组件，在 `cross600`（源图聚类 `n_clusters=150`, `n_boot=4000`）上计算 **简化版 − 复杂版** 的逐任务配对差 $\Delta_m$ 的 95% CI：

- **预注册容忍带**：$\Delta_{\text{DINO-C}} \ge -0.010$，$\Delta_{\text{DINO-S}} \ge -0.005$，$\Delta_{\text{LPIPS}} \le +0.010$（LPIPS 越低越好）。
- **删除条件**：三个指标的配对 CI **整体落入**容忍带（即 `lo(Δ_DINO-C) ≥ −0.010`、`lo(Δ_DINO-S) ≥ −0.005`、`hi(Δ_LPIPS) ≤ +0.010`），且五画派最差类 Top-1 单类跌幅 ≤ 3pp、`same_blind` 不退化。满足 → **强制删除**复杂组件。
- **保留条件**：复杂版在任一风格指标上的 CI 下界**超出**容忍带上界（如 `lo(Δ_complex−simple, DINO-S) > +0.005`）→ 保留。
- **不可判定**（CI 与容忍带边界重叠且不满足删除条件）→ 默认保留复杂组件，并在文档中标注"未决"。
- 工具：`src/eval/metrics.py::bootstrap_source_ci`。主目标参考线不变：`DINO-C ≥ 0.70–0.73, DINO-S ≥ 0.49–0.51, LPIPS ≤ 0.51, < 100 ms/img`。

## 3. 4090 主线（Round 1 + Round 2），代码目录 `exp/v25/`

### 3.1 Round 1：无训练推理消融（修正 C1：τ 截断）

**事实核对**：`exp/v22/core.py:353` `tau_v = (tau_min + (tau_hi − tau_min)·g_tau).clamp(0.0, 0.78)`。`tau_hi > 0.78` 只会扩大触顶样本比例，**不产生更高终点**；积分自 `tau_v` 至 1.0，故**起点 `tau_min` 才是强度主控**。
**实测核验（Round 1 后补）**：`tau_records` 显示当前所有配置 `frac_at_cap = 0.00`（τ_max ≤ 0.775）——0.78 clamp 在本文配置下**不 binding**，曲线 B 三臂（τ_min 0.50/0.58/0.66 → 实测 mean 0.695/0.720/0.744）是三个真正不同的强度档。完整读数与判定见 [`04_round1_results.md`](04_round1_results.md)。

**P0-1 单模型 vs Hybrid（`exp/v25/r1_inference_ablation.py`）**：
- **曲线 A（PRISM-1，仅调 λ）**：`A_p1_l075`（复用 `0.7715/0.4371/0.4848`）、`A_p1_l100`（复用 `0.7440/0.4430/0.5057`）、实跑 `λ=1.30/1.60`（`α=min(0.45, 0.20λ)`）。
- **曲线 B（V20 D6，起点上移扫描；`tau_hi` 固定 0.78 防截断歧义）**：`tau_min ∈ {0.40(复用 M0_v20_d6 等效), 0.50, 0.58, 0.66}`。**每臂记录实际 τ 分布**（`tau_mean`/`tau_max`/触顶比例，来自 `integrate_flow` 返回的 diagnostics），并在结果 JSON 中存 `tau_min/tau_hi` 配置与实测分布两行。
- **曲线 C（Hybrid 线性混合，仅调 γ）**：复用 `g25/g40/g55`，实跑 `g70/g85`。
- 决策：在 `DINO-C ≈ 0.70/0.73/0.75` 插值比较三曲线的 `(DINO-S, LPIPS, Latency)`；B 追平 C（非劣性通过）→ 淘汰 Hybrid；C 在风格指标 CI 下界超容忍带 → Hybrid 保留为参考上限 + 机制 B 蒸馏目标。

**P0-2 CFG 分支（同脚本）**：固定 `slf_core_v2.pt`（唯一完整训过全分支的 V20 原主权重）+ T4 锚点 + original Gate，两档内容水位（`D5_SOLVER` `τ∈[0.28,0.66]` / D6-high `τ∈[0.48,0.78]`）× `{sc_only, cs_j, full}`。`full` 仅在此权重上评估（C4 规则）。`cs_j` 附 `ws_fig=5.5` 档防误杀。

**P1-3 锚点（同脚本）**：Round 1 胜出主权重 + CFG 模式下，等 `DINO-C≈0.70` 对比 `Anc_T4 / Anc_PRISM1 / Anc_Source`（各自预配 τ 对齐水位），并实测端到端 `ms/img` 与峰值显存。

**P1-4 Gate（同脚本）**：最终主权重 + 锚点上，`DINO-C≈0.65 / 0.70` 两档 × `original vs fixed_g=0.5`。

### 3.2 Round 2：最小受控训练矩阵（修正 C4：配方↔分支对应）

从 `slf_final.pt` 同起点、`1500 updates`、`batch=120`、`AdamW 1.2e-4→1.5e-5`、`EMA=0.996`、`layout=1.0`、同 `seed=20263500`（共享数据顺序与噪声）：

| 训练臂 | `drop_c` | Anchor-Bridge | 有效分支覆盖 | 允许的评测 CFG |
|---|---:|---:|---|---|
| `Tr_T0` | 0% | 0% | 仅内容条件双分支 | **仅 `cs_j`** |
| `Tr_T1` | 30% | 0% | 双分支 + 无内容分支 | `cs_j` + `full` |
| `Tr_T2` | 0% | 25%（真 `z0_tr`） | 仅内容条件双分支 | **仅 `cs_j`** |
| `Tr_T3` | 30% | 25% | 双分支 + 无内容分支 | `cs_j` + `full` |
| `Tr_T4_swd`（条件追加） | 取 T0–T3 最优 | 同上 | 随胜者 | 随胜者 |

- **评测规则**：配方对比（T0–T3）一律先在共同有效的 **`cs_j` 双分支**两档水位（`DINO-C≈0.65/0.70`）进行；`full` 四分支仅对 `drop_c>0` 的 T1/T3 附加评测，用于回答"配方 × CFG 是否交互"，**不得**用 full 读数直接比较 T0/T2。
- Anchor-Bridge 修复不变：先以 `B4 + canonicalizer` 预计算 4400 张训练图真 `z0_tr`（~1 min，与机制 A/B 共享缓存 `runs/v25/cache/`）。

## 4. 3060 One-Shot（Round 3，与 4090 并行）

### 4.1 修正 C3：`O3` vs `O3_label_free` 归因后置

V24 现有对比（`O3` t45 `0.5845/0.5245` vs `O3_label_free` t45 `0.6769/0.4666`）**混杂了三个变量**：配对来源（类别标签配对 vs 无标签教师对）、教师编辑幅度（标注管线教师 vs `mid_cos=0.965` 保守教师）、参考图分布；且 O 系列训练本就 `onehot=0`，标签只影响配对构造，不存在"网络输入含类别"变量。**结论：收敛阶段不新跑 O3 变体训练**；归因实验移入机制 C1——用校准到相同 `mid_cos` 带（0.80–0.86）的无标签教师对与 O3 的标签对做**幅度匹配对比**，剩余差距才可归因于"标签配对"本身。

### 4.2 Round 3 实际执行项（`exp/v25/r3_oneshot_minimal.py`）

1. **盲测补全（复用已有权重，零训练）**：`O3_label_free` 补 `swap_dist` 参考更换敏感度 + SG-25 OOD 125-pair，与 `O3`（`0.5867 / ΔS +0.157 / swap 0.442`）同表对比。
2. **跨风格监督必要性（单变量，保留）**：`O3_self_only`——同 `O3_rgb_cnx_gated` 结构、同 1200 updates、同初始化，仅把 50% 跨风格对替换为 100% 自配对重建；`t45/t55` + `swap_dist`。
3. **单一 strength 标定（零训练）**：`O3_rgb_cnx_gated` 同权重补 `τ ∈ {0.40, 0.60}`（与已测 0.45/0.55 成曲线）+ 端到端延迟。

## 5. ETA（校准值同初版）

| 阶段 | 内容 | ETA |
|---|---|---:|
| 4090 Round 1 | P0-1（8 实跑）+ P0-2（6）+ P1-3/P1-4（8）+ bootstrap + τ 分布记录 | ~25 min |
| 4090 Round 2 | `z0_tr` 缓存 + T0–T3（4 训 + 8 评 cs_j + 4 评 full 附加）+ 条件 T4_swd | ~45 min（不含 SWD）/ ~75 min（含） |
| 3060 Round 3 | 盲测补全 + `O3_self_only`（1 训 3 评）+ τ 补扫 | ~36 min |
| 收尾 | 汇总 + 接口冻结 + 文档/图表/pytest/提交 | ~15 min |

里程碑 1（Round 1 + Round 3 完）≈ **35–45 min**；里程碑 2（含 Round 2 + 冻结）≈ **1.5–2.1 h**。收敛期产出冻结为 `PRISMCore.transfer(z_src, style_proto, strength)` 与 `PRISMRef.transfer(content_rgb, reference_rgb, strength)` 双接口（`src/model/prism_v25.py`）。
