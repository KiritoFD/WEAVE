# A01 Round 1 配对 Bootstrap 非劣性 CI 报告（2026-11-20）

任务来源：[agent2.md §三 A01](../../19_pull_together/agent2.md)。数据：`runs/v25/measured/<arm>/eval.json`（750 行逐任务记录，读数 cross600）。

**统计基础唯一来源**：[src/eval/metrics.py](../../../src/eval/metrics.py)（`bootstrap_source_ci` 源聚类重抽样 / `paired_cluster_ci` 聚类逻辑）。工具 `tools/v25_cpu/a01_noninferiority.py` 只做扩展包装（严格配对校验 + 跨指标共享重采样索引 + 预注册容忍带三态判定 + eval.json 加载），**未重新发明任何公式**。单测 `tests/v25/test_stats.py` 验证包装层与 `bootstrap_source_ci`/`paired_cluster_ci` 逐位一致。

## 一、协议

- **配对**：两臂 cross 任务按 `task_id` 严格配对，600/600 完全一致（缺失、重复、NaN、集合不一致立即 `ValueError`；本轮 7 组比较全部通过校验）。
- **聚类**：150 个 source（`src_stem`），每源恰 4 个 cross 任务（聚类组完整性受校验）；cluster bootstrap 有放回重抽源、保留重复。
- **联合分布**：`n_boot=1500`、`seed=2026`，一次生成 `1500×150` 重采样索引矩阵，**全部指标共用同一矩阵**（与 `bootstrap_source_ci` 内部 RNG 构造逐位一致，单测断言相等），保持逐 replicate 的指标间联合分布。
- **预注册容忍带**（[01_convergence_plan.md §2](../../19_pull_together/01_convergence_plan.md)，simple − complex 方向）：`Δdino_c ≥ −0.010`，`Δdino_s ≥ −0.005`，`Δlpips ≤ +0.010`（**LPIPS 越低越好**，单测覆盖方向性）。
- **三态判定（非劣性 ≠ 显著性）**：`PASS`=三指标 CI 整体带内（可删）；`FAIL`=复杂版在某指标 CI 越出带远端（显著更优，保留）；`INCONCLUSIVE`=CI 与带边重叠（默认保留，标注未决）。**CI 是否跨零不作为任何判定输入**。
- **删除附加守卫**（预注册）：五画派最差类 Top-1 单类跌幅 ≤ 3pp；same150 盲测 Top-1 不退化（本轮实现为 ≤ 0.5pp = 1 任务）。PASS 但守卫被违 → 降级 INCONCLUSIVE。

## 二、结果总表（7 组）

方向均为 simple − complex；final 为预注册规则（CI 三态 × 守卫）的最终判定。

| # | 比较（simple vs complex） | verdict | final | 关键读数 |
|---|---|---|---|---|
| 1 | `P02_d6h_csj` vs `P02_d6h_full`（删 full CFG） | PASS | **INCONCLUSIVE** | CI 全带内，但 Minimalism Top-1 −7.50pp > 3pp 守卫 |
| 2 | `P14_d6h_fixed_g50` vs `P02_d6h_csj`（删密度门控） | **FAIL** | **FAIL** | `Δdino_c CI = [−0.0187, −0.0113]` 整体低于 −0.010 带下界 |
| 3 | `P02_d5_csj_ws42` vs `P02_d5_full`（删 full @ D5） | PASS | **INCONCLUSIVE** | CI 全带内，但 Minimalism Top-1 −21.67pp |
| 4 | `P13_ancp1_t040` vs `P02_d6h_csj`（PRISM-1 锚替代 T4） | INCONCLUSIVE | INCONCLUSIVE | 三指标全部骑在带边 + τ 未对齐混杂 |
| 5 | `V25_Tr_T1` vs `V25_Tr_T0` @ D5 cs_j | INCONCLUSIVE | INCONCLUSIVE | `Δdino_s lo = −0.0051` 逊带边 −0.005 仅 0.0001 |
| 6 | `V25_Tr_T1` vs `V25_Tr_T0` @ D6h cs_j | **PASS** | **PASS** | 三指标带内 + 守卫全过；Top-1 +1.00pp CI 不含零 |
| 7 | `B_d6_t066` vs `C_hyb_g70`（敏感性参考） | INCONCLUSIVE | INCONCLUSIVE | 水位不对齐（0.7007 vs 0.7129），**非判定输入** |

same150 盲测 Top-1 全部 96.67%，所有比较 same-blind 跌幅 +0.00pp，守卫无违。

## 三、逐组 CI 详表

### 1. `P02_d6h_csj` vs `P02_d6h_full` —— 删 full CFG 分支（40→20 NFE）

| metric | mean | CI95 | 带内 |
|---|---:|---|---|
| `dino_c` | −0.0039 | [−0.0058, −0.0019] | yes |
| `dino_s` | +0.0048 | [+0.0036, +0.0059] | yes |
| `lpips` | +0.0035 | [+0.0028, +0.0042] | yes |
| `target_top1`（辅助） | −0.0017 | [−0.0167, +0.0133] | − |

verdict PASS（CI 三指标全带内，与 04 文档 §三 P0-2 点估计一致），**但** worst-class 守卫违例：Minimalism Top-1 `0.800 → 0.875`（full 更优，−7.50pp > 3pp），Rococo 反向 +4.2pp。预注册删除条件要求守卫同时满足 → **final: INCONCLUSIVE（默认保留 full，标注未决）**。逐类 Top-1：csj `[ER .225, Imp .892, Min .800, Roc .533, Uki .300]` vs full `[.225, .883, .875, .492, .283]`。

### 2. `P14_d6h_fixed_g50` vs `P02_d6h_csj` —— 删 sigmoid 密度门控（fixed g=0.5）

| metric | mean | CI95 | 带内 |
|---|---:|---|---|
| `dino_c` | −0.0149 | [−0.0187, −0.0113] | **no（CI 整体越下界）** |
| `dino_s` | +0.0081 | [+0.0053, +0.0111] | yes |
| `lpips` | +0.0037 | [+0.0019, +0.0054] | yes |
| `target_top1`（辅助） | −0.0267 | [−0.0475, −0.0067] | − |

**verdict: FAIL（保留密度门控）**。fixed g=0.5 的内容损失 CI 整体低于带下界（复杂版显著更优），且 Ukiyo_e Top-1 −10.00pp、总体 Top-1 −2.67pp（CI 不含零）。**修正 04 文档 §三 P1-4 的初步读法**（该读法只在插值对齐后比较了风格指标，未配对检验内容指标）。注意混杂：fixed 臂实测 τ 分布整体下移（mean 0.630 恒定 vs original 0.689），删除门控的操作语义本身就包含 τ 重分布；若把 fixed 常数重新标定到同内容水位属于**新臂**（未实测），需 GPU 补测后才能重判。

### 3. `P02_d5_csj_ws42` vs `P02_d5_full` —— 删 full @ D5 水位

| metric | mean | CI95 | 带内 |
|---|---:|---|---|
| `dino_c` | +0.0121 | [+0.0075, +0.0170] | yes |
| `dino_s` | +0.0049 | [+0.0029, +0.0069] | yes |
| `lpips` | +0.0006 | [−0.0007, +0.0019] | yes |
| `target_top1`（辅助） | −0.0467 | [−0.0667, −0.0283] | − |

verdict PASS，但 worst-class 守卫大幅违例：Minimalism Top-1 `0.750 vs 0.967`（**−21.67pp**），总体 Top-1 −4.67pp CI 不含零 → **final: INCONCLUSIVE**。D5 档的 full 分支对 Minimalism 判别有实质贡献，不能按三指标带内直接删除。

### 4. `P13_ancp1_t040` vs `P02_d6h_csj` —— PRISM-1 锚替代 T4 锚

| metric | mean | CI95 | 带内 |
|---|---:|---|---|
| `dino_c` | −0.0091 | [−0.0131, −0.0052] | no（lo 越界） |
| `dino_s` | −0.0058 | [−0.0084, −0.0034] | no（lo 越界） |
| `lpips` | +0.0115 | [+0.0091, +0.0139] | no（hi 越界） |
| `target_top1`（辅助） | −0.0150 | [−0.0417, +0.0117] | − |

**verdict: INCONCLUSIVE**（三指标全部骑在带边，既不满足删除也不满足保留）。**混杂未消**：τ_min 0.40 vs 0.48 未对齐（内容水位 0.6691 vs 0.6782），04 文档已注明"τ 对齐后重判"；且 worst-class Ukiyo_e −18.33pp。**本轮不能判定 PRISM-1 锚可替 T4**，维持 T4 为默认锚。

### 5/6. `V25_Tr_T1`（drop_c 30%）vs `V25_Tr_T0` —— Round 2 配方对比（共同有效分支 cs_j）

@ D5 cs_j：

| metric | mean | CI95 | 带内 |
|---|---:|---|---|
| `dino_c` | −0.0036 | [−0.0060, −0.0012] | yes |
| `dino_s` | −0.0032 | [−0.0051, −0.0015] | **no（lo −0.0051 vs −0.0050，逊边 0.0001）** |
| `lpips` | +0.0035 | [+0.0028, +0.0043] | yes |
| `target_top1`（辅助） | +0.0317 | [+0.0158, +0.0483] | − |

@ D6h cs_j：

| metric | mean | CI95 | 带内 |
|---|---:|---|---|
| `dino_c` | −0.0021 | [−0.0034, −0.0007] | yes |
| `dino_s` | −0.0014 | [−0.0023, −0.0004] | yes |
| `lpips` | +0.0017 | [+0.0013, +0.0022] | yes |
| `target_top1`（辅助） | +0.0100 | [+0.0017, +0.0200] | − |

**@ D6h verdict: PASS（守卫全过）** —— T1 配方在主配置水位上对 T0 非劣，且 Top-1 +1.00pp（CI 不含零）；**@ D5 verdict: INCONCLUSIVE**（dino_s 下界恰好压线，差 0.0001；但 Top-1 +3.17pp 显著更高）。综合：T1 未被否定、总体 Top-1 双水位均显著占优，且 drop_c>0 是评测 `full` 分支的前提（C4 规则）——**支持 T1 作为 Round 2 胜出配方的证据成立一半（D6h 确认 / D5 压线未决）**。

### 7. `B_d6_t066` vs `C_hyb_g70` —— 敏感性参考（非判定输入）

| metric | mean | CI95 | 带内 |
|---|---:|---|---|
| `dino_c` | −0.0121 | [−0.0168, −0.0078] | no |
| `dino_s` | +0.0146 | [+0.0119, +0.0174] | yes |
| `lpips` | +0.0087 | [+0.0054, +0.0120] | no |
| `target_top1`（辅助） | +0.0083 | [−0.0150, +0.0333] | − |

水位不对齐（0.7007 vs 0.7129），仅作趋势参考：单模型 D6 在最近实测点风格占优（+0.0146 CI 不含零）、内容略亏，与 04 文档曲线插值读数方向一致。

## 四、未决项（不计算、不外推、不伪造配对）

1. **`D6 vs Hybrid @ dino_c≈0.73`：UNDECIDED**。实测臂内容水位覆盖到 B 曲线 0.7007（`B_d6_t066`）与 Hybrid 0.7129（`C_hyb_g70`）；0.73 处**至少一臂没有任何逐任务实测**，按 A01 验收第 5 条（禁止均值外推构造假配对样本）不得插值配对。04 文档清单第 1 项的"插值臂逐对配对"在本轮**不执行**。解决路径：4090 补一条 `tau_min≈0.72` 的 D6 臂直接实测 0.73 水位后，用同一工具配对判定。
2. **`V25_Tr_T3 vs V25_Tr_T0`：未计算**。`runs/v25/measured/` 尚无 `V25_Tr_T3_*` 评测产物（Round 2 训练矩阵未全部完成）；GPU 产出后用 `compare_arms` 直接补算，工具与协议已就绪。
3. `V25_Tr_T2 vs V25_Tr_T0` 不在本任务清单内；T2 双水位臂已存在，同工具可随时补算。

## 五、对 04 文档初步读数的修正汇总

| 04 文档初步读法（点估计/插值） | 本轮配对 CI + 守卫结论 |
|---|---|
| P0-2：D6h 档 Δ 全带内 → 删 full | **INCONCLUSIVE**：CI 确认带内，但 Minimalism −7.50pp 触最差类守卫，暂不删 |
| P0-2：D5 档 cs_j 两项优于 full | **INCONCLUSIVE**：Minimalism −21.67pp，full 在该画派有实质贡献 |
| P1-4：fixed_g=0.5 带内 → 删密度门控 | **FAIL**：内容损失 CI 整体越带下界，密度门控保留（fixed 常数重标定属新臂，未实测） |
| P1-3：PRISM-1 锚差在带缘，待对齐重判 | **INCONCLUSIVE**（维持）：三指标骑边 + τ 未对齐 + Ukiyo_e −18.33pp |
| Round 2 配方 T1 vs T0 | **D6h PASS / D5 压线 INCONCLUSIVE**；T1 Top-1 双水位显著占优 |

## 六、复算命令

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  python -m tools.v25_cpu.a01_noninferiority          # 本报告全部数字

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  python -m pytest tests/v25/test_stats.py -q         # 16 passed
```

CPU-only（numpy + stdlib）；无 CUDA / torch API 调用；线程数限制 2。
