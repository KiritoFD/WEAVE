# A02：T4 / PRISM-1 / D6 教师与锚点一致性诊断（血缘审计）

> **TASK**: A02（Slot A，P0 批）｜**日期**: 2026-11-20｜**状态**: DONE（本地部分）；B4↔T4 张量校验 **BLOCKED**（缓存仅在 4090，命令已登记为 G 队列待办）
> 工具：[`tools/v25_cpu/a02_teacher_lineage.py`](../../../tools/v25_cpu/a02_teacher_lineage.py)｜测试：[`tests/v25/test_a02_lineage.py`](../../../tests/v25/test_a02_lineage.py)（31 passed，CPU）
> 验收依据：[`docs/19_pull_together/agent2.md`](../../19_pull_together/agent2.md) §三 A02；Round 1 语境：[`docs/19_pull_together/04_round1_results.md`](../../19_pull_together/04_round1_results.md)（顶部 CI 修正块）

## 一、血缘链定义与登记方法

每条"教师目标生成路径"或"评测路径"登记为五元组（在四元组基础上补 branches/gate 拆分）：

```
model_sha → anchor_sha → solver_hash → output_hash
```

- **model_sha**：checkpoint 文件 sha256（仅本地存在的文件可算；4090 独占文件记 `unknown`，由 4090 manifest 回填）。
- **anchor_sha**：锚点版本键。T4 用 `T4_HASH = config_hash(T4_CONFIG)`（本地可复算 = `04007b4f0355`，与 [exp/v22/t4.py](../../../exp/v22/t4.py) 内值一致，漂移测试守卫）；B4 锚用其 ckpt sha（本地实测 `2716436fe0ee…`）；张量级 sha 由 4090 校验命令回填。
- **solver_hash**：solver dict 排序 JSON sha256（与 `exp.v22.config_hash` 同一序列化约定，键序无关；已测试与 live 模块逐位一致）。
- **output_hash**：评测臂 = `runs/<v>/measured/<arm>/eval.json` 的文件 sha（本地可算）；训练侧 = 缓存 `.pt` 文件 sha（本地缺文件 → `unknown`）。

关键 solver hash（12 hex）：D5 `07904c2bb16b`｜D6 `e14eb16ca1e9`｜**D6H `f52948ae3451`**（= D6 + tau_min 0.48 / tau_hi 0.78，[exp/v25/r1_inference_ablation.py:31](../../../exp/v25/r1_inference_ablation.py)）｜DualSWD 臂 `0d88c4ae06b5`｜曲线B t066 `fa46f171706e`｜P13 锚臂 `9885c0e7d0f7`。

全量登记（17 条 record，含 file:line 出处）由 `python -m tools.v25_cpu.a02_teacher_lineage` 输出；此处列主干：

| record | side | model | anchor | solver8 | branches/gate | seed | 出处 |
|---|---|---|---|---|---|---|---|
| `ANCHOR_T4_TEST750` | anchor | 无 ckpt（AnisotropicS10Teacher 2 步优化） | t4_opt / T4_HASH `04007b4f` | - | cache | 20261185 | [t4.py:45-83](../../../exp/v22/t4.py) |
| `ANCHOR_B4_TRAIN4400` | anchor | B4_distill50 (`2716436f`) | b4_canon | - | cache | 前馈确定 | [precompute_train_latents.py:75-84](../../../exp/v25/precompute_train_latents.py) |
| `EVAL_V20_D6_ORIG` | eval | slf_core_v2 | t4_opt | D6 `e14eb16c` | full/original | 20261185 | [run_v20.py:389](../../../exp/v20/run_v20.py) |
| `EVAL_V24_M0_V20_D6` | eval | slf_core_v2 | t4_opt | D6 `e14eb16c` | full/original | 20261185 | [p0_m0_m2.py:67-85](../../../exp/v24/p0_m0_m2.py) |
| `EVAL_V25_P02_D6H_CSJ` | eval | slf_core_v2 | t4_opt | D6H `f52948ae` | cs_j/original | 20261185 | [r1_inference_ablation.py:165-172](../../../exp/v25/r1_inference_ablation.py) |
| `EVAL_V25_R2_TRAINED` | eval | slf_Tr_T{0..3} | t4_opt | D6H `f52948ae` | cs_j/original | **20261119** | [r2_train_ablation.py:186-195](../../../exp/v25/r2_train_ablation.py) |
| `TRAIN_V25_TEACHER_CACHE` | train | slf_core_v2 | **b4_canon** | D6H `f52948ae` | cs_j/original | 20263502 | [precompute_train_latents.py:35-43,112-122](../../../exp/v25/precompute_train_latents.py) |
| `TRAIN_V25_R2_SELFPAIRED` | train | slf_final (base) | **z_src_clean** | 无（自配对 FM 目标） | - | 20263500 | [r2_train_ablation.py:54-165](../../../exp/v25/r2_train_ablation.py) |
| `EVAL_PRISM1_L100` / `TRAIN_V25_PRISM1_TARGETS` | eval/train | s2_prism1_v2 | z0_chroma α=0.20（λ=1.0） | 单步前馈 | - | 确定 | [p0_m0_m2.py:31-63](../../../exp/v24/p0_m0_m2.py) / precompute:86-110 |
| `LEGACY_V24_M3_M4_POOL` / `LEGACY_V24_M6_ZP1_ANCHOR` | legacy | - | **avg_pool 伪锚** / **z_p1 充当 D6 锚** | - | - | - | [p1_m3_m4_residual.py:179,213](../../../exp/v24/p1_m3_m4_residual.py) / [p2_m6_flow_bridge.py:82-89](../../../exp/v24/p2_m6_flow_bridge.py) |

## 二、三张差异表（核心结论）

### 表 (a) 完全一致对（可互相归因）

| a | b | verdict | 说明 |
|---|---|---|---|
| `EVAL_V20_D6_ORIG` | `EVAL_V24_M0_V20_D6` | **IDENTICAL** | V20 原版 D6 求解器 dict 与 V22/V24 `D6_SOLVER` 逐键相同（hash 同），同 T4 锚（T4_HASH 绑定）、同 full/original、同 EVAL_SEED=20261185 噪声 → V24 M0_v20_d6 作为"复用基线"在配置级成立；model_sha 待 4090 manifest 回填（同路径） |
| `TRAIN_V25_PRISM1_TARGETS` | `EVAL_PRISM1_L100` | **IDENTICAL** | 训练侧 z_p1_tr 与评测侧 z_p1_100 同 λ=1.0 配方（α=0.20），同一 ckpt 路径；仅域不同（train4400 vs test750），归因限于"配方级" |

### 表 (b) 可等价对（需 4090 校验确认后才可归因）

| a | b | verdict | 缺什么证据 |
|---|---|---|---|
| `ANCHOR_T4_TEST750` | `ANCHOR_B4_TRAIN4400` | **EQUIVALENT_PENDING_4090** | [precompute_train_latents.py:5](../../../exp/v25/precompute_train_latents.py) 声称 B4+canonicalizer 锚 = "T4 锚的训练侧等价物"，但两者生成路线完全不同（T4 = 2 步各向异性 S10-VGG 优化输出；B4 = 218K C3UNet 前馈蒸馏，V6/P1 时代产物）。注意：该配方的原始出处 [routed_student.py:407-408](../../../exp/v19/routed_student.py) 是**错标签反事实臂**的锚，不是 T4 正臂——"T4 等价"是书面断言而非代码事实。P1-3 已证锚语义可移动 Top-1 ~48pp，故必须先过张量校验 |
| `TRAIN_V25_TEACHER_CACHE` | `EVAL_V25_P02_D6H_CSJ` | **EQUIVALENT_PENDING_4090** | 教师三元组除锚外全部对齐（同 slf_core_v2、同 D6H hash `f52948ae`、同 cs_j、同 original gate）；差的正是 b4_canon vs t4_opt 锚（表上行）+ 4090 侧 model_sha 回填。**这是机制 A/B 教师读数与评测读数之间的主闸门** |

### 表 (c) 确定不一致对（立即报告，禁止互相归因）

| a | b | verdict | 不一致点与评级 |
|---|---|---|---|
| `TRAIN_V25_R2_SELFPAIRED` | `EVAL_V25_R2_TRAINED` | **INCONSISTENT** | **Round 2 AB 锚 = z_src 干净 latent**（`za = z1`，[r2_train_ablation.py:123-137](../../../exp/v25/r2_train_ablation.py)）vs 评测 T4 锚：训练-推理分布差，**设计内已知**（AB 桥 tau0~U(0.40,0.78) 部分缓解，但 P1-3 证明 z_src 锚在推理端风格坍缩 Top-1 7.17%）。评级：已知有意为之、缓解不消除——Tr_T2/T3 的增益**不得**归因为"教师锚一致" |
| `EVAL_V25_P13_ANCSRC` | `EVAL_V25_P02_D6H_CSJ` | **INCONSISTENT** | z_src 锚判负臂（Top-1 7.17%）与 T4 参照：锚语义相反，仅作反例档案保留 |
| `LEGACY_V24_M3_M4_POOL` | 任何 V25 教师/锚 | **INCONSISTENT** | avg_pool(3×3 replicate) 伪 z0 锚（M3/M4/M5 失败臂）；precompute 已显式用真 z0_tr 取代；其衍生缓存（含 `ref_teacher_pairs_2000.pt` 的 M5/M6 教师目标）禁止再作教师 |
| `LEGACY_V24_M6_ZP1_ANCHOR` | `EVAL_V20_D6_ORIG` | **INCONSISTENT** | avg_pool→PRISM-1→其输出再当 D6 锚的双重伪造链（M6 失败臂） |

附加发现（output 级血缘差，非配置不一致但影响配对 CI）：**评测噪声种子分裂**——V20/V24/R1 全部用 `EVAL_SEED=20261185`，而 Round 2 评测用 `20261119`（[r2_train_ablation.py:181](../../../exp/v25/r2_train_ablation.py)）。跨 Round 的配对 Bootstrap（如 CI 清单 #5 "V25_Tr_T* vs 最优基线"）混合了两次独立噪声抽样，需补同种子重跑一侧或显式标注 seed-unmatched。

### (d) 参考信息：同引擎不同强度档（DISTINCT_LINEAGE，仅限 A01 同水位插值比较）

`M0_v20_d6(full/D6)`、`B_d6_t066(full/τ-shift)`、`P02_d6h_csj(cs_j/D6H)`、`B_dswd_t058(DualSWD 权重+ws5.2)`、`P13_ancp1(τ 未对齐+p1 锚)` 互相之间均为 DISTINCT——它们是合法的不同臂，但**不是**同一教师配置，任何"删组件"式归因必须走 A01 配对 CI，不能当同配置读数直减。

## 三、B4 锚 vs T4 锚一致性检验（本地 BLOCKED → 4090 待办）

本地 `runs/v22/cache/`、`runs/v25/cache/`、`runs/v24/cache/` 与 slf_core_v2/s2_prism1_v2/slf_final 均不存在（4090 独占），仅 `runs/v6/P1_B4_distill50/P1_B4_distill50.pt` 在本地（sha `2716436fe0ee…`）。**未执行任何 ssh**。G 队列待办一条命令：

```bash
ssh 4090 "cd /root/Workspace/WEAVE && OMP_NUM_THREADS=2 python -m tools.v25_cpu.a02_teacher_lineage --check-b4-t4 --device cuda > runs/v25/cache/a02_b4_vs_t4.json"
```

该命令在 4090 上：加载 T4 缓存（T4_HASH 版本校验）→ 用同一 `b4(canon(z0,c1_gentle),0,canon,tid)` 配方对 test750 的 750 个 z0 重建 B4 锚 → 输出逐任务 L2（mean/p50/p95/max）+ 逐通道分位数差 + 以 `L2(T4, z0)` 为尺度的 divergence ratio + 双侧张量 sha256。

判读规则（登记于 `b4_t4_stats`）：`ratio < 0.15` → equivalent-scale（可开闸表 (b) 归因）；`ratio > 0.60` → distinct-scale（锚视为不同 lineage，机制 B 教师需换 T4 型锚重造或在评测侧补 B4 锚臂）；中间带 → inconclusive，交由 τ 对齐后的实测评测臂裁决。注意判读是**归因闸门**而非替换许可：即使 ratio 小，B4 与 T4 仍是不同对象，机制 B 的"教师=评测"声明必须引用该 JSON 的双侧 sha。

## 四、机制 A/B/C 使用约束清单（给 G 队列与 B 槽的硬规则）

1. **教师身份铁律**：机制 A/B 读数只能归因到身份键 `(model_sha, anchor_sha, solver_hash, branches, gate)` 完全匹配的教师；当前唯一登记的 D6 教师目标是 `TRAIN_V25_TEACHER_CACHE`（`z_d6_tr`，D6H `f52948ae` / cs_j / original）。
2. **B4↔T4 闸门**：训练侧（B4 锚）教师读数与评测侧（T4 锚）读数之间的任何比较，在上节命令出 JSON 且 ratio 达标前 **BLOCKED**；不得以"teacher_spec 写了 T4-equivalent"为由跳过。
3. **Round-2 AB 例外条款**：Tr_T2/T3 允许在 T4 锚上评测（r2 实际做法），但禁止描述为"教师锚一致训练"；Tr_T2/T3 vs `P02_*`/`M0_*` 的直接归因禁止。
4. **种子条款**：跨 Round（20261185 vs 20261119）配对 CI 视为 seed-unmatched；决策级比较必须同种子重跑一侧。
5. **历史禁用**：V24 M3/M4/M5（avg_pool 伪锚）与 M6（z_p1 当锚）的教师目标/缓存禁止进入任何 V25+ 教师或锚缓存。
6. **DualSWD 隔离**：`B_dswd_t058` 权重与 ws_fig 均不同，独立 lineage，只允许 A01 同水位插值比较。
7. **口径**：一切质量决策读 full test750 的 cross600（2026-11-14 用户决定）；dev 预览不入血缘表。
8. **缓存复用**：今后所有锚/教师缓存必须随张量存 `LineageRecord.to_dict()`（身份键+sha），加载时比对不匹配即重生成（沿用 [exp/v22/t4.py](../../../exp/v22/t4.py) 的四情形校验模式；其 raw meta 无 `t4_hash` 时按 legacy 接受为 `None` → 版本 unknown，属"不报错但标 unknown"的既有契约）。

## 五、结论与风险

**SCIENTIFIC_CONCLUSION**
- confirmed（配置级证实）：① V20→V22/V24/V25 的 D6/D5 求解器 dict 逐位一致（hash 相同）；② V25 教师缓存与评测主配置 `P02_d6h_csj` 的 (ckpt, solver, branches, gate) 三元组对齐，唯一缺口是锚；③ Round 2 AB 训练锚（z_src 干净 latent）与评测 T4 锚确定不一致（训练-推理分布差，设计内已知）；④ 评测噪声种子在 R1/R2 间分裂。
- rejected：① "B4+canonicalizer 锚 = T4 等价物"目前**无证据支撑**（书面断言；配方原始出处是错标签反事实臂）；② V24 M3/M5/M6 的伪锚教师链（已被 precompute 明确废弃）。
- inconclusive：B4↔T4 张量级等价性（BLOCKED，4090 命令已登记）；PRISM-1 锚替换 T4 锚（等 A01 CI #3 的 τ 对齐配对）。

**RISKS**
- 若 4090 校验判 distinct-scale，机制 A/B 已训练/将训练的教师目标（z_d6_tr 基于 B4 锚）与评测锚失配，需在评测侧补 B4 锚臂或重造教师——建议 G4（机制 B0）启动前先跑该命令。
- 跨 Round seed 不匹配若被忽视，CI 清单 #5 可能产生假阳性/假阴性配对结论。
- `runs/v19/raw/V19_T4_sweet_vgg_nosid` 的 meta 无 `t4_hash`（legacy None 接受路径），版本追溯只能靠 T4_CONFIG 漂移测试与张量 sha（4090 JSON 已含 `z_t4_sha256`，建议随后固化进缓存元数据，对应约束 8）。

**NEXT_TASK_SUGGESTION**：G 队列把 `--check-b4-t4` 命令排在 G2/G4 之前；JSON 回来后用 `b4_t4_stats` 判读并回填本页表 (b)；将 `a02_b4_vs_t4.json` 的双侧张量 sha 写入 `train_latents_4400.pt` 的 `teacher_spec`（约束 8 的首个落地实例，可并入 B03 血缘检查器）。
