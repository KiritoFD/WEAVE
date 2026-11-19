# V25 GPU-4090 队列结果（总控代 G1 归档）

> 队列管理：tmux（`v25r1` → `v25r2q` → `v25pre`）。所有读数 cross600 × S10 × test750 四裁判。
> 数据源：`runs/v25/r1_inference_ablation.json`、`runs/v25/r2_train_ablation.json`（本地镜像 `runs/v25/`）；台账 `docs/performance/metrics.csv`（480 行）。

## Round 1（完成，21 新臂 + 6 复用）

详细表格与四判定点分析见 [`docs/19_pull_together/04_round1_results.md`](../19_pull_together/04_round1_results.md)。摘要：

- **曲线 B**（单模型 D6，τ_min 起点上移 0.50/0.58/0.66 → `0.6842/0.6926/0.7007` D-C，D-S 稳定 0.503–0.507）在同内容水位风格反超 Hybrid（点估计 +0.008~+0.013）；**D-C=0.73 处为外推，判定未决**（agent2.md 铁律），待真实工作点。
- **P0-2**：`cs_j` ≈ `full`（CI 由 A01 出）；`sc_only` 损 Top-1 −3pp。
- **P1-3**：`Anc_PRISM1` ≈ `Anc_T4`（τ 未对齐 confound 待 CI）；`Anc_Source` 判负（风格坍缩）。
- **P1-4**：`fixed_g=0.5` ≈ `original`（可删密度门控，待 CI）。
- τ 分布实测：`frac_at_cap=0.00`，0.78 clamp 在当前配置不 binding（17 条记录在 r1 JSON）。

## Round 2（完成，4 训练臂 × cs_j 两档 = 8 臂）

| 臂 | drop_c | AB | d5 档 D-C / D-S / LPIPS / Top-1 | d6h 档 D-C / D-S / LPIPS / Top-1 |
|---|---|---|---|---|
| `Tr_T0` | 0% | 0% | 0.6122 / 0.5403 / 0.5632 / 54.00% | 0.6634 / 0.5175 / 0.5393 / 55.67% |
| `Tr_T1` | 30% | 0% | 0.6086 / 0.5371 / 0.5668 / **57.17%** | 0.6613 / 0.5162 / 0.5411 / 56.67% |
| `Tr_T2` | 0% | 25% | 0.6146 / 0.5378 / 0.5639 / 54.33% | 0.6659 / 0.5160 / 0.5390 / 55.17% |
| `Tr_T3` | 30% | 25% | 0.6108 / 0.5360 / 0.5674 / 56.17% | 0.6632 / 0.5151 / 0.5411 / **56.83%** |

- checkpoint sha256（前 16 位）：`e348be4e014ec6d6`(T0) / `da883c8a6060c162`(T1) / `936c71e3d63761b5`(T2) / `a9abd0de3ed8f7a1`(T3)，路径 `runs/v25/r2_tmatrix/`。
- 初步读法（点估计）：drop_c 与 AB 对内容/风格几乎零影响，各自把 Top-1 推高 ~1pp（T1 +3.2pp d5 档最明显）；配方间差异全部在噪声边缘，**最终取舍必须等 A01 配对 CI**；`Tr_T4_swd` 是否追加取决于 T0–T3 CI 后是否出现真实胜者。
- 同150 盲测全部 96.7%，无配方损伤。

## 共享前置（完成）

`runs/v25/cache/train_latents_4400.pt`（577MB）：`{z0_tr, z_anc_tr(B4+canonicalizer=c1_gentle), z_p1_tr, z_d6_tr, sid_tr, tid_tr, teacher_spec}`。
**教师三元组（机制 A/B 全程锁定）**：`ckpt=slf_core_v2.pt + anchor=B4(T4-equivalent) + solver=D6H + branches=cs_j + gate=original`；`z_p1_tr` 来自 `s2_prism1_v2 λ=1.0`。

## 队列状态与下一步

- **当前**：GPU 空闲，等待 B01 交付 `exp/v25/mechanism_a.py / mechanism_b.py`（前置缓存已就绪）。
- **下一序**（agent2.md §二）：G2 机制 A1 启发式门控（零训练，~8 臂评测）→ G3 机制 A2/A3（1500 updates × 2）→ G4 机制 B0（3000 updates）→ G5 B1（6000 updates）→ G6 种子复验 → G7 端到端 + test750。
- D6 vs Hybrid 在 D-C≈0.73 的未决判定，将在机制 B 教师曲线实测覆盖 0.73 后由 A01 用真实工作点出 CI。
