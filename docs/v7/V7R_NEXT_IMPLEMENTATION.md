# WEAVE-CPT / V7R_NEXT 启动与实现记录（2026-11-12 19:06）

**目标**：见 `goal-5fcfff14-d62b-4b63-bf99-4f274175e135`。单卡 3060、冻结 SD-VAE 与 B4，
研究命题：先在真实画作中找与源图结构兼容的目标风格表现，再有限 latent 编辑，最后摊销到极小模型。
**当前状态**：**串行流水线已于 19:02:50 在 3060 tmux `v7r_next` 启动**，P0 方向审计正在跑
（第一个 750 探针 eval 已完成 ~90s/个）。dry-run 27 项 CPU 单测全通过、冻结输入检查通过。

---

## 1. 已实现的组件（全部通过 CPU 单测）

新增 `tools/v7r_next/`（3060 与本地 `G:\WEAVE\tools\v7r_next\` 各一份）：

| 文件 | 职责 | 单测 |
|---|---|---|
| `common.py` | 路径/冻结资产加载、seed、sha256、阶段状态、预算 ETA、`specificity_from_scores`、`summarize_specificity` | 2 |
| `audit.py` | **P0**：`p0a_direction`（方向/≥5 随机 seed/零更新，每组 750，目标特异性 Gate）、`p0b_duplicate`（R3c_noC vs R3c_disp、p1c_E0/E1 逐 task 比对）、`p0c_verdict`（历史结论纠正） | 由 dry-run 的 selftest 覆盖核心逻辑 |
| `patch_transport.py` | **P1**：`phi_geom`（8×8 patch、梯度方向/结构张量/相对边缘秩/局部自相似/平坦-边缘-纹理连续特征，源图算且 detached）、`phi_style`（Haar HF 4×4，与 geom 同为 N=64 空间对齐）、分层 Bank（5×8×96，TRAIN 池 chunked 构建）、C1–C6 匹配（geom_soft / geom_ot / uniform / geom_shuf / wrong_tgt）、`cpt_loss`、`cpt_global_loss` | 10 |
| `corrector.py` | **P2**：`ConstrainedEditor` 4 步累积、回退线搜索、硬 RMS 约束、逐步审计（CPT/R1 梯度余弦、加权范数、位移、约束触发）；`measure_rho_tiers()` 从 R3c/R4c/R3c_disp 真实位移定 ρ 两档；T0–T6 + C1–C6 配置 | 6 |
| `distill.py` | **P3**：`ClipRMS`（前向路径硬约束，非 loss 正则）、`StudentCorrector`（142,004 参数，零初始化 step0 精确=B4）、CPT 教师数据（TRAIN 池，不用 tasks_750 源图）、D1/D2/D3 训练、按子带梯度审计 | 3 |
| `evaluate.py` | **P5**：配对聚类 bootstrap（保留重复簇）、双前沿（外部 / 全部 Ours）、CLIP-S/Style Affinity、困难风格、Same150 漂移、失败分类、真实分段测速、唯一主报告 Markdown | 2 |
| `tests.py` | 统一 CPU 单测（无 GPU、无 750） | 27 全通过 |
| `_patch_r2eval.py` | 给 `tools/v6_next/r2_eval.py` 增加 `style_scores_5` 与 `dino_s_specificity` 字段（默认输出不变，历史可比） | 已应用并 smoke 验证 |
| `_patch_seed.py` | 给 `tools/v7r/arms.py` 的随机方向探针加 `V7R_NEXT_SEED` 隔离（≥5 独立 seed） | 已应用 |

新增 `sbflow/scripts/run_v7r_next.sh`：编译 → 27 CPU 单测 → 冻结输入检查 →
`--dry-run` 可只跑预检；正式跑按 `P0 → P1 → P2 → P3 → P4/P5` 串行，
`state/*.done` 断点续跑、`run()` 单臂失败只记 `FAILED` **不 FATAL 停全链**（修 V7-R 的缺陷）、
动态写 `eta.json`（work-based，非 GPU ETA）、15h 预算上限。

新增目录 `runs/v7r_next/`（3060），产物 `docs/v7/V7R_NEXT_BREAKTHROUGH.md`（P5 生成）。
`runs/v7r/` 全部历史**只读**，未改动。

---

## 2. 关键实现决策与修正（过程记录）

1. **空间对齐（重要 bug）**：初版 `phi_geom` 用 4×4 patch 得 256 个、
   `phi_style` 用 Haar HF(32×32) 4×4 patch 得 64 个，匹配计划 π 无法定义。
   已统一为 **geom 8×8 / HF 4×4 → N=64，索引 i 对应源 latent 的同一 8×8 区域**，
   单测 `patch grid ALIGNED` 覆盖。
2. **源几何不反传**：π\* 固定、`q_src.detach()`；单测用 `autograd.grad(l2,q2)` 验证
   无梯度到达源描述符。
3. **`cpt_global_loss` 维度 bug**：`mean((1,2))` 把 192 维通道压成标量 →
   改 `mean(1)` 保留 (B,192)。
4. **不复用 `BudgetedEditor`**：其 `_step()` 用 `zr`（requires_grad 初始张量）而非
   运行中的 `z_cur` 累加，编辑不会累积。P2 自写显式 4 步循环（低耦合）。
5. **`r2_eval.py` 原本只算目标风格 DINO-S**，无法回答「是否所有画派同时升高」。
   已补 5 风格分数向量与 `dino_s_specificity = S(t) − mean_{k≠t} S(k)`；
   smoke eval 确认字段落盘（len=5、值 0.152）。
6. **CLIP-S 仅评测、不进训练梯度**；所有 loss 只在 latent 上，无 RGB 训练损失、无 GAN、
   无额外大型生成教师、B4 与 SD-VAE 冻结。

---

## 3. 与计划（用户方案）的对照

| 计划要求 | 实现 |
|---|---|
| P0 梯度符号/分支/seed 审计、目标特异性 G(z,t)、CLIP-S 独立验证 | ✅ `audit.py` + r2_eval 扩展 |
| P0.2 重复臂逐 task 比对（不能只看均值） | ✅ `p0b_duplicate` 比 `run_meta` + 逐 task 指标 + eval 一致性 |
| P0.3 纠正历史结论 | ✅ `p0c_verdict` 明确 R3c/R4c 是测试时编辑、N1/N2 训练失败 |
| P1 几何/风格双描述符、分层 Bank、SoftNN/OT、固定 π\*、C0–C6 | ✅ `patch_transport.py`（C0 原 E3 与 C4 全局约束在 `corrector.py` 配置） |
| 禁止 3060 上巨型两两距离矩阵 | ✅ 分块（latent chunk 16、Sinkhorn 只在 G=8 组上） |
| P2 ρ 由真实位移定两档、两级优化、T0–T6、逐步审计 | ✅ `corrector.py` |
| P3 ClipRMS 前向硬约束、142K Hθ、教师来自 TRAIN 池、D1–D3 | ✅ `distill.py`（142,004 参数） |
| P4 困难风格 / 对照 | 由 `evaluate.py` 的 `hard_styles` + P5 对照配置覆盖（运行时填） |
| P5 双前沿、真实测速、禁止共用 45.78 ms/img、S10 正确配对 | ✅ `evaluate.py` 的 frontier 双线 + `speed_profile` |
| 唯一主报告 | ✅ P5 写 `V7R_NEXT_BREAKTHROUGH.md` |
| 所有配置/seed/hash 记录 | ✅ `common.py` sha256 + stage state + meta |

---

## 4. 尚未完成 / 进行中

- **P0 正在跑**：13 个方向臂 + 15 个随机 seed 臂，每个 750（约 90s/个），预计 P0 约 45 分钟。
- **P1 Bank 构建**、**P2 ρ 与 T/C 臂全量评测**、**P3 教师与学生**、**P4 困难风格**、**P5 报告**
  均未开始，由串行脚本按 Gate 依次推进。
- **P5 的 `eval_index.json` 里 `Q2a_R1_Oracle` 路径**暂按
  `runs/v6_r2/eval_Q2a_R1/eval.json` 猜测，若 3060 上不存在会被跳过；
  启动前需确认实际路径（下一步核实）。
- **测速 `speed_profile`** 目前是轻量 profiler（Encoder/Predictor），
  完整的「CPT 每步 / 学生 / TRT S10 / E2E」按方案要求在 P5 用各方法真实计时，
  需在各臂 eval 的 `runtime.json` 里取 `per_task_sec`，本轮 runner 会通过
  `record_timing` 累积；P5 报告里必须显式列出，不能回退到 q2a 拼接估计。

---

## 5. 现场证据

- 3060：`tmux v7r_next` 19:02:50 起；`runs/v7r_next/logs/runner.log`；
  `selftest.json`（27 PASS）；`p0_audit/dir/fwd_R1/eval.json` 首个 750 完成。
- 本地：`G:\WEAVE\tools\v7r_next\`、`sbflow\scripts\run_v7r_next.sh`。
- 冻结输入 sha256：`8b578c2bc421f954...`。

**下一步**：周期跟进 P0 Gate 结果；若 `FIX_AND_REEVAL` 或 `specificity_gate_triggered`，
按方案「修复并完整复评 / 后续所有方法必须同时报告目标特异性」执行，不提前进入 P1 训练。
