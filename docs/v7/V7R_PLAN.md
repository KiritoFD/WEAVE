# WEAVE V7-R：风格监督与条件变换的机制修复（追加 14h）

> 状态：**已实现，未启动 GPU**（等用户确认卡空后在 3060 tmux 串行执行）。
> 串行入口：`sbflow/scripts/run_v7r_14h.sh`（`--dry-run` 预检：编译 + CPU 单测 + 冻结输入检查）。
> 代码：`tools/v7r/`；产物：`runs/v7r/`；报告：`docs/v7/V7R_EXTENSION_REPORT.md`。
> V7 第一轮结果与全部冻结资产只读。

## 0. 背景与三个根本问题

V7 第一轮（`docs/v7/V7_REPORT.md`）确认：主体流程已执行但研究未完成——
T2–T4 内容保持优于强 R1 编辑但风格低于 B4；M1 恒等坍缩、M2 内容崩坏；
T5 为 SKIPPED_BY_GATE；C2/C3/C4 未执行；两小时即结束是 Gate 提前终止而非预算用尽。

V7-R 集中回答：

- **Q1** 真实风格能量能区分画派，为何不能提供有效生成梯度？
- **Q2** 相对风格变换未训练已有响应（B4-relative Top-1 31.5%），训练后为何退化（M1 2.2%）？
- **Q3** 能否保留 B4 的内容映射，学习有效的源→目标修正，并用测试时优化增强？

执行原则：**一个假设失败即切换预注册替代分支，绝不提前终止整条流水线**；候选标 `NO` / `SKIPPED_BY_GATE` / `MISSING`。

## 1. 阶段矩阵（14h 预算，实际估计 ~6h）

| 阶段 | 预算 | 内容 | 产物 |
|---|---|---|---|
| P0 | 1.5h | 失效归因：M1/M2 带 checkpoint 复现（0/100/400/800）轨迹审计；B4 150 源×5 ID 全量捷径审计 + 残差子带能量；c1_gentle 源风格探针（按画家分组） | `runs/v7r/p0_failure_audit.json`、`docs/v7/V7R_FAILURE_AUDIT.md` |
| P1 | 2.5h | P1-A：S1/S2/R1 × {正向/反向/随机/错风格} 12 个单步梯度探针，全部 750；P1-B：真实画作局部 patch 库（LL 4×4 + HF 4×4 + raw 8×8 投影，平滑最近邻能量）；P1-C：E0(原V6)/E1(V7-S1)/E2(S2校正)/E3(局部patch)/E4(E3+低频) 同预算 4 步对比，全部 750 | `p1_energy/patch_banks.pt`、`p1_gate.json` |
| P2 | 4h | 新 Predictor：`z0=B4(c,t); z_out=z0+Hθ(c,z0,[s],t)`（Hθ≈130K 参数，零初始化输出，B4 冻结）；N1=target-only vs N2=source+target（唯一差异）；3000 steps、有效 batch 240、bf16、warmup+cosine；ckpt 0/500/1500/3000 全部完整 750 评测。预注册替代：N1/N2 失败且 P0 确认泄漏 → c1_gentle 受限风格随机化路线（B4 初始化，V6 目标） | `p2_predictor/N1|N2[/RAND]`、`p2_gate.json` |
| P3 | 3h | Corrector：R0/R1c/R2c/R3c(P1有效)/R4c/R5c(最佳Predictor组合)+短路径 R1c_s1/s2 + 内容约束变体 R3c_noC/disp；每步记录梯度 RMS、方向、接受/拒绝 α、累计位移 | `p3_corrector/*`、`p3_gate.json` |
| P4 | 2h | 新 Predictor 有效 → 协同验证；否则按证据选一分支：EXTEND(6000 steps) / ROLLBACK / SECOND_SEED | `p4_joint/p4_decision.json`、`p4_verdict.json` |
| P5 | 1h | 统一统计：全 750；模型级 Pareto（同任务集）+ 逐任务配对差（vs B3 与 vs B4）；机制问题 A/B/C；最终报告 | `final_matrix.json`、`final_verdict.json`、`V7R_EXTENSION_REPORT.md` |

## 2. 对 V7 两处判定的收紧（本计划的研究前提）

1. **P1 的 GO ≠ P2 的目标有效**：T3/T4 梯度范数比 ~570 且 imbalanced=True，按 V7 自身协议属于梯度平衡未解决的候选，不能据此否定"R1+真实分布联合优化"。P1-A 用单步探针 + 反向/随机/错风格对照直接检验梯度方向的有效性；T3/T4 的失衡在 P1/P3 中以"step-0 定标 + 回溯线搜索 + POST-SCALE 范数审计"解决（预定义策略，一次性）。
2. **M1 同风格好是代数性质**：`z+A(c,t)-A(c,s)` 在 t=s 时恒等于 zs，Same-LPIPS 0.0745 只是 VAE 往返量级，不构成重建能力证据。P0-A 通过轨迹区分"差分被训练成零"还是"差分方向不再指向目标"。

## 3. 关键实现约定

- 相对 V7 的修正口径：DINO-C 分箱不再用作等内容比较（不同任务落同箱不可配对）；P5 只用 (a) 模型级 Pareto（全体 600 任务均值）(b) 逐任务配对差。
- V7 记账更正：T5=SKIPPED_BY_GATE；Same-LPIPS 0.0745=恒等代数性质——均写入 `V7R_FAILURE_AUDIT.md`。
- Hθ：12ch 输入 [c, z0, z0−c] + Style-ID embedding FiLM，3×FiLMResBlock(48)，输出 conv 零初始化 ⇒ step0 精确等于 B4；同风格任务直接沿用 B4（H 只作用于跨风格任务）。
- 训练损失（corrector）：`w_s·L_style + 0.525·L_content_inv + 1.0·||z_out−z0||² + 0.003·weakR1`；step-100 加权梯度审计失衡（>100×）时按预定义策略一次性重定标并记录。
- 风格随机化约束（替代路线，固定不调参）：LL 通道均值/方差 ×U[0.8,1.2]，HF 幅度 ×U[0.6,1.4]。
- E3/E4 平滑最近邻能量：patch 按库逐维标准化，τ=1.0；4×4(latent)+4×4(HF)（E3），+8×8 投影 patch 与 LL 库（E4）。
- P4.3 门槛不变（ΔDINO-S≥+0.020 & CI>0、DINO-C≥0.8125、LPIPS≤0.4712、Top-1 容忍、覆盖与 Same150 不退化）。

## 4. 冻结项

`tasks_750.json`、references/source caches、B3/B4 anchor、R1、c1_gentle 统计、
`runs/v6_r2/`、`runs/v6_w/`、`runs/v7/`、`docs/v6/`、`docs/v7/V7_REPORT.md` —— 全部只读。

## 5. 研究成功标准

除 P4.3 外的机制判定：
- **A**：新（局部）风格目标是否比 R1 分类边界提供更自然的编辑方向（P1-A 方向性 + P1-C + P3 Pareto）。
- **B**：显式源—目标条件是否确实改善训练（N2 vs N1 逐任务配对差）。
- **C**：保留 B4 的强内容映射再学有限修正，是否比重学相对输出更稳定（N-best vs V7 M1/M2）。

优先突破点：**B4 当前 DINO-S 0.4214 / DINO-C 0.8199 附近的 Pareto 边界**（在 DINO-C≈0.80–0.83 显著提高 DINO-S），而非 Top-1 冲高。


---

# V7-R+ 追加（2026-11-12 定稿，随 V7-R 一并执行）

> 在原 V7-R 五阶段之后按预注册顺序追加五个实验模块；预算目标 14h、硬上限 16h
> （BUDGET_TOTAL_SEC=57600）。顺序：`P0/P1 → P2(N1/N2) → E1 → E2 → E3 → E5 → E4 → P3 → P4 → P5`。
> 预算不足时的保留优先级：`P0 归因 > N1/N2 > E2 > E3 > E1 > E5 > E4 > 最终报告`；
> 报告与既有候选的完整评测永不跳过。不为填满预算而等待。

## E1：源风格条件因果干预（0.75h，无需重训）

固定已训练最佳 corrector（p2_gate.best_ckpt），完整 750 评测 5 个臂：
`E1_A`（正确源/正确目标）、`E1_B`（错误源）、`E1_C`（固定源=id0）、`E1_D`（错误目标，z0 随有效目标重建）、
`E1_R1src`（推理时用 R1(c) 预测源 ID 替代真值）。记录逐 trace 任务的
target/source 条件敏感度（固定内容与 z0，换 ID 的残差 RMS）。判定：
`E1_A − E1_B` 的 ΔDINO-S 配对 CI>0 ⇒ 源条件确实被利用；`E1_R1src − E1_A` 给出免真值依赖的代价。
汇总写入 `runs/v7r/e1_summary.json`。

## E2：内容条件去风格化训练（2h，两个新训练臂）

- E2-A = 原 N2 配方（复用，不重训，标记 REUSED）。
- E2-B：N2 step0 起步 + 受限随机外观扰动（LL×U[0.8,1.2]、HF×U[0.6,1.4]），无额外约束。
- E2-C：同 E2-B + 双视图残差一致性 `||H(c1,z0,s,t)−H(c2,z0,s,t)||²`，
  z0 用**同一原始 B4 结果**（不把 B4 对增强的响应混入）；一致性权重在 step-100
  审计按梯度尺度一次性对齐（clamp 1e-3..1e3）。训练前先测增强的内容代理变化
  （`augment_precheck.json`）。
- 各 3000 steps，ckpt 0/1000/2000/3000 全部完整 750 评测；探针新增五 ID 输出散布。
- 成功判据：源残留下降且 DINO-S 不降、目标风格建立能力提高——只减残留不算成功。

## E3：学习真实风格表征（2.5h）

- 冻结 R1 embed(192d) → MLP 192-256-128-64（≈91K 参数，L2 归一化）；
  `L = SupCon(同风格跨画家为正) + 0.5·CE`；800 steps，batch 240。
- E3-A 随机负样本；E3-B 困难负样本（dino_224 CLS 离线挖掘的跨风格内容相似图，每锚每风格 top-4）。
- Phase-1（画家分组）：val 分类、同风格检索@1、相似度间隔、困难负样本判别、R1 原始表征对照。
- Phase-2：真实嵌入高斯能量（1−cos + 0.5·(e−μ)ᵀΣ⁻¹(e−μ)，可微）作为编辑目标，
  臂 `p1c_E3A_repr` / `p1c_E3B_repr` 从 B4 起 4 步预算编辑、完整 750；完成后重跑 p1 gate，
  只有生成指标改善才允许进入 P3 的 R3c 目标池。

## E4：训练充分性与重复性（2h，按曲线预注册分支）

对全部已训练 corrector 臂（N1/N2/E2_B/E2_C/E5/种子臂）的完整 ckpt 评测曲线：
- A 仍在改善（末段 ΔDINO-S>+0.0005 且内容未失控）→ 续训至 6000（ckpt 4000/5000/6000 全量评测）。
- B 先升后降（早期优于最终 +0.002）→ 回滚最佳 ckpt + 低学习率（1e-4）1500 步续训对照。
- C 停滞 → 预注册替代级联：E2_C → E5 → E2_B 训练下一未训臂。
- 种子规则：任一臂相对 B4 出现 ΔDINO-S ≥ +0.001 的正改善 ⇒ 必须补 seed43 复训。

## E5：Core64 采样偏置消融（1.25h）

`E5` = N2 配方但 `--core-frac 0.0`（全部从全池均衡采样），其余完全相同；3000 steps，
ckpt 0/1000/2000/3000 全量评测。重点观察 ER/Ukiyo-e、五风格 DINO-S 均衡性、
向 Imp/Min 原型集中的趋势（MinOff）、同等 DINO-C 下的风格收益（逐任务配对差）。

## 执行与记账

- ETA 四元组：`BUDGET_TOTAL_SEC=57600` / `ELAPSED_SEC` / `GPU_ACTIVE_SEC_EST`（实测吞吐汇总）/
  `ETA_REMAINING_SEC`（剩余工作 × 实测单件耗时），每阶段后更新 `runs/v7r/eta.json`。
- 训练吞吐与 750 评测墙钟在启动后前 100–200 step / 首次评测实测，动态重算 ETA。
- E3 的最近邻计算按 512 行分块，避免 12GB 上巨型两两距离矩阵。
- 最终只保留一份报告 `docs/v7/V7R_EXTENSION_REPORT.md`（整合 runs/v7r/ 全部结果）。

## 最终验收问题（在原 A/B/C 之上）

- A. 源风格条件是否真被利用（E1 + N1/N2 对照）。
- B. 条件去风格化能否提升真实目标风格而不只是减源风格（E2）。
- C. 学习到的画派表征是否提供比 R1 更有效的生成方向（E3 phase-2）。
- D. 训练失败是优化不足、崩塌还是监督失效（E4 + P0 轨迹）。
- E. Core64 原型化采样是否限制目标风格覆盖（E5）。
