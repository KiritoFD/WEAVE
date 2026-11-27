# V26-L：36 小时连续研究队列（4090 长队列，主攻 PRISM-Flow 任意参考化）

> **状态**：执行中（自 `c51cf2f` 之后的独立 36h Wall-Clock 预算）｜**前置**：V26 诊断阶段完成（K0/K1/K2/SR-CFG/少步/SG-25 **不重复**）
> **核心目标**：**将闭集 PRISM-Flow 转化为严格任意参考的 One-Shot 模型，并通过受控实验确定最简有效配方。**
> **设备约束**：4090 = 主执行队列（共享 GPU，按 **~12GB 可用显存** 设计）；3060 = 机会性执行（外部 17 进程竞争，**不计入保证预算**）；子 Agent 并发 ≤2；无 worktree。

---

## 一、为什么重启长队列（V26 诊断阶段的边界）

1. `queue_k.py` 实际只执行了 K0→SR-CFG→K1→K2；**K3–K8 从未入队**——诊断阶段完成 ≠ 36h 研究计划完成；
2. **V26 的 S1fix 只修复了 accum RNG，未修复参考池**：`assign_c2nt_refs` 仍调用同画派构造的 `nbr_idx`（K1 已证明）——"跨风格训练"从未得到一次干净验证；
3. 因此 L0 必须先做这个因果对照，然后主攻 PRISM-Flow 任意参考化。

## 二、实验排程（每阶段为上限，提前完成即进入下一阶段）

| 阶段 | 时间 | 核心实验 | 交付 |
|---|---|---|---|
| **L0** | 0–4h | 真实跨画派参考采样因果实验（A0/A1/A2） | 训练失败根因裁决 |
| **L1** | 4–13h | PRISM-Flow 任意参考化四臂（F0/F1/F2/F3） | 新 One-Shot 主线 |
| **L2** | 13–19h | Full Decoder 同域匹配微调（D0/D1/D2） | S10 因果消融 |
| **L3** | 19–26h | 动态 SWD × 全局/局部参考分支消融（S0–S3） | 风格监督机制 |
| **L4** | 26–31h | 完整可微 Rollout 上界（T0/T1/T2） | 训练—推理一致性 |
| **L5** | 31–34h | PRISM-Flow 少步轨迹归因；条件 Q32 验证 | 减法与理论裁决 |
| **L6** | 34–36h | 关键复验、统计与资产归档 | 最终结果 |

## 三、关键实验设计要点

### L0（A0/A1/A2，各 1200 updates + 全量 test750）
- A0 = 旧同画派近邻参考；A1 = **严格跨画派均匀采样**；A2 = 跨画派 + 目标画派 Core 典型性筛选；
- **不允许缺跨画派邻居时静默回退同画派**；显式断言 `source_style != reference_style`；
- 统计实际 5×5 配对矩阵；同初始化/同有效 batch/同损失；额外验证参考更换敏感性；
- 定位：**故障归因实验**，不是把 C2nt 升回主线。无论结果如何 L1 继续。

### L1（F0/F1/F2/F3）
- 初始化复用 V20/V22 Flow ckpt（`SLFUNet` + `StyleFiberEncoder` + `RefAdapter` + S10），不从零训练；
- F0 = 原五类条件闭集上界（不参与严格 One-Shot 比较）；F1 = 全局参考 Fiber；F2 = F1 + **跨画派 Flow 教师监督**（教师训练期可用类别标签，论文如实声明）；F3 = F2 + 结构保护局部 RefAttn（仅 F2 证明有价值才跑）；
- F1/F2/F3 推理**禁止输入目标画派 ID**；先冻结主干训 Adapter+调制层，F1/F2 在 1200–2000 upd 稳定后再扩解冻；
- 验收：Cross600 四指标 + 20 路线 + 正确/错误参考 + SG-25 OOD + Same150 无 bypass + 端到端成本；
- **若 F2 不加局部注意力更好，直接保留 Global-only 更简模型**（K1 证据：局部 Value 损害内容）。

### L2（D0/D1/D2，独立于 L1 成败必做）
- D0 = 原 Full SD-VAE 冻结；D1 = Full 同域微调；D2 = 剪枝 S10 同域微调；同数据划分/同重建目标/同步数；
- **完全相同的生成 latent** 解码对比：DINO-C/S、LPIPS、CSD Top-1、高频频谱、局部边缘、重建质量、4090 解码耗时；
- 未收敛 → INCONCLUSIVE + 优先延长 matched 训练，不得按早期曲线下结论。

### L3（S0–S3，从 L1 最佳初始化出发，全部真正跨画派采样）
- S0 VGG均值×Global / S1 SWD×Global / S2 VGG均值×Global+RefAttn / S3 SWD×Global+RefAttn；
- SWD 先在训练初期**校准梯度尺度**（K2-B：‖∇SWD‖≈20×‖∇meanVGG‖），保证公平对照；
- 1200 upd 初赛 → 胜出臂延长 4000–6000 + 第二种子；不能只看 DINO-S，必须同时监控 DINO-C/LPIPS/困难风格/灰化。

### L4（T0/T1/T2）
- 以 L1/L3 胜出者为初始化；单步预览 vs 4 步截断 vs 4 步全反传；
- **按 12GB 约束**：100-step 探针 → activation checkpointing + microbatch 1–2 + 梯度累积 → 仍不行则降级两步全反传并如实记录。

### L5
- 首选：**PRISM-Flow 轨迹诊断**（原步数/8/6/4 步 RK2 + 非均匀时间网格 + 早/晚 CFG 调度，先训练内部确定网格再全量验证）；
- 条件替代：Q32 匹配消融——**先解决 K0 报告与 SUMMARY 的判定不一致**（结构保持度量退化问题），建立跨维度结构保持测量并证明 Q32 不损坏内容后才启动，否则不再消耗 4090 时间。

### L6
- 第二种子复验、全量 test750、SG-25 OOD、Same150 无 bypass、20 路线矩阵、源图聚类配对 CI、复杂度与真实 NFE、难例对比图、资产归档；
- 最终模型必须明确归类 `GLOBAL_REF_ONLY` / `GLOBAL_REF_PLUS_SWD` / `LOCAL_REF`，**禁止拼接多 checkpoint 最优指标**。

## 四、队列机制（写入执行代码，防止提前宣布完成）

- 新建 `exp/v26/gpu4090/queue_k_long.py`（**不覆盖**已完成的 `queue_k.py`）；
- 两级任务：一级 = L0–L6 科学必做；二级 = 预算填充池（第二/三种子 → 训练长度消融 1200/3000/6000/9000 → 训练集规模 25/50/100% → 困难路线均衡采样对照 → SWD K/尺度/梯度平衡 → 多参考敏感性 → Decoder 追加训练）；
- 状态 JSON 必含：`science_tasks_done` / `budget_remaining` / `gpu_compute_seconds` / `wallclock_elapsed` / `pending_replications` / `blocked_tasks` / `valid_results_count`——**只有必做任务完成 + 高价值备用任务处理 + 预算窗口结束（或明确资源阻断）才允许标记完成**；提前完成须报告剩余预算与无法继续的原因；
- 每任务三态路由 `success_next / failure_next / inconclusive_next`（失败不断链：L0 无收益→L1 照常；L1 未达标→最佳 F1/F2 进 L3；L2 未收敛→留 ckpt 备用预算续；L3 无收益→L4 用均值基线；L4 OOM→自动降级两步/截断；L5 Q32 不过闸→继续少步曲线）；
- 隔离：不触碰外部进程；启动前读可用显存留余量；独立 run_id/config hash；每 500–1000 upd 保存可恢复状态；不用 test750/SG-25 调参；每阶段完结立即拉回 + 哈希核对；无 worktree，仅总控提交。

## 五、前置 CPU 修正（Agent A，不占 GPU）

1. **K2-A 更正**：原始 margin 比值 ≠ 判别力倍数（三种距离数值尺度不同）——补 AUC、标准化类间间隔；修正逐画派 one-vs-rest margin 实现 bug（原实现只保留两端都为 s 的对 → m_out 恒空 → 负值）；
2. **K2-C 更正**：全题材 Rococo=0% 不能单独排除 CSD 题材偏置——补**真实 Rococo 图像正对照**（真图 CSD 天花板）与独立风格感知评价。

## 六、最终验收七问 + 交付

七问：① C2nt 失败是否源于错误参考采样 ② PRISM-Flow 能否免类别 ID 保持强风格 ③ Global-only 是否足够（局部 RefAttn 是否应删）④ SWD 是否有真实 Pareto 收益 ⑤ S10 优势是域适配还是剪枝 ⑥ 完整 BPTT 相比截断有无独立价值 ⑦ PRISM-Flow 能否免重训减步。

交付：`docs/v26/V26_LONG_36H_RESULTS.md`、`V26_LONG_ABLATIONS.md`、`V26_LONG_NEGATIVE_RESULTS.md`、`V26_FINAL_MODEL_RECIPE.md`、最佳模型 + 两种子、全量 test750/SG-25/配对 CI、Full Decoder 消融、4090 真实利用时间与逐实验成本。

**禁止以"诊断阶段有结论"代替 36h 长队列完成判定。**

## 七、优先级

**真正跨画派采样 ＞ PRISM-Flow 任意参考化 ＞ Decoder 严格对照 ＞ SWD 实训 ＞ 完整 Rollout ＞ Q32。**
