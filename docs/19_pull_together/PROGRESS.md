# V25 Pull-Together 执行日志

> 格式：`[UTC+8 时间] 机器 — 动作 → 结果`。每完成一个检查点（CP1–CP4）更新本文件并提交。

## 2026-11-19

- [文档重组] 按用户指示将原 `docs/19_pull_together.md` 收编为 `docs/19_pull_together/` 目录：
  - `README.md`（索引与铁律）/ `01_convergence_plan.md`（收敛修订版：τ≤0.78 截断修正、非劣性等效判定、O3 归因后置、配方↔CFG 分支对应）/ `02_mechanism_plan.md`（V25-M 机制 A/B/C）/ `03_schedule_and_eta.md`（12h 排程）/ 本文件。
  - 事实核对：`exp/v22/core.py:353` 确认 `tau_v` clamp 上限 0.78，Round 1 曲线 B 改为 `tau_min ∈ {0.40, 0.50, 0.58, 0.66}` 起点上移扫描（`tau_hi` 固定 0.78），并要求每臂记录实测 τ 分布。
- [机器状态] 4090 空闲（0%, 1.8GB 残留）；3060 空闲（`O3_label_free` 已收尾，结果已入台账：t45 `0.6769/0.4666/11.33%`，t55 `0.7300/0.4575/8.83%`）。
- [启动] `exp/v25/` 三个脚本就绪并双卡开跑：
  - **4090 `tmux v25r1`**：`r1_inference_ablation.py`（P0-1 曲线 A/B/C + P0-2 CFG 分支 + P1-3 锚点 + P1-4 Gate，21 实跑臂 + 6 复用，每臂记录实测 τ 分布 `tau_records`）。运行日志 `runs/v25_r1.log`，增量 JSON `runs/v25/r1_inference_ablation.json`。
  - **4090 `tmux v25r2q`**：Round 2 排队接力（等 `v25r1` 结束自动启动 `r2_train_ablation.py`，Tr_T0–T3 训练 + cs_j 两档评测）。
  - **3060 `schtasks v25r3`**：`r3_oneshot_minimal.py`（τ 补扫 t40/t60 → O3_self_only 训练评测 → O3_label_free swap_dist + SG-25 盲测）。日志 `runs/v25_r3.log`。
  - 教训记录：3060 SSH 会话直启会被回收，`schtasks` 内联 `/TR` 引号转义损坏（exit=2）；改用 **bat 文件 + schtasks** 启动成功；python 需 `-u` 防缓冲。
  - 命名备注：Round 1 臂名 `B_d6_t058` 因浮点截断记为 `B_d6_t057`（0.58×100=57.999），以 JSON `tau_cfg` 字段为准。

