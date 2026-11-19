# V25 Pull-Together 执行日志

> 格式：`[UTC+8 时间] 机器 — 动作 → 结果`。每完成一个检查点（CP1–CP4）更新本文件并提交。

## 2026-11-19

- [文档重组] 按用户指示将原 `docs/19_pull_together.md` 收编为 `docs/19_pull_together/` 目录：
  - `README.md`（索引与铁律）/ `01_convergence_plan.md`（收敛修订版：τ≤0.78 截断修正、非劣性等效判定、O3 归因后置、配方↔CFG 分支对应）/ `02_mechanism_plan.md`（V25-M 机制 A/B/C）/ `03_schedule_and_eta.md`（12h 排程）/ 本文件。
  - 事实核对：`exp/v22/core.py:353` 确认 `tau_v` clamp 上限 0.78，Round 1 曲线 B 改为 `tau_min ∈ {0.40, 0.50, 0.58, 0.66}` 起点上移扫描（`tau_hi` 固定 0.78），并要求每臂记录实测 τ 分布。
- [机器状态] 4090 空闲（0%, 1.8GB 残留）；3060 空闲（`O3_label_free` 已收尾，结果已入台账：t45 `0.6769/0.4666/11.33%`，t55 `0.7300/0.4575/8.83%`）。
- [启动] （待填：exp/v25 Round 1 脚本 + 双卡启动记录）
