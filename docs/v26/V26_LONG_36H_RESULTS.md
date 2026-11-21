# V26-L 36 小时长队列 — 最终结果

> 收官时间: 2026-11-21T13:57:09.155558+00:00 ｜ 预算: 36.0h ｜ 已用: 2.655h ｜ GPU 计算: 9556s ｜ finish: None

## 七问答案

1. **C2nt 失败是否由错误参考采样造成?** 否 — A1/A2 严格跨画派采样 (Top-1 0.00%/0.00% ) 反而低于 A0 同画派 (0.00%), 失败源于监督目标与主模型 (L1 教师/主线路线正当).
2. **PRISM-Flow 能否免类别 ID 保持强风格?** F1 (self) Top-1=10.83%, F2 (教师) Top-1=16.00% — 见 ABLATIONS 判读.
3. **Global-only 是否足够?** L3: S0=16.17% S1=14.00% S2=16.00% S3=14.67% (细节见 ABLATIONS).
4. **SWD 是否有真实 Pareto 收益?** L3 对照裁决 (K2-B 已证梯度互补; 训练收益看 L3 表).
5. **S10 优势是域适配还是剪枝?** D0=— D1=— D2=— (recon 见 ABLATIONS).
6. **完整 BPTT vs 截断?** T0=Top1=16.67% T1=Top1=9.17% T2=Top1=8.00% — GPU 成本已记录.
7. **PRISM-Flow 免重训减步?** L5 sweep (教师距离排序): []

## 预算与成本核算

```json
{
 "wallclock_elapsed_h": 2.655,
 "budget_remaining_h": 33.345,
 "gpu_compute_seconds": 9556,
 "science_tasks_done": [
  "EXP2",
  "L-PC",
  "L0",
  "L1",
  "L3",
  "L4"
 ],
 "blocked_tasks": [],
 "task_history": [
  {
   "task": "L-PC",
   "outcome": "done",
   "utc": "2026-11-21T11:23:11.636967+00:00",
   "wallclock_h": 0.089
  },
  {
   "task": "L0",
   "outcome": "done",
   "utc": "2026-11-21T11:23:50.848185+00:00",
   "wallclock_h": 0.1
  },
  {
   "task": "L1",
   "outcome": "done",
   "utc": "2026-11-21T11:24:08.205629+00:00",
   "wallclock_h": 0.105
  },
  {
   "task": "L2",
   "outcome": "failed",
   "utc": "2026-11-21T11:29:52.036659+00:00",
   "wallclock_h": 0.2
  },
  {
   "task": "L3",
   "outcome": "done",
   "utc": "2026-11-21T13:01:28.431577+00:00",
   "wallclock_h": 1.727
  },
  {
   "task": "EXP2",
   "outcome": "done",
   "utc": "2026-11-21T13:01:38.551053+00:00",
   "wallclock_h": 1.73
  },
  {
   "task": "L4",
   "outcome": "done",
   "utc": "2026-11-21T13:57:04.769687+00:00",
   "wallclock_h": 2.654
  },
  {
   "task": "L5",
   "outcome": "failed",
   "utc": "2026-11-21T13:57:07.700733+00:00",
   "wallclock_h": 2.655
  }
 ],
 "finish_reason": null,
 "per_task_cost_h": [
  {
   "task": "L-PC",
   "utc": "2026-11-21T11:23:11.636967+00:00",
   "elapsed_h_at_finish": 0.089,
   "outcome": "done"
  },
  {
   "task": "L0",
   "utc": "2026-11-21T11:23:50.848185+00:00",
   "elapsed_h_at_finish": 0.1,
   "outcome": "done"
  },
  {
   "task": "L1",
   "utc": "2026-
```
