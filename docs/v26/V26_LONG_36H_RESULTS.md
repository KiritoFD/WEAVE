# V26-L 36 小时长队列 — 最终结果

> 收官时间: 2026-11-22T00:29:25.903668+00:00 ｜ 预算: 36.0h ｜ 已用: 8.94h ｜ GPU 计算: 32184s ｜ finish: science_incomplete (failed tasks routed through, budget remains)

## 七问答案

1. **C2nt 失败是否由错误参考采样造成?** 否 — A1/A2 严格跨画派采样 (Top-1 11.33%/12.00% ) 反而低于 A0 同画派 (16.17%), 失败源于监督目标与主模型 (L1 教师/主线路线正当).
2. **PRISM-Flow 能否免类别 ID 保持强风格?** F1 (self) Top-1=10.83%, F2 (教师) Top-1=16.00% — 见 ABLATIONS 判读.
3. **Global-only 是否足够?** L3: S0=16.17% S1=14.00% S4(Bary)=14.17% S2=16.00% S3=14.67% — Global-only 足够; RefAttn 无贡献 (S2≈S0), 回退 S0 最简配方.
4. **SWD 是否有真实 Pareto 收益?** 否 — 梯度尺度校准后 (82× 差距归一) S1/S4 低于 S0 2.0-2.2pp Top-1 (DINO-C 略升/LPIPS 略降, tradeoff 形态移动而非净增益); 簇间分化度探针 Bary 0.0133 ≤ VGG 0.0135.
5. **S10 优势是域适配还是剪枝?** 同 750 生成 latent 四解码: D0=Top1=18.67% D1(真图FT)=Top1=10.83% D2(S10真图FT)=Top1=12.83% S10=Top1=16.17% — 真图域 FT 对生成 latent 解码是负迁移 (LPIPS +0.11); S10 解码 23ms vs Full 121ms (5.2×) — S10 非瓶颈, 现役保留 (注意: D0 的 Top-1 优势限于 C2nt 臂 latent 分布).
6. **完整 BPTT vs 截断?** T0=Top1=16.67% T1=Top1=9.17% T2=Top1=8.00% — T1≈T2 (Δ0.9pp<噪声) → 截断足够, 全 BPTT 无独立价值; rollout 训练把 tradeoff 推向内容 (DINO-C 0.74) 牺牲 Top-1.
7. **PRISM-Flow 免重训减步?** 可以基本无损 — 10 步基线 16.17% → 4 步 RK2-mid rk2_04_mid: Top1=15.83% DINO-C=0.693049736097455 | eul_04_uniform: Top1=15.67% DINO-C=0.6871271977076928 (Top-1 −0.34pp, DINO-C +0.012, LPIPS −0.029, NFE −60%); 教师距离排序: rk2_04_mid 最优.
8. **SG-25 OOD (最终候选臂, 125 对)**: S0=0.1544 / δs=0.1711; F2=0.1566 / δs=0.1735 (C2nt 基线: swap=0.5349, δs=+0.2216) — 参考响应外推到训练五画派之外.
9. **20 条有向迁移路线 (Top-1 均值)**: S0=0.1617; F2=0.16; S0_seed23=0.1617; S0_seed45=0.16 — 逐路线矩阵见 ABLATIONS.
   CI[S0_minus_F2] Δ=0.001 95%[-0.0098,0.0101] p>0=0.542
   CI[S0_minus_S0_seed23] Δ=-0.0003 95%[-0.0093,0.0071] p>0=0.4255
   CI[S0_minus_S0_seed45] Δ=0.001 95%[-0.007,0.0078] p>0=0.5505
   CI[F2_minus_S0_seed23] Δ=-0.0011 95%[-0.0062,0.0058] p>0=0.2615
10. **难例对比图**: /root/Workspace/WEAVE/runs/v26/gpu4090/hard_examples/hard_examples_S0_vgg_global.png (worst routes: ['Impressionism->Early_Renaissance', 'Impressionism->Rococo', 'Minimalism->Rococo', 'Minimalism->Ukiyo_e']).

## 口径说明

- 队列状态 science_tasks_done 不含 L2/L5: 两者经 4 轮修复后由**队列外独立子进程**补跑完成, json 已入库 (l2_decoder_matched.json / l5_fewstep_q32.json) — 本文档数字以 json 为准;
- L2 评测 latent = A0_legacy_nbr_t45 (C2nt 臂) 的 750 输出; L5 init = L3 best S0_vgg_global;
- L5 Q32 gate: relational_retention 0.5749 < 0.70 → FAIL, Q32 匹配训练未消耗 4090;
- 全部评测: 完整 test750 读 cross600, S10 解码, 冻结裁判 (src/eval/measure).

## 预算与成本核算

```json
{
 "wallclock_elapsed_h": 8.94,
 "budget_remaining_h": 27.06,
 "gpu_compute_seconds": 32184,
 "science_tasks_done": [
  "EXP2",
  "L-PC",
  "L0",
  "L1",
  "L3",
  "L4",
  "L6"
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
  },
  {
   "task": "L6",
   "outcome": "done",
   "utc": "2026-11-21T13:57:09.473177+00:00",
   "wallclock_h": 2.655
  },
  {
   "task": "pool_seed23",
   "outcome": "done",
   "utc": "2026-11-21T14:23:31.954820+00:00"
  },
  {
   "task": "pool_seed45",
   "outcome": "done",
   "utc": "2026-11-21T14:49:58.581981+00:00"
  },
  {
   "t
```
