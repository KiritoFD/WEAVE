# V10 进展与交接（Content-Conditioned Local Style Score）

> 预注册计划：[WEAVE_V10_CC_SCORE_4070LAPTOP_12H.md](WEAVE_V10_CC_SCORE_4070LAPTOP_12H.md)；
> **最终结果：[V10_RESULTS.md](V10_RESULTS.md)**。
> 本文件是进展事实源；**没有跑出的数字不写进本文件**。
> 产物根 `runs/v10_local_4070/`（untracked）。
>
> **口径修正（2026-11-14，用户决定）**：任务级评测/决策一律用**完整 test750**
> （读数取 cross600），**不用 dev 预览集**。LD0–LD4 基线由 v9 dev+test eval 行合并重算。
> 完整 raw 图集通过 `datasets/wikiart_raw` 符号链接（J:\WEAVE_DATA）可访问；
> 数字命名与池子 id 的映射 manifest 待查（开放项，仅影响探针/出图）。

## 状态总览（最终）

| 阶段 | 状态 | 关键产物 / 结论 |
|---|---|---|
| P0 特征+探针 | ✅ | `p0_report.json`：split 完整性✓（源图零重叠复验，5000 幅 d5 画作确属 train）；吞吐 1129 latents/s；S10 解码链健康（MAE 1.6–11.2/255、LPIPS 0.033–0.129） |
| P1 bank+hψ+KDE | ✅ | `bank/kde_bank.pt` 215MB（895,688 train patches）；hψ MSE 0.509→0.116 |
| P2 实验 A/B/C | ✅ 全过 | A 门槛过（V10-B .302 > ld3 .122 > r1 .054，5/5）；B 非颜色账（texture .265 > color .227）；C 单调 −0.98✓、条件探针 +10.8pp、混淆 57.7% |
| P3 V10-E（750） | ✅ | E4 最优：Top-1 +1.17pp vs LD0、dino_s/affinity 全场最高；**未超 LD4**；post-hoc E2_mix null、E2_ld4plus 负（−1.0pp） |
| P4 DSM | ✅ keep_kde | 修复 2 个实现 bug 后 DSM .204 < KDE .302（σ=0.1 分支欠拟合） |
| P5 750 级结论 | ✅ | `V10_RESULTS.md`（含互补性否定、下一步 v10.1 候选） |

## 过程事故记录（写给后续会话）

1. **NaN（Minimalism）**：`_rms/_std` 在完全平坦色块区域（d5 Minimalism）方差恰为 0，
   sqrt 反向 = NaN。修复：clamp BEFORE sqrt（`features.py`）。
2. **d5__ 前缀泄漏疑云**：池含 5,000 幅 `d5__` id，初版 stem 重叠检查因前缀不同漏报
   风险；归一化前缀后复验 **0 泄漏**（全是 d5 train 画作，非 150 评测源）。
3. **DSM 方向伪反向**：`L=D·f−f²/2` 未 detach D，雅可比项污染 → −0.50；修复后 +0.20。
4. **DSM 训练尖峰**：per-σ 损失尖刺（lr 1e-3→3e-4 + clip 1.0 缓解未根除）；
   σ=0.1 分支欠拟合（holdout model mse 0.117 vs 恒等 0.010）。
5. **口径修正**：dev 预览集（136 cross，Top-1 粒度 0.74pp）由用户决定弃用；
   E 臂全部按 750 重跑；v9 基线 dev+test 合并重算。
6. **wikiart_raw 命名**：J:\WEAVE_DATA 为数字命名 + tgz，与池子 `Style__artist_painting`
   命名的映射 manifest 缺失（开放项）。

## 已知坑（沿用）

- KDE 全库距离 chunk ≤1024 queries；score 累积 fp32。
- decode+measure 分进程（referee 显存独占）。
- `runs/v10_local_4070/` 大文件不入 git；manifest.json 记录 sha256 与超参。
