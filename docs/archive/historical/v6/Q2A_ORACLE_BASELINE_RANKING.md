# Q2A+Oracle1 × 新 SD0-S10 Decoder：docs/baseline 32 方法排行榜定位

**日期**：2026-11-12  
**协议**：`paper_rgb_v1`，cross600（150 源图 × 4 跨风格目标），与 `docs/baseline/UNIFIED_FULL_RANKING.md` 完全一致。  

**指标定义**（来自 `tools/evaluation_v2/metrics.py` 与 `tools/vae_swap_eval/style_metrics.py`）：
- `DINO-S` = 生成图 DINOv2-small CLS 与**目标风格 30 张冻结参考 CLS** 的最大余弦（不是与源图比）
- `CLIP-S` = 生成图 CLIP ViT-B/32 与**目标风格 30 张参考 CLIP 归一化均值原型**的余弦
- **`Style Affinity = (DINO-S + CLIP-S)/2`** —— 排行榜主指标
- `DINO-C` = 生成图 vs 源图 round-trip 的 DINO CLS 余弦；`LPIPS` = AlexNet LPIPS(vs 源图 round-trip)
- `CSD Top-1` / `Margin` 用 `deliver/testset/refs/csd_proto.pt` 五风格原型

**协议校验**：本脚本重算 sdvae/Oracle1 得 `affinity=0.5548, dino_s=0.4281, clip_s=0.6814`，与榜单第 5 名 `0.5548 / 0.4282 / 0.6814` 逐位吻合 → 本轮数字可直接与 32 方法比。

## 1. Style Affinity 主榜（32 baseline + 2 个新 Decoder，降序）

| 新排名 | 方法 | 类别 | VAE | Affinity | DINO-S | CLIP-S | DINO-C | LPIPS | CSD Top-1 |
|---:|---|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | StyleShot | Baseline | native / free | 0.6949 | 0.5750 | 0.8148 | 0.3569 | 0.7697 | 92.6% |
| 2 | StyleID | Baseline | native / free | 0.6469 | 0.5143 | 0.7795 | 0.3885 | 0.5554 | 73.2% |
| 3 | AesPA | Baseline | native / free | 0.5784 | 0.4661 | 0.6906 | 0.6075 | 0.6247 | 31.0% |
| 4 | Q2a + R1 Oracle (1 step) + S10 | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | 0.5587 | 0.4360 | 0.6814 | 0.7345 | 0.5014 | 36.3% |
| 5 | Q2a + R1 Oracle + SD0-S10 C1 (本轮新Decoder) **NEW** | Ours | SD0-S10 C1 (10.7M) | 0.5576 | 0.4321 | 0.6832 | 0.7324 | 0.5015 | 36.0% |
| 6 | Q2a + R1 Oracle + SD0-S10 C0 (本轮新Decoder) **NEW** | Ours | SD0-S10 C0 (10.7M) | 0.5575 | 0.4323 | 0.6827 | 0.7324 | 0.5022 | 36.8% |
| 7 | Q2a + R1 Oracle (1 step) + SDVAE | Ours | A0 SD-VAE (标准 49.5M) | 0.5548 | 0.4282 | 0.6814 | 0.7307 | 0.5222 | 34.8% |
| 8 | V6 ScoreDiff N=24 + CEDIT | Ours | C_edit 联合重训 VAE | 0.5541 | 0.4317 | 0.6766 | 0.6384 | 0.5714 | 24.5% |
| 9 | V6 ScoreDiff N=12 strong + CEDIT | Ours | C_edit 联合重训 VAE | 0.5513 | 0.4288 | 0.6739 | 0.6845 | 0.5185 | 18.7% |
| 10 | Q2a + R1 Oracle (1 step) + CEDIT | Ours | C_edit 联合重训 VAE | 0.5491 | 0.4424 | 0.6558 | 0.7138 | 0.6758 | 19.0% |
| 11 | P1_B4_distill50 (218K) + SDVAE | Ours | A0 SD-VAE (标准 49.5M) | 0.5483 | 0.4217 | 0.6749 | 0.8199 | 0.4523 | 21.3% |
| 12 | P1_B4_distill50 (218K) + S10 | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | 0.5483 | 0.4242 | 0.6725 | 0.8152 | 0.4396 | 22.2% |
| 13 | V6 ScoreDiff N=24 + S10 | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | 0.5459 | 0.4172 | 0.6746 | 0.6369 | 0.5217 | 28.5% |
| 14 | P1_B3_distill20 (218K) + SDVAE | Ours | A0 SD-VAE (标准 49.5M) | 0.5455 | 0.4186 | 0.6724 | 0.8325 | 0.4397 | 17.5% |
| 15 | P1_B3_distill20 (218K) + S10 | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | 0.5454 | 0.4207 | 0.6702 | 0.8270 | 0.4282 | 18.8% |
| 16 | V6 ScoreDiff N=12 strong + S10 | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | 0.5453 | 0.4165 | 0.6741 | 0.6797 | 0.4738 | 22.8% |
| 17 | V6 ScoreDiff N=12 strong + SDVAE | Ours | A0 SD-VAE (标准 49.5M) | 0.5400 | 0.4109 | 0.6690 | 0.6776 | 0.5219 | 27.5% |
| 18 | V6 ScoreDiff N=24 + SDVAE | Ours | A0 SD-VAE (标准 49.5M) | 0.5392 | 0.4105 | 0.6678 | 0.6308 | 0.5751 | 32.0% |
| 19 | P1_B4_distill50 (218K) + CEDIT | Ours | C_edit 联合重训 VAE | 0.5390 | 0.4262 | 0.6518 | 0.7843 | 0.6535 | 13.0% |
| 20 | P1_B3_distill20 (218K) + CEDIT | Ours | C_edit 联合重训 VAE | 0.5369 | 0.4231 | 0.6507 | 0.7947 | 0.6504 | 12.3% |
| 21 | WEAVE (Published) | Baseline | native / free | 0.5367 | 0.4003 | 0.6731 | 0.8079 | 0.4974 | 16.7% |
| 22 | StyTr2 | Baseline | native / free | 0.5317 | 0.4074 | 0.6560 | 0.7023 | 0.7386 | 39.5% |
| 23 | Seedream 4.5 | Baseline | native / free | 0.5314 | 0.3885 | 0.6744 | 0.7343 | 0.5202 | 37.3% |
| 24 | SaMAM | Baseline | native / free | 0.5199 | 0.3832 | 0.6565 | 0.8239 | 0.3882 | 2.5% |
| 25 | V6 D1 Gentle (s42) | Baseline | native / free | 0.5152 | 0.3829 | 0.6474 | 0.8994 | 0.3639 | 3.2% |
| 26 | V6 ScoreDiff (lr0.055, s42) | Baseline | native / free | 0.5146 | 0.3808 | 0.6483 | 0.9024 | 0.2370 | 5.0% |
| 27 | V6 ScoreDiff (lr0.048, s42) | Baseline | native / free | 0.5115 | 0.3781 | 0.6449 | 0.9188 | 0.2079 | 4.0% |
| 28 | CUT | Baseline | native / free | 0.5093 | 0.3755 | 0.6432 | 0.8435 | 0.6193 | 5.7% |
| 29 | V6 D1 Calibrated (a0.70, s42) | Baseline | native / free | 0.5074 | 0.3759 | 0.6389 | 0.9345 | 0.2508 | 2.0% |
| 30 | Identity | Baseline | native / free | 0.4902 | 0.3633 | 0.6170 | 1.0000 | 0.0000 | 0.5% |
| 31 | SD-Turbo | Baseline | native / free | 0.4894 | 0.3633 | 0.6155 | 0.9998 | 0.0048 | 0.5% |
| 32 | AdaIN | Baseline | native / free | 0.4853 | 0.3426 | 0.6279 | 0.2271 | 0.7641 | 34.3% |
| 33 | SaMST | Baseline | native / free | 0.4288 | 0.2738 | 0.5838 | 0.1628 | 0.8496 | 20.0% |
| 34 | WCT | Baseline | native / free | 0.3658 | 0.1542 | 0.5774 | 0.0232 | 1.0010 | 20.0% |

> **新 Decoder 定位**：C0 第 **6** / 34，C1 第 **5** / 34。

## 2. 本轮 4 个 Decoder 的 cross600 全指标

| 变体 | Affinity | DINO-S | CLIP-S | DINO-C | LPIPS | CSD Top-1 | CSD Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdvae | 0.5548 | 0.4281 | 0.6814 | 0.7362 | 0.5149 | 0.3450 | -0.0551 |
| old_s10 | 0.5616 | 0.4404 | 0.6828 | 0.7336 | 0.5034 | 0.3767 | -0.0411 |
| c0 | 0.5575 | 0.4323 | 0.6827 | 0.7324 | 0.5022 | 0.3683 | -0.0457 |
| c1 | 0.5576 | 0.4321 | 0.6832 | 0.7324 | 0.5015 | 0.3600 | -0.0479 |

## 3. 各指标在新旧 Ours 方法中的排名（Affinity 降序 / DINO-S 降序 / CLIP-S 降序 / DINO-C 降序 / LPIPS 升序 / CSD Top-1 降序）

- **Style Affinity**：第5名 0.5576；第6名 0.5575
- **DINO-S**：第6名 0.4323；第7名 0.4321
- **CLIP-S**：第4名 0.6832；第5名 0.6827
- **DINO-C**：第18名 0.7324；第19名 0.7324
- **LPIPS(低更好)**：第15名 0.5015；第16名 0.5022
- **CSD Top-1**：第5名 0.3683；第7名 0.3600

## 4. same150（同风格无谓漂移）

| 变体 | Affinity | DINO-S | CLIP-S | DINO-C | LPIPS | CSD Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| sdvae | 0.7026 | 0.6009 | 0.8044 | 0.7890 | 0.3931 | 0.9400 |
| old_s10 | 0.7109 | 0.6124 | 0.8095 | 0.7863 | 0.3857 | 0.9600 |
| c0 | 0.7066 | 0.6052 | 0.8079 | 0.7852 | 0.3867 | 0.9667 |
| c1 | 0.7061 | 0.6045 | 0.8078 | 0.7848 | 0.3855 | 0.9667 |

## 5. 配对 Bootstrap（vs sdvae, cross600, 按源图聚类保留重复簇）

| 对比 | 指标 | 均值差 | 95% CI | 排除 0 |
|---|---|---:|---|---|
| old_s10 | style_affinity | +0.0068 | [+0.0051, +0.0085] | 是 |
| old_s10 | dino_s | +0.0122 | [+0.0096, +0.0150] | 是 |
| old_s10 | clip_s | +0.0014 | [-0.0002, +0.0030] | 否 |
| old_s10 | dino_c | -0.0025 | [-0.0062, +0.0010] | 否 |
| old_s10 | lpips | -0.0114 | [-0.0139, -0.0090] | 是 |
| old_s10 | csd_top1 | +0.0317 | [+0.0067, +0.0583] | 是 |
| old_s10 | csd_margin | +0.0141 | [+0.0093, +0.0187] | 是 |
| c0 | style_affinity | +0.0027 | [+0.0010, +0.0043] | 是 |
| c0 | dino_s | +0.0041 | [+0.0015, +0.0069] | 是 |
| c0 | clip_s | +0.0013 | [-0.0003, +0.0027] | 否 |
| c0 | dino_c | -0.0037 | [-0.0069, -0.0009] | 是 |
| c0 | lpips | -0.0126 | [-0.0140, -0.0112] | 是 |
| c0 | csd_top1 | +0.0233 | [-0.0050, +0.0533] | 否 |
| c0 | csd_margin | +0.0094 | [+0.0038, +0.0151] | 是 |
| c1 | style_affinity | +0.0029 | [+0.0012, +0.0045] | 是 |
| c1 | dino_s | +0.0040 | [+0.0013, +0.0068] | 是 |
| c1 | clip_s | +0.0018 | [+0.0003, +0.0032] | 是 |
| c1 | dino_c | -0.0038 | [-0.0070, -0.0009] | 是 |
| c1 | lpips | -0.0133 | [-0.0147, -0.0119] | 是 |
| c1 | csd_top1 | +0.0150 | [-0.0150, +0.0450] | 否 |
| c1 | csd_margin | +0.0073 | [+0.0014, +0.0133] | 是 |

## 6. 五目标风格各自 Affinity（cross600）

| 变体 | Early_Ren | Impressionism | Minimalism | Rococo | Ukiyo_e |
|---|---:|---:|---:|---:|---:|
| sdvae | 0.5709 | 0.6217 | 0.4823 | 0.5226 | 0.5763 |
| old_s10 | 0.5806 | 0.6266 | 0.4879 | 0.5253 | 0.5875 |
| c0 | 0.5760 | 0.6195 | 0.4864 | 0.5193 | 0.5862 |
| c1 | 0.5757 | 0.6197 | 0.4870 | 0.5196 | 0.5864 |

## 7. 结论

1. **协议可信**：重算 sdvae 与榜单项逐位吻合，新数字可直接比较。
2. **新 Decoder 在 34 方法里的位置**：C0 Style Affinity 0.5575（第 6）、C1 0.5576（第 5），都**略高于榜单第 5 名 Q2a+Oracle+SDVAE 0.5548**，但**低于第 4 名 Q2a+Oracle+S10 0.5587**（旧 S10 本轮重算 0.5616，甚至更高）。
3. **与「目前最好的 Q2A+Oracle」的关系**：榜单里最好的 Ours 是 `Q2a + R1 Oracle + S10`（Affinity 0.5587，全榜第 4）；新 C0/C1 没超过它。三个外部 Baseline（StyleShot 0.6949 / StyleID 0.6469 / AesPA 0.5784）仍在前面。
4. **显著项（Bootstrap）**：见第 5 节——新 Decoder 相对 sdvae 在 Affinity / LPIPS / CSD 上的变化是否显著以 CI 为准，不要只看点估计。
5. **C1 vs C0**：两者 Affinity 几乎相同（0.5576 vs 0.5575），C1 响应蒸馏在下游 Q2A+Oracle1 上没带来可测的风格收益。

## 8. 产物

- 出图：`deliver/q2a_oracle_images.tar.gz`（1.18GB，4 变体 × 750 + 150 源图 + 750 latent + manifest）
- 本轮打分：`deliver/q2a_oracle_scores_baseline_protocol.json`（含 32 方法 baseline 解析结果 + 新 Decoder 全指标 + Bootstrap）
- 逐任务 3000 条：`deliver/q2a_oracle_per_task_baseline_protocol.json`（含每图对 5 风格的 DINO/CLIP 分数向量，方便后续算其他指标）
- 上一版错误口径（生成图 vs 源图余弦）的结果保留在 `deliver/q2a_oracle_scores.json`，**不要用于和 docs/baseline 比**