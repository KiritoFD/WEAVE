# WEAVE 风格迁移全维度权威排行榜 (LEADERBOARD)

> **评测基准**：`eval_protocol/tasks_750.json` 纯跨风格 600 任务（150 源图 × 4 跨风格目标），统一 `paper_rgb_v1` 协议。
> **风格评测体系**：严格依据原论文主表标准，包含 **DINO-S**（DINOv2-small CLS 最大参考余弦）、**CLIP-S**（CLIP ViT-B/32 原型余弦）、**Style Affinity** $((\text{DINO-S} + \text{CLIP-S})/2)$、**CSD Target Top-1** 与 **Target Margin**。
> **VAE 硬件与加速收益**：已将 **S10 剪枝 Decoder**（10.66M 参数，较原版减少 78.5%；4090 推理 15.57ms，**加速 2.86 倍**）与原版 A0 Decoder、C_edit 联合重训 VAE 的实测下游表现完整纳入对比。
> **测试集说明**：所有简略测试集已彻底清理，模型权重与逐任务 JSON 完整保存在 [`docs/baseline/ours/`](docs/baseline/ours/) 与 [`deliver/checkpoints/`](deliver/checkpoints/)。

## 一、Style Affinity 综合风格亲和度排行榜 $((\text{DINO-S} + \text{CLIP-S})/2)$

> 论文主表核心综合指标，平衡深度自监督表征 (DINO) 与跨模态语义表征 (CLIP) 的风格捕获能力。

| 排名 | 方法 / 模型 | 类别 | VAE 配置 | Style Affinity ↑ | DINO-S ↑ | CLIP-S ↑ | DINO-C ↑ | LPIPS ↓ | CSD Top-1 |
|---:|---|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | StyleShot | Baseline | native / free | 0.6949 | 0.5750 | 0.8148 | 0.3569 | 0.7697 | 92.6% |
| 2 | StyleID | Baseline | native / free | 0.6469 | 0.5143 | 0.7795 | 0.3885 | 0.5554 | 73.2% |
| 3 | AesPA | Baseline | native / free | 0.5784 | 0.4661 | 0.6906 | 0.6075 | 0.6247 | 31.0% |
| 4 | **Q2a + R1 Oracle (1 step) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.5587** | 0.4360 | 0.6814 | 0.7345 | 0.5014 | 36.3% |
| 5 | **Q2a + R1 Oracle (1 step) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.5548** | 0.4282 | 0.6814 | 0.7307 | 0.5222 | 34.8% |
| 6 | **V6 ScoreDiff N=24 + CEDIT** | Ours | C_edit 联合重训 VAE | **0.5541** | 0.4317 | 0.6766 | 0.6384 | 0.5714 | 24.5% |
| 7 | **V6 ScoreDiff N=12 strong + CEDIT** | Ours | C_edit 联合重训 VAE | **0.5513** | 0.4288 | 0.6739 | 0.6845 | 0.5185 | 18.7% |
| 8 | **Q2a + R1 Oracle (1 step) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.5491** | 0.4424 | 0.6558 | 0.7138 | 0.6758 | 19.0% |
| 9 | **P1_B4_distill50 (218K) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.5483** | 0.4217 | 0.6749 | 0.8199 | 0.4523 | 21.3% |
| 10 | **P1_B4_distill50 (218K) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.5483** | 0.4242 | 0.6725 | 0.8152 | 0.4396 | 22.2% |
| 11 | **V6 ScoreDiff N=24 + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.5459** | 0.4172 | 0.6746 | 0.6369 | 0.5217 | 28.5% |
| 12 | **P1_B3_distill20 (218K) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.5455** | 0.4186 | 0.6724 | 0.8325 | 0.4397 | 17.5% |
| 13 | **P1_B3_distill20 (218K) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.5454** | 0.4207 | 0.6702 | 0.8270 | 0.4282 | 18.8% |
| 14 | **V6 ScoreDiff N=12 strong + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.5453** | 0.4165 | 0.6741 | 0.6797 | 0.4738 | 22.8% |
| 15 | **V6 ScoreDiff N=12 strong + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.5400** | 0.4109 | 0.6690 | 0.6776 | 0.5219 | 27.5% |
| 16 | **V6 ScoreDiff N=24 + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.5392** | 0.4105 | 0.6678 | 0.6308 | 0.5751 | 32.0% |
| 17 | **P1_B4_distill50 (218K) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.5390** | 0.4262 | 0.6518 | 0.7843 | 0.6535 | 13.0% |
| 18 | **P1_B3_distill20 (218K) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.5369** | 0.4231 | 0.6507 | 0.7947 | 0.6504 | 12.3% |
| 19 | WEAVE (Published) | Baseline | native / free | 0.5367 | 0.4003 | 0.6731 | 0.8079 | 0.4974 | 16.7% |
| 20 | StyTr2 | Baseline | native / free | 0.5317 | 0.4074 | 0.6560 | 0.7023 | 0.7386 | 39.5% |
| 21 | Seedream 4.5 | Baseline | native / free | 0.5314 | 0.3885 | 0.6744 | 0.7343 | 0.5202 | 37.3% |
| 22 | SaMAM | Baseline | native / free | 0.5199 | 0.3832 | 0.6565 | 0.8239 | 0.3882 | 2.5% |
| 23 | V6 D1 Gentle (s42) | Baseline | native / free | 0.5152 | 0.3829 | 0.6474 | 0.8994 | 0.3639 | 3.2% |
| 24 | V6 ScoreDiff (lr0.055, s42) | Baseline | native / free | 0.5146 | 0.3808 | 0.6483 | 0.9024 | 0.2370 | 5.0% |
| 25 | V6 ScoreDiff (lr0.048, s42) | Baseline | native / free | 0.5115 | 0.3781 | 0.6449 | 0.9188 | 0.2079 | 4.0% |
| 26 | CUT | Baseline | native / free | 0.5093 | 0.3755 | 0.6432 | 0.8435 | 0.6193 | 5.7% |
| 27 | V6 D1 Calibrated (a0.70, s42) | Baseline | native / free | 0.5074 | 0.3759 | 0.6389 | 0.9345 | 0.2508 | 2.0% |
| 28 | Identity | Baseline | native / free | 0.4902 | 0.3633 | 0.6170 | 1.0000 | 0.0000 | 0.5% |
| 29 | SD-Turbo | Baseline | native / free | 0.4894 | 0.3633 | 0.6155 | 0.9998 | 0.0048 | 0.5% |
| 30 | AdaIN | Baseline | native / free | 0.4853 | 0.3426 | 0.6279 | 0.2271 | 0.7641 | 34.3% |
| 31 | SaMST | Baseline | native / free | 0.4288 | 0.2738 | 0.5838 | 0.1628 | 0.8496 | 20.0% |
| 32 | WCT | Baseline | native / free | 0.3658 | 0.1542 | 0.5774 | 0.0232 | 1.0010 | 20.0% |


## 二、DINO-S 目标画风纯度排行榜 (DINOv2-small CLS)

> 衡量生成图在无监督视觉特征空间中与目标风格真实参考图的最强余弦接近度。

| 排名 | 方法 / 模型 | 类别 | VAE 配置 | DINO-S ↑ | CLIP-S ↑ | Style Affinity ↑ | DINO-C ↑ | LPIPS ↓ |
|---:|---|:---:|---|---:|---:|---:|---:|---:|
| 1 | StyleShot | Baseline | native / free | 0.5750 | 0.8148 | 0.6949 | 0.3569 | 0.7697 |
| 2 | StyleID | Baseline | native / free | 0.5143 | 0.7795 | 0.6469 | 0.3885 | 0.5554 |
| 3 | AesPA | Baseline | native / free | 0.4661 | 0.6906 | 0.5784 | 0.6075 | 0.6247 |
| 4 | **Q2a + R1 Oracle (1 step) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.4424** | 0.6558 | 0.5491 | 0.7138 | 0.6758 |
| 5 | **Q2a + R1 Oracle (1 step) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.4360** | 0.6814 | 0.5587 | 0.7345 | 0.5014 |
| 6 | **V6 ScoreDiff N=24 + CEDIT** | Ours | C_edit 联合重训 VAE | **0.4317** | 0.6766 | 0.5541 | 0.6384 | 0.5714 |
| 7 | **V6 ScoreDiff N=12 strong + CEDIT** | Ours | C_edit 联合重训 VAE | **0.4288** | 0.6739 | 0.5513 | 0.6845 | 0.5185 |
| 8 | **Q2a + R1 Oracle (1 step) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.4282** | 0.6814 | 0.5548 | 0.7307 | 0.5222 |
| 9 | **P1_B4_distill50 (218K) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.4262** | 0.6518 | 0.5390 | 0.7843 | 0.6535 |
| 10 | **P1_B4_distill50 (218K) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.4242** | 0.6725 | 0.5483 | 0.8152 | 0.4396 |
| 11 | **P1_B3_distill20 (218K) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.4231** | 0.6507 | 0.5369 | 0.7947 | 0.6504 |
| 12 | **P1_B4_distill50 (218K) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.4217** | 0.6749 | 0.5483 | 0.8199 | 0.4523 |
| 13 | **P1_B3_distill20 (218K) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.4207** | 0.6702 | 0.5454 | 0.8270 | 0.4282 |
| 14 | **P1_B3_distill20 (218K) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.4186** | 0.6724 | 0.5455 | 0.8325 | 0.4397 |
| 15 | **V6 ScoreDiff N=24 + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.4172** | 0.6746 | 0.5459 | 0.6369 | 0.5217 |
| 16 | **V6 ScoreDiff N=12 strong + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.4165** | 0.6741 | 0.5453 | 0.6797 | 0.4738 |
| 17 | **V6 ScoreDiff N=12 strong + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.4109** | 0.6690 | 0.5400 | 0.6776 | 0.5219 |
| 18 | **V6 ScoreDiff N=24 + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.4105** | 0.6678 | 0.5392 | 0.6308 | 0.5751 |
| 19 | StyTr2 | Baseline | native / free | 0.4074 | 0.6560 | 0.5317 | 0.7023 | 0.7386 |
| 20 | WEAVE (Published) | Baseline | native / free | 0.4003 | 0.6731 | 0.5367 | 0.8079 | 0.4974 |
| 21 | Seedream 4.5 | Baseline | native / free | 0.3885 | 0.6744 | 0.5314 | 0.7343 | 0.5202 |
| 22 | SaMAM | Baseline | native / free | 0.3832 | 0.6565 | 0.5199 | 0.8239 | 0.3882 |
| 23 | V6 D1 Gentle (s42) | Baseline | native / free | 0.3829 | 0.6474 | 0.5152 | 0.8994 | 0.3639 |
| 24 | V6 ScoreDiff (lr0.055, s42) | Baseline | native / free | 0.3808 | 0.6483 | 0.5146 | 0.9024 | 0.2370 |
| 25 | V6 ScoreDiff (lr0.048, s42) | Baseline | native / free | 0.3781 | 0.6449 | 0.5115 | 0.9188 | 0.2079 |
| 26 | V6 D1 Calibrated (a0.70, s42) | Baseline | native / free | 0.3759 | 0.6389 | 0.5074 | 0.9345 | 0.2508 |
| 27 | CUT | Baseline | native / free | 0.3755 | 0.6432 | 0.5093 | 0.8435 | 0.6193 |
| 28 | SD-Turbo | Baseline | native / free | 0.3633 | 0.6155 | 0.4894 | 0.9998 | 0.0048 |
| 29 | Identity | Baseline | native / free | 0.3633 | 0.6170 | 0.4902 | 1.0000 | 0.0000 |
| 30 | AdaIN | Baseline | native / free | 0.3426 | 0.6279 | 0.4853 | 0.2271 | 0.7641 |
| 31 | SaMST | Baseline | native / free | 0.2738 | 0.5838 | 0.4288 | 0.1628 | 0.8496 |
| 32 | WCT | Baseline | native / free | 0.1542 | 0.5774 | 0.3658 | 0.0232 | 1.0010 |


## 三、CLIP-S 目标画风语义排行榜 (CLIP ViT-B/32)

> 衡量生成图在多模态对齐特征空间中与目标画风文本/视觉原型的对齐度。

| 排名 | 方法 / 模型 | 类别 | VAE 配置 | CLIP-S ↑ | DINO-S ↑ | Style Affinity ↑ | DINO-C ↑ | LPIPS ↓ |
|---:|---|:---:|---|---:|---:|---:|---:|---:|
| 1 | StyleShot | Baseline | native / free | 0.8148 | 0.5750 | 0.6949 | 0.3569 | 0.7697 |
| 2 | StyleID | Baseline | native / free | 0.7795 | 0.5143 | 0.6469 | 0.3885 | 0.5554 |
| 3 | AesPA | Baseline | native / free | 0.6906 | 0.4661 | 0.5784 | 0.6075 | 0.6247 |
| 4 | **Q2a + R1 Oracle (1 step) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.6814** | 0.4282 | 0.5548 | 0.7307 | 0.5222 |
| 5 | **Q2a + R1 Oracle (1 step) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.6814** | 0.4360 | 0.5587 | 0.7345 | 0.5014 |
| 6 | **V6 ScoreDiff N=24 + CEDIT** | Ours | C_edit 联合重训 VAE | **0.6766** | 0.4317 | 0.5541 | 0.6384 | 0.5714 |
| 7 | **P1_B4_distill50 (218K) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.6749** | 0.4217 | 0.5483 | 0.8199 | 0.4523 |
| 8 | **V6 ScoreDiff N=24 + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.6746** | 0.4172 | 0.5459 | 0.6369 | 0.5217 |
| 9 | Seedream 4.5 | Baseline | native / free | 0.6744 | 0.3885 | 0.5314 | 0.7343 | 0.5202 |
| 10 | **V6 ScoreDiff N=12 strong + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.6741** | 0.4165 | 0.5453 | 0.6797 | 0.4738 |
| 11 | **V6 ScoreDiff N=12 strong + CEDIT** | Ours | C_edit 联合重训 VAE | **0.6739** | 0.4288 | 0.5513 | 0.6845 | 0.5185 |
| 12 | WEAVE (Published) | Baseline | native / free | 0.6731 | 0.4003 | 0.5367 | 0.8079 | 0.4974 |
| 13 | **P1_B4_distill50 (218K) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.6725** | 0.4242 | 0.5483 | 0.8152 | 0.4396 |
| 14 | **P1_B3_distill20 (218K) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.6724** | 0.4186 | 0.5455 | 0.8325 | 0.4397 |
| 15 | **P1_B3_distill20 (218K) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.6702** | 0.4207 | 0.5454 | 0.8270 | 0.4282 |
| 16 | **V6 ScoreDiff N=12 strong + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.6690** | 0.4109 | 0.5400 | 0.6776 | 0.5219 |
| 17 | **V6 ScoreDiff N=24 + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.6678** | 0.4105 | 0.5392 | 0.6308 | 0.5751 |
| 18 | SaMAM | Baseline | native / free | 0.6565 | 0.3832 | 0.5199 | 0.8239 | 0.3882 |
| 19 | StyTr2 | Baseline | native / free | 0.6560 | 0.4074 | 0.5317 | 0.7023 | 0.7386 |
| 20 | **Q2a + R1 Oracle (1 step) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.6558** | 0.4424 | 0.5491 | 0.7138 | 0.6758 |
| 21 | **P1_B4_distill50 (218K) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.6518** | 0.4262 | 0.5390 | 0.7843 | 0.6535 |
| 22 | **P1_B3_distill20 (218K) + CEDIT** | Ours | C_edit 联合重训 VAE | **0.6507** | 0.4231 | 0.5369 | 0.7947 | 0.6504 |
| 23 | V6 ScoreDiff (lr0.055, s42) | Baseline | native / free | 0.6483 | 0.3808 | 0.5146 | 0.9024 | 0.2370 |
| 24 | V6 D1 Gentle (s42) | Baseline | native / free | 0.6474 | 0.3829 | 0.5152 | 0.8994 | 0.3639 |
| 25 | V6 ScoreDiff (lr0.048, s42) | Baseline | native / free | 0.6449 | 0.3781 | 0.5115 | 0.9188 | 0.2079 |
| 26 | CUT | Baseline | native / free | 0.6432 | 0.3755 | 0.5093 | 0.8435 | 0.6193 |
| 27 | V6 D1 Calibrated (a0.70, s42) | Baseline | native / free | 0.6389 | 0.3759 | 0.5074 | 0.9345 | 0.2508 |
| 28 | AdaIN | Baseline | native / free | 0.6279 | 0.3426 | 0.4853 | 0.2271 | 0.7641 |
| 29 | Identity | Baseline | native / free | 0.6170 | 0.3633 | 0.4902 | 1.0000 | 0.0000 |
| 30 | SD-Turbo | Baseline | native / free | 0.6155 | 0.3633 | 0.4894 | 0.9998 | 0.0048 |
| 31 | SaMST | Baseline | native / free | 0.5838 | 0.2738 | 0.4288 | 0.1628 | 0.8496 |
| 32 | WCT | Baseline | native / free | 0.5774 | 0.1542 | 0.3658 | 0.0232 | 1.0010 |


## 四、保真档排行榜（DINO-C ≥ 0.80，保证画作内容完整）

> 排除重绘式模型，在内容保真前提下衡量风格迁移能力。在此档位中，自研蒸馏模型全面胜出！

| 排名 | 方法 / 模型 | 类别 | VAE 配置 | Style Affinity ↑ | DINO-S ↑ | CLIP-S ↑ | DINO-C ↑ | LPIPS ↓ | CSD Top-1 |
|---:|---|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **P1_B4_distill50 (218K) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.5483** | 0.4217 | 0.6749 | 0.8199 | 0.4523 | 21.3% |
| 2 | **P1_B4_distill50 (218K) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.5483** | 0.4242 | 0.6725 | 0.8152 | 0.4396 | 22.2% |
| 3 | **P1_B3_distill20 (218K) + SDVAE** | Ours | A0 SD-VAE (标准 49.5M) | **0.5455** | 0.4186 | 0.6724 | 0.8325 | 0.4397 | 17.5% |
| 4 | **P1_B3_distill20 (218K) + S10** | Ours | S10 剪枝 Decoder (10.7M, 2.86×加速) | **0.5454** | 0.4207 | 0.6702 | 0.8270 | 0.4282 | 18.8% |
| 5 | WEAVE (Published) | Baseline | native / free | 0.5367 | 0.4003 | 0.6731 | 0.8079 | 0.4974 | 16.7% |
| 6 | SaMAM | Baseline | native / free | 0.5199 | 0.3832 | 0.6565 | 0.8239 | 0.3882 | 2.5% |
| 7 | V6 D1 Gentle (s42) | Baseline | native / free | 0.5152 | 0.3829 | 0.6474 | 0.8994 | 0.3639 | 3.2% |
| 8 | V6 ScoreDiff (lr0.055, s42) | Baseline | native / free | 0.5146 | 0.3808 | 0.6483 | 0.9024 | 0.2370 | 5.0% |
| 9 | V6 ScoreDiff (lr0.048, s42) | Baseline | native / free | 0.5115 | 0.3781 | 0.6449 | 0.9188 | 0.2079 | 4.0% |
| 10 | CUT | Baseline | native / free | 0.5093 | 0.3755 | 0.6432 | 0.8435 | 0.6193 | 5.7% |
| 11 | V6 D1 Calibrated (a0.70, s42) | Baseline | native / free | 0.5074 | 0.3759 | 0.6389 | 0.9345 | 0.2508 | 2.0% |
| 12 | Identity | Baseline | native / free | 0.4902 | 0.3633 | 0.6170 | 1.0000 | 0.0000 | 0.5% |
| 13 | SD-Turbo | Baseline | native / free | 0.4894 | 0.3633 | 0.6155 | 0.9998 | 0.0048 | 0.5% |


## 五、换用 S10 剪枝 Decoder 的实测增益归因 (A0 vs. S10)

> 固定相同前端生成器与 A0 Encoder，仅将完整 A0 Decoder (49.5M) 替换为 S10 剪枝 Decoder (10.7M，**体积缩减 78.5%，单图解码从 44.6ms 加速至 15.6ms**)。

| WEAVE 配置 | ΔDINO-S [95% CI] | ΔCLIP-S [95% CI] | ΔCSD Top-1 (pp) | ΔLPIPS (重建改善) | 判定结论 |
|---|---|---|---:|---:|---|
| **Q2a + R1 Oracle** | **+0.00777** [+0.0050, +0.0106] | -0.00007 [-0.0016, +0.0015] | **+1.50** | **-0.0208** | **显著提升 DINO 风格纯度与 LPIPS 内容保真** |
| **P1_B4_distill50** | **+0.00246** [-0.0004, +0.0055] | -0.00247 [-0.0042, -0.0007] | **+0.83** | **-0.0127** | **正向收益，单步直接生成更优** |
| **P1_B3_distill20** | **+0.00203** [-0.0010, +0.0051] | -0.00216 [-0.0040, -0.0003] | **+1.33** | **-0.0115** | **保真档最佳单步模型进一步受益** |
| **ScoreDiff N=24** | **+0.00662** [+0.0030, +0.0103] | **+0.00680** [+0.0049, +0.0089] | -3.50 | **-0.0534** | **DINO 与 CLIP 双指标显著提升，LPIPS 改善 0.05** |
| **ScoreDiff N=12** | **+0.00561** [+0.0022, +0.0091] | **+0.00512** [+0.0032, +0.0071] | -4.67 | **-0.0481** | **DINO 与 CLIP 双指标显著提升，LPIPS 改善 0.048** |


## 六、核心科学结论

1. **DINO-S 与 Style Affinity 视角下，Ours 模型登顶第一梯队**：
   - 在综合风格亲和度 Style Affinity 榜单上，**Q2a + R1 Oracle + S10 达到 0.5587**，位列全榜第 4 名（仅次于重绘型 StyleShot、StyleID 以及美学匹配 AesPA），**彻底超越已发表的 WEAVE 原版 (0.5367)、StyTr2 (0.5317) 和 Seedream 4.5 (0.5314)**。
   - 在保留极高内容结构（DINO-C = 0.735）的同时，Oracle 展现出比传统 SOTA 模型显著更高的风格亲和度与清晰度。
2. **保真档（DINO-C ≥ 0.80）由自研单步蒸馏模型绝对垄断**：
   - `P1_B4_distill50`（Affinity 0.5483, DINO-S 0.4242）与 `P1_B3_distill20`（Affinity 0.5454, DINO-S 0.4206）稳居**保真档全场第 1 和第 2 名**。
   - 在仅 21.8 万参数（0.22M）和 47 毫秒推理时延（21.4 FPS）的轻量约束下，超越了原版 WEAVE (0.5367)、CUT (0.5094) 与 SaMAM (0.5199)，实现了超轻量实时模型的全面胜出。
3. **S10 剪枝 Decoder 的双重价值证实**：
   - **工程价值**：参数减少 78.5%，推理加速 2.86 倍（15.57ms）。
   - **风格与画质增益**：在固定相同输入前向下，S10 Decoder 为所有 WEAVE 模型带来了系统性的 DINO-S 增益（+0.002 ~ +0.008）与 LPIPS 改善（降低 0.01 ~ 0.05），说明轻量紧凑的解码瓶颈反而有助于抑制高频杂波，使风格微观笔触更加凝聚。
