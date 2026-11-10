# 统一测试集完整排名（600 跨风格任务，paper_rgb_v1 协议）

> **测试集**：`eval_protocol/tasks_750.json` 包含的 600 个跨风格任务（150 源图，每风格 120 任务）。已彻底删除所有 screen50、heldout400、same150、eval200 等简略切片测试集，以后所有模型和 baseline 统一使用本测试集。
> **评分协议**：统一为 `paper_rgb_v1`（CSD、DINOv2、LPIPS 均以原始源图 RGB 为零点，TRT fp16 VAE 解码）。
> **指标范围**：纯跨风格 600 任务（严格排除 150 个 same-style 自重构任务，避免虚高）。

## 一、全模型 Target Top-1 总排名（风格是否真正成立）

| 排名 | 方法 | 类别 | Target Top-1 ↑ | Target Margin ↑ | CSD ΔT ↑ | DINO-C ↑ | LPIPS ↓ | 5类改善 | Ukiyo-e ΔT | 评测来源 |
|---:|---|:---:|---:|---:|---:|---:|---:|:---:|---:|---|
| 1 | StyleShot | Baseline (外部) | 92.6% | +0.2175 | +0.3846 | 0.3569 | 0.7697 | 5/5 | +0.6025 | `docs/baseline (evaluator_v2)` |
| 2 | StyleID | Baseline (外部) | 73.2% | +0.1533 | +0.3134 | 0.3885 | 0.5554 | 5/5 | +0.5527 | `docs/baseline (evaluator_v2)` |
| 3 | StyTr2 | Baseline (外部) | 39.5% | -0.0407 | +0.1257 | 0.7023 | 0.7386 | 4/5 | +0.0623 | `docs/baseline (evaluator_v2)` |
| 4 | Seedream 4.5 | Baseline (外部) | 37.3% | -0.0562 | +0.2075 | 0.7343 | 0.5202 | 5/5 | +0.4510 | `docs/baseline (evaluator_v2)` |
| 5 | **Q2a + R1 Oracle (1 step)** | Our Model (本轮实测) | **37.2%** | -0.0536 | +0.0249 | 0.6966 | 0.5438 | 3/5 | -0.0798 | `runs/v6_next/eval_750_oracle_s1/eval.json` |
| 6 | **V6 ScoreDiff N=24** | Our Model (本轮实测) | **35.5%** | -0.0352 | +0.0666 | 0.5725 | 0.5455 | 3/5 | -0.0118 | `runs/v6_next/eval_750_scorediff_N24/eval.json` |
| 7 | AdaIN | Baseline (外部) | 34.3% | -0.0360 | +0.0126 | 0.2271 | 0.7641 | 2/5 | -0.0335 | `docs/baseline (evaluator_v2)` |
| 8 | AesPA | Baseline (外部) | 31.0% | -0.0473 | +0.1386 | 0.6075 | 0.6247 | 4/5 | +0.0193 | `docs/baseline (evaluator_v2)` |
| 9 | **V6 ScoreDiff N=12 strong** | Our Model (本轮实测) | **30.8%** | -0.0588 | +0.0642 | 0.6242 | 0.4919 | 3/5 | -0.0053 | `runs/v6_next/eval_750_scorediff_N12/eval.json` |
| 10 | **P1_B4_distill50 (218K)** | Our Model (本轮实测) | **24.2%** | -0.1201 | +0.0602 | 0.7960 | 0.4744 | 4/5 | -0.0324 | `runs/v6_next/eval_750_P1_B4_distill50/eval.json` |
| 11 | **P1_B3_distill20 (218K)** | Our Model (本轮实测) | **20.8%** | -0.1373 | +0.0609 | 0.8098 | 0.4620 | 4/5 | -0.0245 | `runs/v6_next/eval_750_P1_B3_distill20/eval.json` |
| 12 | WCT | Baseline (外部) | 20.0% | -0.1003 | -0.0926 | 0.0232 | 1.0010 | 1/5 | -0.1330 | `docs/baseline (evaluator_v2)` |
| 13 | SaMST | Baseline (外部) | 20.0% | -0.1288 | -0.0615 | 0.1628 | 0.8496 | 1/5 | -0.0554 | `docs/baseline (evaluator_v2)` |
| 14 | WEAVE (Published) | Baseline (外部) | 16.7% | -0.2557 | +0.0895 | 0.8079 | 0.4974 | 5/5 | +0.0895 | `docs/baseline (evaluator_v2)` |
| 15 | **P1_B0_ctrl218 (218K)** | Our Model (本轮实测) | **7.5%** | -0.2635 | +0.0338 | 0.8702 | 0.4051 | 4/5 | -0.0114 | `runs/v6_next/eval_750_P1_B0_ctrl218/eval.json` |
| 16 | **Q2a Direct (218K)** | Our Model (本轮实测) | **7.0%** | -0.2479 | +0.0429 | 0.8703 | 0.4001 | 3/5 | -0.0056 | `runs/v6_next/eval_750_Q2a_embed_style/eval.json` |
| 17 | CUT | Baseline (外部) | 5.7% | -0.2385 | +0.0477 | 0.8435 | 0.6193 | 3/5 | +0.0712 | `docs/baseline (evaluator_v2)` |
| 18 | V6 ScoreDiff (lr0.055, s42) | Baseline (外部) | 5.0% | -0.3345 | +0.0554 | 0.9024 | 0.2370 | 5/5 | +0.0327 | `docs/baseline (evaluator_v2)` |
| 19 | V6 ScoreDiff (lr0.055, s43) | Baseline (外部) | 5.0% | -0.3340 | +0.0554 | 0.9020 | 0.2369 | 5/5 | +0.0324 | `docs/baseline (evaluator_v2)` |
| 20 | V6 ScoreDiff (lr0.048, s42) | Baseline (外部) | 4.0% | -0.3575 | +0.0513 | 0.9188 | 0.2079 | 5/5 | +0.0339 | `docs/baseline (evaluator_v2)` |
| 21 | V6 ScoreDiff (lr0.048, s43) | Baseline (外部) | 3.7% | -0.3576 | +0.0512 | 0.9185 | 0.2079 | 5/5 | +0.0334 | `docs/baseline (evaluator_v2)` |
| 22 | **L11 Direct (218K)** | Our Model (本轮实测) | **3.5%** | -0.2953 | +0.0264 | 0.8909 | 0.3619 | 3/5 | -0.0175 | `runs/v6_next/eval_750_L11_k1024_dim16/eval.json` |
| 23 | V6 D1 Gentle (s42) | Baseline (外部) | 3.2% | -0.3030 | +0.0504 | 0.8994 | 0.3639 | 5/5 | +0.0254 | `docs/baseline (evaluator_v2)` |
| 24 | V6 D1 Gentle (s43) | Baseline (外部) | 3.0% | -0.3011 | +0.0524 | 0.8996 | 0.3634 | 5/5 | +0.0301 | `docs/baseline (evaluator_v2)` |
| 25 | SaMAM | Baseline (外部) | 2.5% | -0.2886 | +0.1019 | 0.8239 | 0.3882 | 5/5 | +0.0809 | `docs/baseline (evaluator_v2)` |
| 26 | V6 D1 Calibrated (a0.70, s42) | Baseline (外部) | 2.0% | -0.3721 | +0.0428 | 0.9345 | 0.2508 | 5/5 | +0.0338 | `docs/baseline (evaluator_v2)` |
| 27 | V6 D1 Calibrated (a0.70, s43) | Baseline (外部) | 2.0% | -0.3722 | +0.0434 | 0.9342 | 0.2507 | 5/5 | +0.0372 | `docs/baseline (evaluator_v2)` |
| 28 | Identity | Baseline (外部) | 0.5% | -0.4543 | -0.0000 | 1.0000 | 0.0000 | 0/5 | -0.0000 | `docs/baseline (evaluator_v2)` |
| 29 | SD-Turbo | Baseline (外部) | 0.5% | -0.4548 | -0.0035 | 0.9998 | 0.0048 | 0/5 | -0.0029 | `docs/baseline (evaluator_v2)` |


## 二、保真档排名（DINO-C ≥ 0.80）

> 排除几乎放弃内容保真的极端风格化模型，保留内容结构较为完整的模型。

| 排名 | 方法 | 类别 | Target Top-1 ↑ | Target Margin ↑ | CSD ΔT ↑ | DINO-C ↑ | LPIPS ↓ |
|---:|---|:---:|---:|---:|---:|---:|---:|
| 1 | **P1_B3_distill20 (218K)** | Our Model (本轮实测) | **20.8%** | -0.1373 | +0.0609 | 0.8098 | 0.4620 |
| 2 | WEAVE (Published) | Baseline (外部) | 16.7% | -0.2557 | +0.0895 | 0.8079 | 0.4974 |
| 3 | **P1_B0_ctrl218 (218K)** | Our Model (本轮实测) | **7.5%** | -0.2635 | +0.0338 | 0.8702 | 0.4051 |
| 4 | **Q2a Direct (218K)** | Our Model (本轮实测) | **7.0%** | -0.2479 | +0.0429 | 0.8703 | 0.4001 |
| 5 | CUT | Baseline (外部) | 5.7% | -0.2385 | +0.0477 | 0.8435 | 0.6193 |
| 6 | V6 ScoreDiff (lr0.055, s42) | Baseline (外部) | 5.0% | -0.3345 | +0.0554 | 0.9024 | 0.2370 |
| 7 | V6 ScoreDiff (lr0.055, s43) | Baseline (外部) | 5.0% | -0.3340 | +0.0554 | 0.9020 | 0.2369 |
| 8 | V6 ScoreDiff (lr0.048, s42) | Baseline (外部) | 4.0% | -0.3575 | +0.0513 | 0.9188 | 0.2079 |
| 9 | V6 ScoreDiff (lr0.048, s43) | Baseline (外部) | 3.7% | -0.3576 | +0.0512 | 0.9185 | 0.2079 |
| 10 | **L11 Direct (218K)** | Our Model (本轮实测) | **3.5%** | -0.2953 | +0.0264 | 0.8909 | 0.3619 |
| 11 | V6 D1 Gentle (s42) | Baseline (外部) | 3.2% | -0.3030 | +0.0504 | 0.8994 | 0.3639 |
| 12 | V6 D1 Gentle (s43) | Baseline (外部) | 3.0% | -0.3011 | +0.0524 | 0.8996 | 0.3634 |
| 13 | SaMAM | Baseline (外部) | 2.5% | -0.2886 | +0.1019 | 0.8239 | 0.3882 |
| 14 | V6 D1 Calibrated (a0.70, s42) | Baseline (外部) | 2.0% | -0.3721 | +0.0428 | 0.9345 | 0.2508 |
| 15 | V6 D1 Calibrated (a0.70, s43) | Baseline (外部) | 2.0% | -0.3722 | +0.0434 | 0.9342 | 0.2507 |
| 16 | Identity | Baseline (外部) | 0.5% | -0.4543 | -0.0000 | 1.0000 | 0.0000 |
| 17 | SD-Turbo | Baseline (外部) | 0.5% | -0.4548 | -0.0035 | 0.9998 | 0.0048 |


## 三、高保真档排名（DINO-C ≥ 0.85）

> 严格保真档：要求几乎无损原图内容与结构。

| 排名 | 方法 | 类别 | Target Top-1 ↑ | Target Margin ↑ | CSD ΔT ↑ | DINO-C ↑ | LPIPS ↓ |
|---:|---|:---:|---:|---:|---:|---:|---:|
| 1 | **P1_B0_ctrl218 (218K)** | Our Model (本轮实测) | **7.5%** | -0.2635 | +0.0338 | 0.8702 | 0.4051 |
| 2 | **Q2a Direct (218K)** | Our Model (本轮实测) | **7.0%** | -0.2479 | +0.0429 | 0.8703 | 0.4001 |
| 3 | V6 ScoreDiff (lr0.055, s42) | Baseline (外部) | 5.0% | -0.3345 | +0.0554 | 0.9024 | 0.2370 |
| 4 | V6 ScoreDiff (lr0.055, s43) | Baseline (外部) | 5.0% | -0.3340 | +0.0554 | 0.9020 | 0.2369 |
| 5 | V6 ScoreDiff (lr0.048, s42) | Baseline (外部) | 4.0% | -0.3575 | +0.0513 | 0.9188 | 0.2079 |
| 6 | V6 ScoreDiff (lr0.048, s43) | Baseline (外部) | 3.7% | -0.3576 | +0.0512 | 0.9185 | 0.2079 |
| 7 | **L11 Direct (218K)** | Our Model (本轮实测) | **3.5%** | -0.2953 | +0.0264 | 0.8909 | 0.3619 |
| 8 | V6 D1 Gentle (s42) | Baseline (外部) | 3.2% | -0.3030 | +0.0504 | 0.8994 | 0.3639 |
| 9 | V6 D1 Gentle (s43) | Baseline (外部) | 3.0% | -0.3011 | +0.0524 | 0.8996 | 0.3634 |
| 10 | V6 D1 Calibrated (a0.70, s42) | Baseline (外部) | 2.0% | -0.3721 | +0.0428 | 0.9345 | 0.2508 |
| 11 | V6 D1 Calibrated (a0.70, s43) | Baseline (外部) | 2.0% | -0.3722 | +0.0434 | 0.9342 | 0.2507 |
| 12 | Identity | Baseline (外部) | 0.5% | -0.4543 | -0.0000 | 1.0000 | 0.0000 |
| 13 | SD-Turbo | Baseline (外部) | 0.5% | -0.4548 | -0.0035 | 0.9998 | 0.0048 |