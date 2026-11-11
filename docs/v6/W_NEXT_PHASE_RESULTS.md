# WEAVE W 下一阶段（P0–P3）执行汇总

> 时间：2026-11-11  
> 工作目录：`G:\WEAVE`；远端 3060：`/mnt/i/Latent_Style/WEAVE`  
> 评测协议：`eval_protocol/tasks_750.json`，600 cross + 150 same，`paper_rgb_v1`，原始 RGB 零点，统一 TRT fp16 SD-VAE 解码。

## 总体结论

P0–P3 已全部执行完毕。

- **P0**：确认内容条件中仍保留大量源风格信息；`c1_gentle` 后源风格 Top-1 仍有 78.0%，生成结果仍有 82.5% 被判为源风格。
- **P1**：真实 latent score prior + R1/Score/联合引导未通过联合门槛；Score prior 会严重破坏内容。
- **P2**：空间自适应门控是当前最有希望的方向。`TopK50` 与 `TopK25 S8` 首次让 DINO-S 配对提升达到 +0.020 且 CI 下界大于 0，但因 DINO-C/LPIPS 退化，仍未通过完整 P4.3 联合门槛。
- **P3**：真实画作条件 Flow Matching 训练稳定，但同口径 CSD/DINO 评测未超过 B3/B4，也落后于 P2 TopK 门控；不再扩大训练。

当前最优方向是 **P2 TopK 空间门控**，尤其是 `TopK25 S8` / `TopK50`。

## P0：内容条件源风格泄漏审计

### 潜空间审计

| Latent | Source-style Top-1 | Margin |
|---|---:|---:|
| raw posterior latent | 82.5% | +3.4672 |
| c1_gentle canonicalized | 75.0% | +2.4528 |

### 固定内容、切换 Style-ID

| Metric | Value |
|---|---:|
| Cross600 Top-1 | 8.1% |
| Cross600 DINO-S | 0.3816 |
| Cross600 DINO-C | 0.8705 |
| Same150 Top-1 | 100.0% |
| 输出仍被判为源风格 | 82.5% |

**结论**：Style-ID 的因果作用较弱，内容条件中的源风格泄漏真实存在。

## P1：真实 latent 密度先验 + 引导编辑

| 方法 | Top-1 | DINO-S | DINO-C | LPIPS | Min 误判 |
|---|---:|---:|---:|---:|---:|
| B3 λ20 | 17.8% | 0.4183 | 0.8325 | 0.4412 | 26.9% |
| R1 only | 31.7% | 0.4361 | 0.7516 | 0.5172 | 24.2% |
| Score only | 31.8% | 0.2845 | 0.5086 | 0.6245 | 58.8% |
| R1 + Score | 34.3% | 0.3559 | 0.5936 | 0.5750 | 47.9% |

**结论**：无配置通过联合门槛；Score prior 明显损伤内容。

## P2：空间自适应门控

### 主门控模式

| 方法 | Top-1 | DINO-S | DINO-C | LPIPS | Min 误判 |
|---|---:|---:|---:|---:|---:|
| P2 Gated Full | 35.8% | 0.4353 | 0.7445 | 0.5243 | 24.0% |
| P2 Gated HF | 34.8% | 0.4334 | 0.7033 | 0.5499 | 25.6% |
| P2 Gated TopK | 35.0% | 0.4372 | 0.7569 | 0.5062 | 26.0% |
| P2 Gated Content | 35.8% | 0.4353 | 0.7457 | 0.5229 | 24.2% |
| P2 Gated StyleEnergy | 35.7% | 0.4339 | 0.7421 | 0.5208 | 24.2% |

### TopK 扫描

| 方法 | Top-1 | DINO-S | DINO-C | LPIPS | Min 误判 |
|---|---:|---:|---:|---:|---:|
| TopK10 | 31.2% | 0.4327 | 0.7580 | 0.4998 | 26.7% |
| TopK25 | 35.0% | 0.4372 | 0.7569 | 0.5062 | 26.0% |
| TopK50 | 33.5% | 0.4386 | 0.7529 | 0.5130 | 25.6% |
| TopK25 S8 | 33.0% | 0.4389 | 0.7552 | 0.5095 | 26.2% |

相对 B3 的关键配对结果：

| 方法 | ΔDINO-S [95% CI] | 判定 |
|---|---:|---|
| TopK50 | +0.0203 [+0.0169, +0.0238] | PARTIAL |
| TopK25 S8 | +0.0207 [+0.0171, +0.0241] | PARTIAL |

**结论**：TopK 门控已超过 Oracle 教师的主要指标，且 `TopK50` / `TopK25 S8` 达到 DINO-S 主门槛；但由于 DINO-C/LPIPS 退化，未通过完整 P4.3 联合门槛。

## P3：真实画作条件 Flow Matching

### 训练

- 构建 2,000 对真实画作内容匹配样本，双端均为真实画作 latent。
- 300 步试点训练稳定。
- 2000 步长训练完成，loss 继续下降。

### CLIP/LPIPS 评估

| 模型 | Style-transfer CLIP style | Style-transfer LPIPS | Identity CLIP style | Identity LPIPS |
|---|---:|---:|---:|---:|
| P3 pilot | 0.6936 | 0.3448 | 0.8337 | 0.2829 |
| P3 long | 0.7062 | 0.4048 | 0.8387 | 0.3044 |

### 同口径 CSD/DINO 评测

| 指标 | P3 RealPair FM | B3 λ20 | B4 λ50 | P2 TopK25 S8 |
|---|---:|---:|---:|---:|
| Top-1 | 17.0% | 17.8% | 20.8% | 33.0% |
| DINO-S | 0.4056 | 0.4183 | 0.4214 | 0.4389 |
| DINO-C | 0.7468 | 0.8325 | 0.8199 | 0.7552 |
| LPIPS | 0.5714 | 0.4412 | 0.4539 | 0.5095 |

相对 B3：ΔTop-1 -0.8pp，ΔDINO-S -0.0126，ΔDINO-C -0.0857，ΔLPIPS +0.1302。判定 **NO**。

**结论**：P3 路线训练稳定，但不具备竞争力；不再扩大训练。

## 后续建议

1. **优先继续 P2 TopK 方向**，重点研究如何在保持 DINO-S 提升的同时减少 DINO-C/LPIPS 损伤。
2. 可尝试更细粒度的门控、内容感知 TopK、多尺度步长或与轻量内容正则联合。
3. 暂停 P3 大规模训练，除非引入新的内容保持机制。
4. 保持现有 750 任务协议与指标定义不变；后续比较必须使用同一口径。

## 产物

- 总报告：`docs/v6/W_REPORT.md`
- 路线计划：`docs/v6/W_NEXT_PLAN.md`
- P1 报告：`docs/v6/W_P1_REPORT.md`
- P2 评测：`runs/v6_w/eval_gated_*`
- P3 同口径评测：`runs/v6_w/eval_p3_real_pair_2000/eval.json`
- 配对比较：`runs/v6_w/w_paired_report.txt`
