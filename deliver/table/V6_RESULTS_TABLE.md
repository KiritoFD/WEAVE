# V6 下一阶段 · 统一结果表（交付用）

**评测器**：`paper_rgb_v1` 评分链路（CSD / DINOv2 / LPIPS + TRT fp16 VAE 解码 → uint8）
**协议**：750 任务 = **600 cross + 150 same**，150 源图，每风格 120/120 均衡
**Cross600 指标一律为 cross-only**（按 `tgt_style != src_style` 分离；
`eval_tiers` 的行里不含 `is_cross`，用默认值会把 750 全算成 cross，Top-1 虚高 3.4 倍）

| 模型 | 参数量 | Cross600 Top-1 | DINO-C | LPIPS | ΔT | Margin | Minimalism 误判 | 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Q2a Direct | 218K | **6.7%** | 0.879 | 0.273 | +0.0487 | -0.2536 | 26.2% | MEASURED: `P0 A0, Test600 (750 protocol)` |
| Q2a + R1 Oracle (1 step) | 218K + R1 | **36.5%** | 0.710 | 0.415 | +0.0343 | -0.0528 | 31.0% | MEASURED: `P0 A1, Test600 (750 protocol)` |
| Q2a + R1 Oracle (4 steps dyn) | 218K + R1 | **24.2%** | 0.796 | 0.355 | +0.0566 | -0.1146 | 26.9% | MEASURED: `P0 A2, Test600 (750 protocol)` |
| V6 ScoreDiff N=24 | 3.27M | **35.2%** | 0.592 | 0.474 | +0.0686 | -0.0434 | 36.0% | MEASURED: `P0 A4, Test600 (750 protocol)` |
| V6 ScoreDiff N=12 strong | 3.27M | **30.8%** | 0.624 | 0.492 | +0.0642 | -0.0588 | 35.0% | MEASURED: `eval_750_scorediff_N12` |
| B0 218K control (continued) | 218K | **7.5%** | 0.870 | 0.405 | +0.0338 | -0.2635 | 27.5% | MEASURED: `eval_750_P1_B0_ctrl218` |
| B3 distilled lambda=20 | 218K | **20.8%** | 0.810 | 0.462 | +0.0609 | -0.1373 | 28.5% | MEASURED: `eval_750_P1_B3_distill20` |
| B4 distilled lambda=50 | 218K | **24.2%** | 0.796 | 0.474 | +0.0602 | -0.1201 | 29.2% | MEASURED: `eval_750_P1_B4_distill50` |

## 分目标 Top-1（Cross600）

| 模型 | Early_Ren | Impressionism | Minimalism | Rococo | Ukiyo_e |
|---|---:|---:|---:|---:|---:|
| Q2a Direct | 3.3% | 10.0% | 8.3% | 10.8% | 0.8% |
| Q2a + R1 Oracle (1 step) | 4.2% | 64.2% | 73.3% | 35.8% | 5.0% |
| Q2a + R1 Oracle (4 steps dyn) | 2.5% | 36.7% | 45.0% | 35.8% | 0.8% |
| V6 ScoreDiff N=24 | 14.2% | 40.8% | 95.8% | 15.8% | 9.2% |
| V6 ScoreDiff N=12 strong | 11.7% | 26.7% | 97.5% | 9.2% | 9.2% |
| B0 218K control (continued) | 1.7% | 11.7% | 15.0% | 7.5% | 1.7% |
| B3 distilled lambda=20 | 3.3% | 32.5% | 45.0% | 21.7% | 1.7% |
| B4 distilled lambda=50 | 3.3% | 40.0% | 52.5% | 23.3% | 1.7% |

## 同风格（Same150）

| 模型 | Same Top-1 | Same ΔS |
|---|---:|---:|
| Q2a Direct | 98.0% | +0.0184 |
| Q2a + R1 Oracle (1 step) | 94.7% | -0.1545 |
| Q2a + R1 Oracle (4 steps dyn) | 98.0% | -0.0573 |
| V6 ScoreDiff N=24 | 94.7% | -0.0000 |
| V6 ScoreDiff N=12 strong | 96.7% | -0.0000 |
| B0 218K control (continued) | 98.0% | +0.0162 |
| B3 distilled lambda=20 | 97.3% | -0.0052 |
| B4 distilled lambda=50 | 98.0% | -0.0198 |

## 未在本轮测量、仅作参照（来源不是本次运行）

| 模型 | 参数量 | Top-1 | DINO-C | 来源 |
|---|---|---:|---:|---|
| V6 D1_T1 | 1.88M | 6.5% | 0.870 | 早期 `paper_rgb_v1` 结果，非本轮 |
| 历史 WEAVE 原版 | 待核实 | 16.7% | 0.808 | 已发表系统，非本轮 |
| L11 Direct | 218K | ~3% | ~0.89 | 本轮 Heldout400（400 任务），**不是** Cross600 协议 |
