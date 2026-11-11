# 推理速度可比性说明（2026-11-12）

用户问：本轮新 Decoder 的推理速度，有没有可比数据？论文里的推理速度是什么硬件口径？

**结论先说**：**没有可直接比较的同口径数据**。仓库里存在三套互不兼容的速度口径，且论文/榜单的速度数字**不是**「32 个 baseline 方法各自的推理时间」，而是「我们自己的 VAE Decoder 段」或「整条 WEAVE 生成流水」的实测。本轮新 Decoder 结构未变（同为 10.658M S10），所以推理速度**预期与旧 S10 相同**，但需按同口径重新实测才算数。

---

## 1. 三套口径，互不可比

| # | 来源 | 数字 | 硬件 | 测的是什么 | 后端 / batch | 与什么可比 |
|---|---|---|---|---|---|---|
| **A** | `docs/baseline/UNIFIED_FULL_RANKING.md` §5、[`docs/1111/vae/03_s10_decoder_pruning_and_recovery.md`](../1111/vae/03_s10_decoder_pruning_and_recovery.md) §P2、[`docs/v6/S10_FINAL_REPORT.md`](S10_FINAL_REPORT.md) | A0 Decoder **44.61 ms** → S10 **15.57 ms**（**2.86×**）；forward+input-grad 120.51 → 39.90 ms（3.02×）；峰值显存 11.29→6.36 GB | **单卡 RTX 4090 24GB** | **只有 Decoder 段的 forward / forward+input-grad**，不是端到端 | PyTorch，`bs=1`，512px，**同机同基准** | 只和其他「4090 / bs1 / 纯 Decoder 段」比 |
| **B** | 论文 [`aaai2027_v4/supplement.tex`](../../aaai2027_v4/supplement.tex) §Evaluation Runtime Anatomy（约 633–690 行、1440 行） | 整条流水 **73.0 s / 750 图 = 97.3 ms/img**（10.3 pairs/s）；其中 `encoder+WEAVE` 14.0 s = **18.7 ms/img**（19%）、**TRT decoder 59.0 s = 78.6 ms/img**（81%） | **单卡 RTX 3060 12GB** | **生成-only 端到端**（encode + 8-step ODE + decode；排除 latent 缓存、指标提取、序列化） | **ONNX Runtime FP16 encoder + TensorRT FP16 decoder**，workspace 2 GB，**batch 8**，750 图 | 只和「3060 / TRT / bs8 / 整条 WEAVE」比 |
| **C** | 本轮 [`eval_p5_final.json`](../../deliver/sd0_s10_eval_p5_final.json) `infra` | official 57.83 ms / S10 类 18.1 ms（forward）；forward+grad 122.75 / 58.5 ms；峰值 2.41 / 1.97 GB（forward） | **单卡 RTX 4090 24GB** | 纯 Decoder forward / forward+input-grad | PyTorch，`bs=1`，512px，`iters=10`，3 次 warmup | 与 **A 同口径**（都是 4090/bs1/PyTorch/Decoder 段） |

**为什么 A 与 C 的数字对不上**：A 的 44.61/15.57 ms 是历史 S10 报告里的独立基准进程；C 的 57.83/18.1 ms 是本轮 `eval_sd0_s10.py` 在 P5 里现测的。两者都是 4090/bs1/PyTorch，但**进程、残留显存、warmup 次数、计时窗口不同**（C 用 `iters=10` 且 `torch.cuda.synchronize()`，A 的原脚本未见）。相对关系一致（S10 类约为官方的 1/3），但**绝对毫秒数不能混用**。

**为什么 B 不能与 A/C 比**：
- 硬件不同：3060 12GB vs 4090 24GB
- 后端不同：ONNX/TensorRT FP16 vs PyTorch（FP32/BF16）
- batch 不同：8 vs 1
- 测量对象不同：端到端（含 encoder + ODE） vs 纯 Decoder 段
- 论文特别注明：TRT decoder 的 **78.6 ms/img 是 TRT 在 3060 上的数字**，而榜单里的 15.57 ms 是 4090 上的 PyTorch 纯 Decoder 数字。二者相差 5 倍主要来自硬件+后端+测量范围，**不是矛盾，也不是同一件事**。原报告 §P2 已明确写「**不能把这个加速倍数套到 3060 整条推理**」。

---

## 2. 「论文中的推理速度」到底是什么口径

补充材料原文：

> *"The paper reports the measured generation-only time for the selected D5 checkpoint: 73.0 s for 750 images (97.3 ms/image) on one **RTX 3060 (12 GB)** under the audited **ONNX/TRT pipeline (batch 8, decoder workspace 2 GB)**. Stage breakdown: encoder+WEAVE 14.0 s (18.7 ms/image, 19%), **TRT decoder 59.0 s (78.6 ms/image, 81%)**. The timing includes the 8-step trajectory and VAE encode/decode, but excludes latent caching, metric extraction, and result serialization. Larger decode batch (16) triggers TRT build OOM on a 12 GB card... Increasing decoder workspace from 2 GB to 4 GB improves decode by only ≈0.6%."*

即：
- **主表/补充材料的 97.3 ms/img 是 3060 上整条 WEAVE 生成流水**，不是单个 Decoder，也不含 32 个外部 baseline 方法的速度。
- 论文的 timing 表只针对**我们选中的 D5 checkpoint**，用 `docs/generation_only_timing_summary.json` 作 ledger。
- **32 方法排行榜（`docs/baseline/UNIFIED_FULL_RANKING.md`）里没有任何外部 baseline 的推理时间列**，只有指标（Affinity / DINO-S / CLIP-S / DINO-C / LPIPS / CSD Top-1）。文中出现的「2.86× / 15.57 ms」是**我们自己的 S10 Decoder 替换收益**（§5、§6），不是 baseline 对比。
- 另有一处「47 ms 推理时延（21.4 FPS）」出现在 §6 结论里，指的是 `P1_B4_distill50` 这类 **218K 单步生成器**的整条推理，同样不是 32 baseline 的可比速度。

---

## 3. 本轮新 Decoder 的可比结论

| 判断 | 证据 |
|---|---|
| **结构未变，速度预期不变** | C0/C1 同为 `widths_forward=[64,128,192,256]`、**10.658M** 参数，与旧 S10 完全同拓扑；本轮只改了训练目标（Haar HF / 响应蒸馏），不改算子/通道数 → forward 计算量不变 |
| **本轮实测（口径 C）** | official 57.83 ms；`old_s10` 18.10、`c0` 18.10、`c1` 18.09 ms（forward，bs1，4090，PyTorch）—— 三个 S10 类 Decoder **在测量噪声内完全相同**；forward+grad 58.49/58.53/58.53 ms 同样一致；峰值显存 1.97 GB 均相同 |
| **相对官方加速** | 57.83 / 18.10 ≈ **3.20×**（本轮 C 口径）；历史 A 口径是 44.61 / 15.57 ≈ 2.86×。**倍数不同源于分母测法，不是新 Decoder 变快了** |
| **与论文 B 口径** | 无法直接比。若要可比，必须在 3060 上用 ONNX/TensorRT FP16、bs8、workspace 2GB 重编并实测整条流水；本轮**未做 TensorRT 实测** |

**一句话**：新 C0/C1 的推理速度与旧 S10 相同、比官方 49.49M Decoder 快约 3×（同机 4090/bs1/PyTorch 纯 Decoder 段口径）；但**论文/榜单没有任何「32 个 baseline 各自推理速度」的表**，论文唯一的速度数字是 3060 + ONNX/TRT + bs8 的整条 WEAVE 生成流水 97.3 ms/img，与本轮 Decoder 段数字不同硬件、不同后端、不同测量范围，**不能直接拿来排名**。

---

## 4. 若要产出可比数据，需要补什么

1. **同口径重测历史数字**：用本轮 `eval_sd0_s10.py:infra()` 的同一进程/同一计时窗口，重测 A0(官方)、旧 S10、C0、C1 四个 Decoder，得到一张可内部配对的表（已部分具备，见口径 C）。
2. **TensorRT 实测**（方案 P5 第 4 层明确要求、本轮未完成）：在 4090 上把四个 Decoder 导出 ONNX → TRT FP16，按 bs1/bs8 分别测 forward 与峰值显存；若要对齐论文，需要**在 3060 12GB 上**按 bs8 / workspace 2GB 重跑整条流水。
3. **不要**把 15.57 ms（4090 PyTorch Decoder）与 78.6 ms（3060 TRT Decoder）相除或并列 —— 硬件、后端、batch 全不同。

## 5. 原始出处

- 论文：`aaai2027_v4/supplement.tex` §Evaluation Runtime Anatomy / §Pipeline timing (audited) / §TensorRT compile
- 榜单：`docs/baseline/UNIFIED_FULL_RANKING.md` §5（44.6→15.6 ms）、§6（2.86× / 15.57 ms）
- 历史 S10：`docs/v6/S10_FINAL_REPORT.md` 表 3（44.61/15.57/120.51/39.90 ms）、`docs/1111/vae/03_s10_decoder_pruning_and_recovery.md` §P2
- 本轮：`deliver/sd0_s10_eval_p5_final.json` 各变体的 `infra` 字段；`tools/vae_suite/eval_sd0_s10.py:infra()`
