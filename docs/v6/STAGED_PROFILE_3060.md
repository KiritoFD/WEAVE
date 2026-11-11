# 3060 分段推理 Profile（与论文同硬件 / 同 ckpt / 同 batch）

**日期**：2026-11-12
**机器**：`ssh 3060`（Windows + **RTX 3060 12GB**，正是论文 `generation_only_timing_summary.json` 那张卡）
**脚本**：`tools/profile_staged_3060.py`（在 3060 仓库 `I:\Latent_Style\WEAVE\tools\`）
**结果**：`deliver/staged_profile_3060_b8.json`

## 0. 和论文基线的可比性

| | 论文 ledger | 本次 profile |
|---|---|---|
| GPU | **RTX 3060 12GB** | **同一张 RTX 3060 12GB** ✅ |
| WEAVE ckpt | `canonical_oriented_epoch4/epoch_0004.pt`（主表 4-epoch） | **同一个 ckpt**（3060 上 `archive/runs_submission/` 里有）✅ |
| 生成步数 | 8-step Euler | 8-step Euler（`num_steps=8`）✅ |
| batch | 8 | 8 ✅ |
| 精度 | FP16 / bf16 | Encoder/Decoder FP16、WEAVE bf16 autocast ✅ |
| 计时范围 | generation-only（encode+ODE+decode，排除指标/缓存/序列化） | 同 ✅ |
| **后端** | **ONNX Runtime FP16 encoder + TensorRT FP16 decoder（ws 2GB）** | **PyTorch eager**（官方 VAE / 自建 S10）❌ **唯一差异** |
| 计时方式 | wall clock（750 图） | `torch.cuda.Event`，warmup 5 + 20 iters |

> 由于后端不同（TRT 有算子融合/固定 shape 优化，PyTorch eager 无），**绝对 ms 不能直接相除**；但**同一进程内 4 个 Decoder 的相对比较、以及分段占比结构**是严格可比的——所有变体共享同一个 Encoder 输出与同一批 WEAVE 输出，只有 Decoder 不同。

## 1. 分段用时（batch 8，ms/batch，括号 ms/img）

| 段 | 用时 | ms/img | 说明 |
|---|---:|---:|---|
| **Encoder** | 523.08 | 65.39 | 官方 sd-vae-ft-ema encode，FP16 |
| **WEAVE 8-step** | 207.17 | 25.90 | 论文主 ckpt，bf16 autocast |
| **Decoder 官方 49.49M** | 1734.70 | **216.84** | PyTorch eager |
| **Decoder 旧 S10** | 550.67 | **68.83** | 10.66M，A0 训练 |
| **Decoder 新 C0** | 550.42 | **68.80** | 10.66M，本轮最佳重建 |
| **Decoder 新 C1** | 550.65 | **68.83** | 10.66M，本轮最佳响应 |

## 2. 各 Decoder 的整条流水与分段占比

| Decoder | total ms/batch | **ms/img** | Encoder % | WEAVE % | **Decoder %** | Decoder 加速 | 整条加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 官方 49.49M | 2464.96 | 308.12 | 21.2% | 8.4% | **70.4%** | 1.00× | 1.00× |
| 旧 S10 | 1280.93 | 160.12 | 40.8% | 16.2% | **43.0%** | **3.15×** | **1.92×** |
| **新 C0** | **1280.68** | **160.08** | 40.8% | 16.2% | **43.0%** | **3.15×** | **1.93×** |
| **新 C1** | 1280.91 | 160.11 | 40.8% | 16.2% | 43.0% | **3.15×** | **1.92×** |

**关键结论**：
1. **新 C0/C1 与旧 S10 速度完全相同**（550.42 / 550.65 / 550.67 ms，差 <0.05%）—— 符合同拓扑同参数量（10.658M）、只改训练目标的设计。
2. **Decoder 段加速 3.15×**（PyTorch 口径），与历史 4090 报告的 2.86× 同量级；倍数差异来自 GPU/后端/计时窗口。
3. **换 S10 后瓶颈转移**：官方 Decoder 占 70.4%，换成 S10 后降到 43.0%，Encoder 反而成为最大段（40.8%）。这意味着继续优化 Decoder 的边际收益已经很小，**要再提速得动 Encoder**（但本轮方案明确冻结 Encoder，不做）。

## 3. 与论文 ledger 的对照（同一张卡，不同后端）

论文（TRT）：`total 97.3 ms/img = encoder+WEAVE 18.7 + TRT decoder 78.6`，占比 19% / **81%**。
本次（PyTorch）：官方 Decoder `total 308.12 ms/img = E 65.39 + W 25.90 + D 216.84`，占比 21.2% / 8.4% / **70.4%**。

- **占比结构一致**：两者都是「Decoder 主导（70–81%）」，Encoder+WEAVE 只占 19–30%。这验证了论文「TRT decoder 是 81% 瓶颈」的判断，在 PyTorch eager 下同样成立（70.4%）。
- **绝对值差 ~3.2×**：TRT decoder 78.6 ms/img vs PyTorch 官方 216.84 ms/img；Encoder+WEAVE 18.7 vs 91.29 ms/img。差距来自 ONNX/TRT 的算子融合与固定 batch 优化 —— 也解释了为什么论文能用 TRT 把 750 图压到 73 s。
- **若把本次的 S10 换进论文的 TRT 流水**：按「TRT/PyTorch 官方 decoder = 78.6/216.84 = 0.362」外推，S10 TRT decoder ≈ 68.83×0.362 ≈ **24.9 ms/img**，整条 ≈ 18.7 + 24.9 ≈ **43.6 ms/img（约 2.2× 于论文的 97.3）**。这是**外推估计，非实测**；要证实必须把新 S10 导出 ONNX 并重建 TRT engine（引擎绑定固定权重，不能直接换）。

## 4. 与论文基线的结论（用户要的「对比速度」）

| 口径 | 官方 Decoder | S10 类 Decoder | 加速 |
|---|---:|---:|---:|
| 论文（3060 + TRT + bs8，实测） | 97.3 ms/img 整条（TRT decoder 78.6） | 无 | — |
| 本次（3060 + PyTorch + bs8，实测） | 308.12 ms/img 整条 | **160.08 ms/img** | **1.92×**（整条）/ **3.15×**（Decoder 段） |
| 历史榜单（4090 + PyTorch + bs1，Decoder 段） | 44.61 ms | 15.57 ms | 2.86× |

**一句话**：在论文同一张 3060、同一个 4-epoch 主 ckpt、同样 8-step / batch 8 下，只做 PyTorch→PyTorch 的公平对照，**新 C0/C1 与旧 S10 一样快，整条生成比官方 49.49M Decoder 快约 1.9×、Decoder 段快约 3.15×**；论文的 97.3 ms/img 是 TRT 加速后的数字，本轮没重建 TRT engine，所以不能直接说「比论文快/慢多少」，只能给出同硬件同配置下的相对收益与外推估计。

## 5. 产物 / 复现

- 结果：`deliver/staged_profile_3060_b8.json`（含 `paper_ledger` 原文、`paper_reference`、4 变体全字段）
- 脚本：`tools/profile_staged_3060.py`（本机 + 3060 各一份）
- 3060 上：`I:\Latent_Style\WEAVE\deliver\sd0_s10_profile\staged_profile_3060_b8.json`
- 复现：`cd /d I:\Latent_Style\WEAVE & python tools\profile_staged_3060.py --batch 8 --iters 20 --warmup 5`
- 已传入 3060 的权重：`deliver\sd0_s10_profile\sd0_s10_best_recon.pt`、`sd0_s10_best_overall.pt`、官方 SD0 VAE 的 `config.json` + `diffusion_pytorch_model.safetensors`（后两个其实 3060 本地已有，属冗余拷贝，不影响测速）
- 旧 S10 / A0 Encoder / 论文 ckpt / ONNX / TRT engine 均为 3060 本地原有，未改动
