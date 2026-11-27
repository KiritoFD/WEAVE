# 3060 同硬件分段 Profile 与 TensorRT b8 测速

> 本文合并两篇同主题报告：分段 profile（encoder / WEAVE / decoder 占比）与 TRT b8 测速。
> 两者均在**论文那张 RTX 3060 12GB** 上完成，是唯一与论文计时完全同硬件的实测。
> 口径辨析见 [07_INFERENCE_SPEED.md](07_INFERENCE_SPEED.md)。

---

## 第一部分：分段 Profile

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


---

## 第二部分：TensorRT batch-8 测速

# 新 S10 Decoder 的 TensorRT batch-8 测速（论文同一张 3060 / WSL）

**日期**：2026-11-12
**机器**：`ssh 3060` 的 **WSL**（RTX 3060 12GB，torch 2.10+cu128、TensorRT 11.4.0）
**口径**：与论文 `generation_only_timing_summary.json` 的 TRT decoder 段**完全同口径**——同一张卡、FP16 STRONGLY_TYPED、workspace 2GB、固定 shape、batch 8、512px、CUDA Event 计时（warmup 10 + 50 iters）。

## 结果

| Decoder TRT FP16 b8 | 参数 | ms/batch | **ms/img** | 吞吐 | 加速 |
|---|---:|---:|---:|---:|---:|
| 官方 sd-vae-ft-ema 49.49M（论文现成引擎 `vae_decoder_512_b8_fp16.engine`） | 49.49M | 586.68 | **73.34** | 13.6 img/s | 1.00× |
| **新 SD0-S10 C0（本轮新建引擎 `s10new_c0_decoder_512_b8_fp16.engine`）** | 10.66M | 216.61 | **27.08** | 36.9 img/s | **2.71×** |

新引擎 24.2 MB，构建 115 秒。

## 与论文 ledger 的对照

论文（同 3060、TRT、b8）：整条 97.3 ms/img = `encoder+WEAVE 18.7` + **`TRT decoder 78.6`**。

本次重测官方 TRT decoder 得 **73.34 ms/img**，与论文 78.6 同量级（差 6.7%，来自 TRT 版本 10→11 / 计时窗口 750图 vs 50 iters），**口径对齐成立**。

因此把新 S10 代入论文整条流水：

```
新 S10 整条 ≈ encoder+WEAVE 18.7 + TRT S10 decoder 27.08 ≈ 45.8 ms/img
vs 论文官方 Decoder 97.3 ms/img  →  约 2.13× 整条加速
```

（这是用论文 enc+WEAVE 段 + 本轮实测 TRT S10 decoder 的**拼接估计**，非端到端重测；要精确需把 ONNX encoder 也接进同一计时循环。）

## 与前面 PyTorch 分段 profile 的关系

| 后端 | 官方 decoder | S10 类 decoder | 加速 |
|---|---:|---:|---:|
| PyTorch eager（3060，Windows python，b8） | 216.84 ms/img | 68.80 ms/img | 3.15× |
| **TensorRT FP16（3060，WSL，b8）** | **73.34 ms/img** | **27.08 ms/img** | **2.71×** |

- TRT 相对 PyTorch：官方 2.96×、S10 2.54×。
- 用户要求「只测一个 S10」：因 C0/C1/旧 S10 同拓扑同参数量、PyTorch 口径下三者速度差 <0.05%，故只建/测 c0 一个引擎，结论对三个 S10 都成立。
- **batch 12 未做**（按用户最新指示取消）。

## 分段占比（用 TRT decoder 数字重算整条，同 3060）

论文 enc+WEAVE 18.7 ms/img 不变：

| 配置 | enc+WEAVE | decoder | 整条 | Decoder 占比 |
|---|---:|---:|---:|---:|
| 官方 TRT（论文） | 18.7 | 78.6 | 97.3 | **80.8%** |
| 官方 TRT（本轮重测） | 18.7 | 73.34 | 92.0 | **79.7%** |
| **新 S10 TRT（本轮）** | 18.7 | **27.08** | **45.8** | **59.1%** |

换 S10 后 Decoder 占比从 ~80% 降到 ~59%，但仍是最大段；Encoder+WEAVE 段（18.7 ms/img，本轮 PyTorch 实测 91.3 ms/img）若也做 TRT/ONNX 才有下一步空间——本轮不做。

## 产物 / 复现

- 结果：`deliver/s10_trt_b8_bench.json`；3060 侧 `runs/trt/engines_s10new/one_shot_s10_trt_b8.json`
- 引擎：`runs/trt/engines_s10new/s10new_c0_decoder_512_b8_fp16.engine`（24.2 MB）
- ONNX：`runs/trt/onnx/s10new_c0_decoder_512{,_fp16}.onnx`（42.7 MB / 21.4 MB，ORT fp32 校验 ~7.6e-6）
- 脚本：`tools/trt/one_shot_s10_trt_b8.py`（一键：fp16 转换 → 建引擎 → 测两个引擎）、`tools/trt/bench_s10_trt.py`（dtype 自适应 + `execute_v2/v3` 兼容）、`tools/trt/export_s10_new_onnx.py`
- 复现（WSL）：`cd /mnt/i/Latent_Style/WEAVE && python3 tools/trt/one_shot_s10_trt_b8.py`
- 官方 baseline 引擎 `runs/trt/engines/vae_decoder_512_b8_fp16.engine` 为 3060 原有资产，**未改动、未覆盖**

## 备注（工程坑，已修）

- 3060 的运行环境是 **WSL**，不是 Windows cmd/python；Windows 侧 python 与 WSL 的 tensorrt 版本/绑定不同（`execute_v3` 在 WSL TRT 11.4 不存在，已加 `execute_v2` 回退）。
- fp16 ONNX 必须用仓库的 `convert_onnx_fp16.py`（统一 fp16、STRONGLY_TYPED），不能直接用 fp32 ONNX 建 fp16 引擎。
- `build_trt_engines.py` 的 `logger` 是未定义的模块全局，import 调用需注入 `trt.Logger`（一次性脚本已处理）。
