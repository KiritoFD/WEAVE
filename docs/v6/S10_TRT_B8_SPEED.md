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
