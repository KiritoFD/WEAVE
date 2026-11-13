# ONNX VAE 导出与 TRT 推理加速实验

## 实验目标

将 SD1.5 VAE（Encoder + Decoder）导出为 ONNX，使用 TensorRT 编译加速推理，评估端到端加速效果。

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 3060 12GB |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| TensorRT | 10.x |
| ONNX | 1.17 |
| 测试集 | D5-512（5 风格 × 30 内容 × 5 参考 = 750 对） |
| 分辨率 | 512×512 |
| 精度 | FP16 |

## ONNX 导出

### 导出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `encoder_b8.onnx` | 68.4 MB | VAE Encoder, batch_size=8 |
| `encoder_b16.onnx` | 68.4 MB | VAE Encoder, batch_size=16 |
| `decoder_b8.onnx` | 99.1 MB | VAE Decoder, batch_size=8 |
| `decoder_b16.onnx` | 99.1 MB | VAE Decoder, batch_size=16 |

所有 ONNX 文件位于 `data/onnx_exported/`。

### 导出脚本

- `experiments/inference_speed/export_decoder.py` — Decoder 导出
- Encoder 导出使用 `torch.onnx.export()` 直接导出

### 导出注意事项

1. VAE Encoder 和 Decoder 分别导出，WEAVE 模型本身不需要导出（仍用 PyTorch）
2. Decoder 的 `scaling_factor = 0.18215` 需要在后处理中手动除以该值
3. 导出时使用 `dynamic_axes` 或固定 batch size；固定 batch 可启用更多 TRT 优化

## TRT 编译

### 编译配置

| 参数 | 值 |
|------|-----|
| Batch Size | 8（推荐） |
| Workspace | 2GB ~ 4GB |
| 精度 | FP16 |
| 编译时间 | ~15 min（Decoder, 2GB workspace） |

### Batch Size 选择

| Batch Size | 结果 |
|------------|------|
| 8 | 正常编译，推理稳定，VRAM 使用 ~10GB |
| 16 | **OOM**：TRT 编译时请求 10.7GB VRAM，超出 RTX 3060 12GB 余量，导致跳过优化，实际推理反而更慢 |

### Workspace 对比

| Workspace | Decoder 速度 | 改善 |
|-----------|-------------|------|
| 2GB | 58.98s / 750 图 | baseline |
| 4GB | 58.63s / 750 图 | **+0.6%**（可忽略） |

结论：Decoder 架构已接近 RTX 3060 推理极限，增大 workspace 无显著收益。

## 端到端 Benchmark

### 流水线架构

```
输入图片 (512×512) → [VAE Encoder ONNX] → latent (4×64×64) → [WEAVE PyTorch] → styled latent → [VAE Decoder TRT] → 输出图片 (512×512)
```

### 分阶段耗时（750 张图，batch_size=8，TRT Decoder workspace=2GB）

| 阶段 | 后端 | 耗时 | 每图耗时 | 吞吐 | 占比 |
|------|------|------|---------|------|------|
| Stage 1: VAE Encoder | ONNX Runtime | — | — | — | — |
| Stage 2: WEAVE Transfer | PyTorch (bf16) | — | — | — | — |
| Stage 1+2 合计 | — | **14.0 s** | 18.7 ms | 53.6 img/s | 19.2% |
| Stage 3: VAE Decoder | **TensorRT (FP16)** | **58.98 s** | 78.6 ms | 12.7 img/s | **80.8%** |
| **流水线总计** | — | **73.0 s** | 97.3 ms | 10.3 img/s | 100% |

> 注：Encoder 和 WEAVE 在 pipeline 中紧密耦合，计时为合并值。WEAVE 模型本身仅 1.04M 参数，推理极快；Encoder 是主要耗时来源。

### TRT 引擎编译耗时（一次性）

| 操作 | 耗时 | 说明 |
|------|------|------|
| Decoder ONNX → TRT Engine (b=8, 2GB) | **~15 min** | 首次编译，后续可缓存复用 |
| Decoder ONNX → TRT Engine (b=16, 4GB) | OOM | 请求 10.7GB VRAM，超出 12GB 余量 |

### Decoder 后端方案对比（750 张图）

| 方案 | 总耗时 | 每图耗时 | 吞吐 | 加速比 | 备注 |
|------|--------|---------|------|--------|------|
| PyTorch eager (fp32) | ~129.3 s | 172.4 ms | 5.8 img/s | 1.00× | 基线 |
| PyTorch eager (fp16) | ~100.8 s | 134.4 ms | 7.4 img/s | 1.28× | 仅改精度 |
| torch.compile (max-autotune, fp16, b=16) | **75.6 s** | 100.8 ms | 9.9 img/s | 1.71× | 编译缓存 174.6s |
| **TRT (b=8, fp16, ws=2GB)** | **58.98 s** | 78.6 ms | 12.7 img/s | **2.19×** | 最优 |
| TRT (b=8, fp16, ws=4GB) | 58.63 s | 78.2 ms | 12.8 img/s | 2.21× | 仅 +0.6% |

### torch.compile 详细结果（仅 Decoder，b=16）

| 指标 | 值 |
|------|-----|
| 首次编译耗时 | 174.6 s（一次性） |
| 3 次运行耗时 | 75621 / 76080 / 76472 ms |
| 最优耗时 | 75621 ms（75.6 s） |
| 每图耗时 | 100.8 ms |
| 吞吐 | 9.9 img/s |
| 加速比 (vs baseline) | 1.71× |
| 优化技术 | fp16, channels_last, CUDA Graph Trees, max-autotune, fullgraph, static shape, inference_mode |

### 各阶段显存占用（估算，b=8）

| 阶段 | VRAM | 说明 |
|------|------|------|
| Encoder ONNX | ~3 GB | 输入 512×512 图片 |
| WEAVE PyTorch | ~2 GB | 1.04M 参数 + latent 4×64×64 |
| Decoder TRT (ws=2GB) | ~10 GB | 峰值在 TRT engine 推理 |
| 流水线总计 | **~10 GB** | RTX 3060 12GB 余量充足 |

### 完整流水线耗时分解（750 张图，最优配置 b=8, TRT ws=2GB）

```
┌─────────────────────────────────────────────────────────┐
│ 总耗时: 73.0s (100%)                                      │
├──────────────┬──────────────────────────────────────────┤
│ Enc+WEAVE    │  VAE Decoder TRT                          │
│   14.0s      │            58.98s                         │
│  (19.2%)     │           (80.8%)                         │
└──────────────┴──────────────────────────────────────────┘
```

## 关键结论

1. **Decoder 是瓶颈**：占总耗时 80.8%，是优化的主要目标
2. **TRT 加速显著**：Decoder 从 ~130s 降至 58.98s，2.2× 加速
3. **RTX 3060 已到极限**：增大 workspace 或 batch size 均无额外收益
4. **Encoder + WEAVE 很快**：仅 14s，无需 ONNX 导出
5. **推荐配置**：batch_size=8, TRT FP16, workspace=2GB

## 文件清单

### 本地

```
WEAVE/data/onnx_exported/
├── encoder_b8.onnx          # VAE Encoder ONNX (b=8)
├── encoder_b16.onnx         # VAE Encoder ONNX (b=16)
├── decoder_b8.onnx          # VAE Decoder ONNX (b=8)
└── decoder_b16.onnx         # VAE Decoder ONNX (b=16)
```

### 远程（I:\Github\Latent_Style\WEAVE\）

```
experiments/inference_speed/
├── export_decoder.py         # Decoder ONNX 导出脚本
├── run_end_to_end_750.py     # 端到端 benchmark
├── run_pipeline_750.py       # 流水线 benchmark
├── search_trt.py             # TRT 参数搜索
├── stage4_ort_decode.py      # ONNX Runtime decode
├── vae_decode_optimized.py   # torch.compile 优化版 decode
├── vae_encode_optimized.py   # 优化版 encode
├── diagnose_vae.py           # VAE 诊断
├── inspect_fusion.py         # TRT fusion 检查
├── test_ort.py               # ONNX Runtime 测试
├── results/
│   ├── latents_750.pt        # 预计算的 750 latent
│   └── vae_decode_optimized.json  # torch.compile benchmark 结果
├── exported/
│   └── decoder.onnx          # 旧版 decoder export
└── .compile_cache/           # torch.compile 缓存
```