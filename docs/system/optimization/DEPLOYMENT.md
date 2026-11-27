# CLEAVE V35 — RTX 3060 生产部署包与运行指南（纯 PyTorch 零外部引擎版）

> 本文档说明如何在 RTX 3060 12GB（及同等 SM86+ CUDA 环境）构建、校验并部署 **CLEAVE V35** 紧凑推理包，无需安装 TensorRT / ONNX Runtime 等额外推理引擎，仅依托原生 PyTorch 2.11 + Triton。

---

## 1. 紧凑部署包规格 (`runs/deploy_bundle_v35/`)

通过 [`exp/deploy_v35/bundle.py`](../../../exp/deploy_v35/bundle.py) 可从训练仓库自动剥离全部 GB 级训练缓存（`ccf_pool_cache.pt` 等），抽出纯推理必需的 **8 个文件（合计 `522,191,744` 字节 = `498.0 MiB`）** 并生成 SHA-256 校验清单：

| 资产文件 | 来源路径 | 大小 (Bytes) | SHA-256 前缀 | 所属通路 |
| --- | --- | --- | --- | --- |
| `pcore.pt` | `runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt` (`ema`) | 178,204,478 | `50bcbb4fbc3daf52` | 仅 `E→P` 延迟加载 |
| `ecore.pt` | `runs/v32_nodino/ecore/ECore_NoDINO/ECore_NoDINO_final.pt` (`ema`) | 44,586,942 | `9e226f6f8d1fc82f` | `E-only` / `E→P` |
| `a0_encoder.pt` | `runs/a0/a0_encoder_only.pt` | 136,691,814 | `70715532435ad6c2` | 编码前端 |
| `s10b_decoder.pt` | `deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt` | 42,682,134 | `ac0bf25007d28b22` | 解码输出 |
| `convnext_tiny.pth` | `torch.hub` 离线权重 `convnext_tiny-983f1562.pth` | 114,419,221 | `983f1562536e84ff` | 条件提取 |
| `k8_bank.pt` | `runs/v32_nodino/cache/k8_cnx417_bank.pt` | 5,552,862 | `481851e86278161b` | `E→P` K=8 锚点 |
| `a3_priors.pt` | `runs/v34_frontier/cache/v34_analytic_priors.pt` | 3,208 | `ae88d1423af9643d` | `E→P` A3 起点 / E-A3 |
| `tiny_tensors.pt` | 抽取自 Stiefel `W_stiefel_cnx` (384×32) 与 `luma_proj` (4+1) | 51,085 | `1edbaee7b0adc481` | 条件提取 |

### 构建与完整性验证

```bash
python -m exp.deploy_v35.bundle --bundle-dir runs/deploy_bundle_v35
pytest tests/test_deploy_v35.py -v
```

---

## 2. Python 运行时 API 与严格 `E-only` 显存隔离

核心入口位于 [`exp/deploy_v35/pipeline.py`](../../../exp/deploy_v35/pipeline.py) 与 [`exp/deploy_v35/torch_opt.py`](../../../exp/deploy_v35/torch_opt.py)。

### 2.1 `E-only` 独立快部署（不加载 `P-Core` 44.54M 权重）

```python
import torch
from exp.deploy_v35.pipeline import make_optimized_pipeline

# load_pcore=False 保证 P-Core 权重绝不实例化，显存占用最低
pipe_e = make_optimized_pipeline(
    "runs/deploy_bundle_v35",
    load_pcore=False,
    compile_mode="reduce-overhead",
    compile_batches=(1, 2, 4),
)
pipe_e.warmup(batch_sizes=(1, 2))  # 常驻服务启动预热一次

# rgb01: (B, 3, 512, 512) float32 [0, 1]；tid: (B,) int64；eps: (B, 4, 64, 64)
out = pipe_e.run_e_only(rgb01, tid, lam=0.60, eps=eps)
assert pipe_e.pcore is None and pipe_e.fast_pcore is None
```

### 2.2 `E→P` 完整双核管线（支持 `SourceFeatureCache` 与同图多旋钮交互）

```python
pipe_p = make_optimized_pipeline(
    "runs/deploy_bundle_v35",
    load_pcore=True,
    compile_mode="reduce-overhead",
    compile_batches=(1, 4),
)
pipe_p.warmup(batch_sizes=(1, 4))

# 交互式预览先编排源图一次，再连续调节旋钮 s=0.60 / 0.85 / 1.00 / 1.50
src = pipe_p.prepare_source(rgb01, need_y16=True)
out_085 = pipe_p.run_e_to_p(rgb01, tid, knob_s=0.85, eps=eps, cached_src=src)
out_100 = pipe_p.run_e_to_p(rgb01, tid, knob_s=1.00, eps=eps, cached_src=src)
```

---

## 3. RTX 3060 档位选型与启动配置建议

| 使用场景 | 推荐配置 | 固定 Batch | 稳态吞吐 / 延迟 | 显存 (`nvidia-smi`) | 冷启动特性 |
| --- | --- | --- | --- | --- | --- |
| **交互式单图实时编辑 (`E-only`)** | `Optimized` (`reduce-overhead`) | `B=1` | **`111.8 ms`** (`8.94 img/s`)；缓存源图 **`40.7 ms`** (`24.6 img/s`) | `~1.16 GB` | 服务常驻预热后零编译；开启 `TORCHINDUCTOR_FX_GRAPH_CACHE=1` |
| **批处理风格渲染 (`E-only`)** | `Optimized` (`reduce-overhead`) | `B=2` / `B=4` | **`81.2 ms/img`** (`12.31 img/s` @ `B=2`) / **`33.6 ms/img`** Hot `B=4` | `1.71 GB` (`B=2`) / `2.78 GB` (`B=4`) | 同上 |
| **高质量单图交互 (`E→P`)** | `Optimized` (`reduce-overhead`) | `B=1` | **`221.4 ms`** (`4.52 img/s`)；缓存源图 **`153.7 ms`** (`6.51 img/s`) | **`1.42 GB`** | 单批 `B=1` CUDAGraph 池开销极低 |
| **最高吞吐批处理 (`E→P`)** | `Optimized` (`reduce-overhead`) | **`B=2` 或 `B=4`** | **`180.3–182.9 ms/img`** (**`5.47–5.55 img/s`**)；缓存源图 **`133.6 ms/img`** | `~7.9–8.4 GB` (`B=4` 含缓存池) | 不要在同一进程混编多种不用的 `B`，避免累积多个 CUDAGraph 私有池 |
| **单次脚本 CLI（不常驻服务）** | `Optimized-Eager` (`compile_mode=""`) | `B=1` | `E-only 123.1 ms` / `E→P 305.4 ms` (`1.18×` 无编译) | `~1.4 GB` | 避免单次运行 48–60s 首帧编译/图捕获，总启动仅 ~5.5s |

### 关键运维注意事项

1. **S10B 输入缩放契约**：[`pipeline.py`](../../../exp/deploy_v35/pipeline.py) 与 `FastS10BDecoder` 严格内聚了 `zo / 0.18215` 缩放与内置 `post_quant_conv`，**禁止在外部重复除 `0.18215` 或重复调用 `post_quant_conv`**。
2. **A0 输入域契约**：`FastA0Encoder` 内部统一执行 `rgb01 * 2.0 - 1.0`，外部统一传入 `[0, 1]` 范围的 `(B, 3, 512, 512)` 图像张量。
3. **固定形状与持久化缓存**：生产部署只将实际使用的 `B`（如 `(1,)` 或 `(4,)`）写入 `compile_batches`，对于非整除尾批（如 `test750` 最后 2 张配 `B=4`）采用零拷贝填充到固定形状后切片，即可避免触发二次编译。默认持久化目录：`runs/v35_3060_opt/inductor_cache`。
4. **Windows PyTorch CUDAGraphTrees 内存提示**：`torch.compile(mode="reduce-overhead")` 会为每种 `(module, B)` 创建固定激活缓存池；在 `12GB` RTX 3060 上若只编译单一线上 `B`（如仅 `B=1` 或仅 `B=4`），显存完全可控；切忌在单进程内连续编译 `B=1,2,4,8,12,16` 并长期驻留。
