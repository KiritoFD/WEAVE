# CLEAVE V35 — RTX 3060 12GB P0 基准标定与瓶颈剖析

> **硬件环境**：Windows Server native Python 3.12.10 · `torch 2.11.0+cu128` · CUDA 12.8 / cuDNN 9.19.0 · `triton 3.7.1` · NVIDIA GeForce RTX 3060 12GB (SM86, 170W TDP, Driver 581.08, 12288 MiB VRAM)。
> **基线协议**：与 [3060 优化规范](../3060_PROFILE_OPTIMIZATION.md) 与 [系统总览](../README.md) 锁定同一参考模型与参数（A0 34.16M + CNX 12.35M + E-Core 11.13M + P-Core 44.54M + S10B 10.66M = 112.84M），测量脚本见 [`exp/deploy_v35/benchmark.py`](../../../exp/deploy_v35/benchmark.py) 与 [`exp/deploy_v35/parity.py`](../../../exp/deploy_v35/parity.py)。

---

## 1. P0 源图预处理与参考链路对齐（150 源图 + 16 样本分层校验）

在启动优化前，先核验从 `data/test/*.jpg` 原始 512×512 (`BICUBIC`) 输入出发、经 `A0 Encoder`（需 `[-1, 1]` 缩放：`rgb01 * 2 - 1`）与 `rgb_to_structure_64` 生成的结构特征与历史参考缓存的一致性，以及 Reference 链路运行时钩子计数（来自 `p0_parity.json`）：

| 核验项 | 样本范围 | Cosine Similarity | Mean Abs Diff | Max Abs Diff | 说明 |
| --- | --- | --- | --- | --- | --- |
| `z_s` vs `n2c_z.pt["z_src"]` | 全部 150 源图 | **0.9924** | 0.0563 | 3.1250 | 离线 A0-encoder vs 原始 SD1.5 VAE encoder |
| `c64` vs `test750_struct_cache.pt` | 全部 150 源图 | **0.9941** | 0.0237 | 1.8704 | 原始 JPG vs S10B 解码缓存半精度结构特征 |
| `E-only` (`l=0.60`, `load_pcore=False`) | 16 分层样本 | 1.0000 | 0.0000 | 0.0000 | **P-Core 未加载且调用 0 次** (`pcore_batch_calls=0`) |
| `s=0` 恒等门控 | 16 分层样本 | 1.0000 | 0.0000 | 0.0000 | `mae_255 = 0.00` |

### 关键基线发现（实测验证）

1. **`A0 Encoder` 域对齐修正**：`AutoencoderKL.encode` 严格要求 `[-1, 1]` 输入（`rgb01 * 2 - 1`）。直接输入 `[0, 1]` 会造成 `mae=0.557` 漂移（`[-1, 1]` 为 `0.050`）。
2. **`steps=6` 积分器实际 NFE（运行时钩子实测）**：P0 实测 `integrate_v33_trajectory` 在 `steps=6` 下，由于第 0 步 `prev_v is None` 自动回退 2 阶 midpoint，实际触发 **`pcore_batch_calls = 7.0`**（每次 4B CFG = **`28.0` 条件等价 NFE**），加上 `E-Core = 1.0` 共计 **`29.0` 流模型条件评估**，而非名义 `24 + 1 = 25`。

---

## 2. RTX 3060 P0 基线延迟、吞吐与显存全扫（B = 1..16）

### 2.1 `E-only` 独立快分支（`l=0.60`, `load_pcore=False`, 1 Flow NFE）

| 口径 / Batch | GPU P50 (ms/img) | GPU P95 (ms/img) | Wall P50 (ms/img) | 吞吐 (img/s) | Peak Alloc (MB) | Peak Reserved (MB) | nvidia-smi (MB) | 温度 / 功耗 / 频率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Host E2E `B=1` | 144.91 | 152.09 | 145.69 | 6.90 | 681.4 | 842.0 | 1257 | 57°C / 133W / 2077MHz |
| Host E2E `B=2` | 150.87 | 152.46 | 151.28 | 6.63 | 1072.0 | 1434.0 | 1851 | 62°C / 147W / 2062MHz |
| Host E2E `B=4` | 144.41 | 145.25 | 144.61 | 6.92 | 1855.1 | 2488.0 | 2909 | 66°C / 155W / 2047MHz |
| Host E2E `B=8` | 141.57 | 142.54 | 141.68 | 7.06 | 3421.4 | 4676.0 | 5097 | 67°C / 157W / 2010MHz |
| Host E2E `B=12` | **140.47** | 140.66 | 140.54 | **7.12** | 4987.6 | 6828.0 | 7245 | 69°C / 156W / 2047MHz |
| Host E2E `B=16` | 140.70 | 140.98 | 140.75 | 7.11 | 6553.9 | 9062.0 | 9479 | 70°C / 152W / 2032MHz |
| Hot Cached `B=1` | 61.68 | 63.67 | 62.43 | 16.21 | 588.6 | 906.0 | 1327 | 69°C / 141W / 2047MHz |
| Hot Cached `B=4` | 73.90 | 74.48 | 74.13 | 13.53 | 1327.6 | 2488.0 | 2909 | 69°C / 154W / 2032MHz |
| Hot Cached `B=8` | 72.55 | 72.90 | 72.67 | 13.78 | 2366.5 | 4652.0 | 5069 | 70°C / 154W / 2017MHz |

* **旋钮响应 (`B=1`)**：`l=0.0` 直通 `0.02 ms`（全部模块调用 0 次）；`l=0.25..0.90` 稳定在 `145.15–145.89 ms`。
* **多进程冷启动 (`B=1` × 3 次中位数)**：模型加载 `2634.4 ms` + 首帧推理 `3888.8 ms`，总计 **`6840.6 ms`**。

### 2.2 `E→P` 完整双核管线（`s=1.00`, `7`×4B + `1`×E-Core = 29 Flow NFE）

| 口径 / Batch | GPU P50 (ms/img) | GPU P95 (ms/img) | Wall P50 (ms/img) | 吞吐 (img/s) | Peak Alloc (MB) | Peak Reserved (MB) | nvidia-smi (MB) | 温度 / 功耗 / 频率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Host E2E `B=1` | 372.73 | 378.50 | 373.54 | 2.68 | 858.1 | 1030.0 | 1465 | 65°C / 135W / 2055MHz |
| Host E2E `B=2` | 346.90 | 351.04 | 347.36 | 2.88 | 1249.7 | 1578.0 | 2013 | 68°C / 148W / 2040MHz |
| Host E2E `B=4` | 322.10 | 323.85 | 322.34 | 3.10 | 2033.3 | 2650.0 | 3085 | 71°C / 156W / 2040MHz |
| Host E2E `B=8` | 310.10 | 310.40 | 310.23 | 3.22 | 3599.1 | 4840.0 | 5275 | 72°C / 158W / 2025MHz |
| Host E2E `B=12` | 306.22 | 306.34 | 306.31 | 3.27 | 5165.8 | 7014.0 | 7445 | 72°C / 157W / 2010MHz |
| Host E2E `B=16` | **304.90** | 305.24 | 304.96 | **3.28** | 6731.6 | 9188.0 | 9619 | 72°C / 155W / 2032MHz |
| Hot Cached `B=1` | 287.41 | 289.95 | 288.31 | 3.48 | 766.4 | 1030.0 | 1465 | 69°C / 140W / 2040MHz |
| Hot Cached `B=4` | 250.89 | 252.68 | 251.14 | 3.99 | 1505.8 | 2638.0 | 3073 | 72°C / 155W / 2032MHz |
| Hot Cached `B=8` | 239.95 | 240.46 | 240.10 | 4.17 | 2545.2 | 4804.0 | 5235 | 72°C / 159W / 2032MHz |

* **多进程冷启动 (`B=1` × 3 次中位数)**：模型加载 `2529.8 ms` + 首帧推理 `3181.2 ms`，总计 **`5933.4 ms`**。

---

## 3. 分阶段拆解与 `torch.profiler` 热点定位

### 3.1 分阶段 GPU 事件耗时 (`ms/img P50`)

| 阶段 | `B=1` (ms/img) | 占比 | `B=4` (ms/img) | `B=4` 批总耗时 (ms) | 核心特征与瓶颈判断 |
| --- | --- | --- | --- | --- | --- |
| `h2d` (Pinned uint8 → GPU) | 0.27 | 0.1% | 0.16 | 0.63 | 可忽略 |
| **`a0_encode`** | **71.38** | **18.1%** | **66.03** | 264.12 | 512px Conv+GroupNorm+Attention 带宽饱和，几乎不随 Batch 摊薄 |
| `cnx_struct_stiefel_y16` | **18.79** | 4.8% | 6.27 | 25.07 | `B=1` 存在 ~12ms 固定同步开销（`torch.linalg.svd` cuSOLVER sync + 重复灰度计算） |
| `ecore_a3` | **25.24** | 6.4% | 6.79 | 27.15 | `B=1` kernel launch + `nn.MultiheadAttention` 开销极大（`B=4` 降至 6.79ms） |
| **`pcore_trajectory` (7×4B)** | **230.99** | **58.5%** | **178.67** | 714.68 | 最大耗时项；每步重复算 `dino_proj`/`SPADE`/`cond_mlp` 静态列 + 7 次步间 sync |
| **`s10b_decode`** | **47.47** | **12.0%** | **68.68** | 274.72 | **`B=4` 反而显著劣化**（`47.5→68.7 ms/img`）：512px NCHW↔NHWC 拷贝 + 未融合 GroupNorm/SiLU |
| `d2h` (GPU → Pinned uint8) | 0.33 | 0.1% | 0.18 | 0.73 | 可忽略 |

### 3.2 `torch.profiler` (CPU + CUDA) 算子瓶颈证据

从 `p0_baseline.json` 中采集的 `torch.profiler` 前排算子显示：
1. **严重内存拷贝与 Layout 抖动**：
   - `E-only B=4` 中，`aten::copy_` 高达 **`109.69 ms`**（`9.6%` 总 CUDA 时间，`103` 次），与未名元素级 kernel (`103.95 ms`)、`aten::native_group_norm` (`48.19 ms`)、`nchwToNhwcKernel` 并列占据 512px `A0`/`S10B` 绝大部分带宽。
2. **海量小 kernel 与高 CPU 分发税**：
   - `E→P B=1` 单次推理包含 **`1,394` 次 `aten::copy_`**、`443` 次 `nchwToNhwcKernel`、`542` 次 `aten::add`、`392` 次 `aten::add_`、`228` 次 `aten::native_group_norm`、`187` 次 `aten::silu`。
3. **计算/架构冗余**：
   - `P-Core` 在 7 步 4B 调度里重复了 7 遍与时间 $\tau$ 无关的 `dino_proj(q32)`、`mid1.spade`/`mid2.spade`（其中 `s0/00` 两支输入全零还白算卷积）、`one_hot`/`normalize(u_style)`；
   - `E-Core` 与 `P-Core` 瓶颈层使用 `nn.MultiheadAttention`（触发 `_native_multi_head_attention`）；
   - `Stiefel` 分解使用 `torch.linalg.svd`，阻止了计算图捕获。

### 3.3 据此制定的纯 PyTorch Infra 优化路线

基于上述 P0 实证，确定后续纯 Torch 优化优先级（详见 [OPTIMIZATION.md](OPTIMIZATION.md)）：
1. **消除同步与图外冗余**：Host 预计算 ODE 调度表去除步内 `.item()` 同步；`A0` 裁掉无用的 `logvar` 通道；共享 `rgb_to_gray` 与预注册算子卷积核；用 10 步纯 `bmm` **Newton-Schulz** 替代 cuSOLVER `svd`。
2. **P-Core 轨迹级代数外提与注意力重构**：把 `gamma` 混合、`dino_proj`、`mid1/mid2.spade`（2B 配对后索引广播到 4B）和 `cond_mlp[0]` 静态列全部外提出 7 步循环；将 `BottleneckAttention` 替换为 `F.scaled_dot_product_attention`，将 `.repeat_interleave(2)` 替换为 `F.interpolate(..., mode='nearest')`。
3. **纯 PyTorch Triton 内核融合与 CUDA Graph**：利用 `torch.compile(mode="reduce-overhead", dynamic=False)`（带 `TORCHINDUCTOR_FX_GRAPH_CACHE=1` 持久化与 `cudagraph_mark_step_begin()` 安全封装），以及仅在编译下开启 `S10B channels_last`（避开 PyTorch Eager GroupNorm 的 NHWC 劣化陷阱）。
