# CLEAVE V35 — 纯 PyTorch 优化消融、内核融合与方案演进

> **核心原则**：不引入第三方外部推理引擎，纯依托 PyTorch 2.11 + Triton 3.7.1 在固定输入形状下实施算子/代码简化、代数不变量外提、注意力重写与编译图融合；每项改造均在 16 分层样本下先通过数值一致性校验，再合入下一步。

---

## 1. PyTorch 级联消融实测总表（RTX 3060, Host E2E RGB→RGB P50）

全部消融均在同一 RTX 3060 环境下使用相同 16 样本分层集对比（数据来自 `p1_sweep.json`、`p1_extra.json`、`p2_torch_ablation.json`）：

| 方案阶段 | 核心改动 | `E-only B=1` (ms/img) | `E-only B=4` (ms/img) | `E→P B=1` (ms/img) | `E→P B=4` (ms/img) | 16 样本 `mae_255` (`E` / `P s=1.00`) | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **P0 `Reference`** | 原始基线 (`AutoencoderKL`, SVD Stiefel, MHA, 7×4B) | 144.91 (`1.00×`) | 144.41 (`1.00×`) | 372.73 (`1.00×`) | 322.10 (`1.00×`) | 0.00 / 0.00 | 基准 |
| **P1a `host_grid`** | ODE 网格 Host 预计算消除步间 `.item()` 同步 + 跳过分布封装 | 144.37 (`1.00×`) | 143.85 (`1.00×`) | 345.53 (`1.04×`) | 314.81 (`1.02×`) | 0.00 / 0.08 | **采纳**（零损耗消同步） |
| **P1b `channels_last_all`** (Eager) | 全模型 Eager `channels_last`（含 `A0`/`S10B`） | **236.47 (`0.61×`)** | — | **439.96 (`0.82×`)** | **390.21 (`0.82×`)** | 0.07 / 0.12 | **否定**（Eager GroupNorm NHWC 严重退化） |
| **P1c `nchw_fp16`** (Eager) | 全 NCHW + FP16 权重 + `nearest` 上采样 | 147.20 (`0.98×`) | — | 338.53 (`1.10×`) | 313.26 (`1.03×`) | 0.03 / 0.12 (`all`: 0.21/0.26) | `FP16` 精度无溢出风险 |
| **P2a `eager_opt`** | `FastA0`(去 logvar) + `FastCond`(NS10) + `SDPA` + `HoistedPCoreStep` | **123.11 (`1.17×`)** | 138.96 (`1.04×`) | **305.43 (`1.18×`)** | 301.94 (`1.06×`) | **0.22 / 0.27** | **采纳**（无编译即省 67ms @ B1） |
| **P2b `compiled_opt`** (**最终 `Optimized`**) | P2a + `S10B channels_last` + `torch.compile(reduce-overhead, dynamic=False)` | **111.88 (`1.29×`)** | **82.37 (`1.75×`)** | **219.38 (`1.64×–1.70×`)** | **181.18 (`1.77×–1.78×`)** | **0.22 / 0.26** | **采纳（默认生产配置）** |
| **P3a `fast_euler6`** (`Fast` 6 步) | P2b + 首步改用 Euler (`6` 批 / `24` NFE) | 111.68 (`1.29×`) | 82.16 (`1.75×`) | **204.91 (`1.76×–1.82×`)** | **167.75 (`1.91×–1.92×`)** | 0.22 / **1.46** | **可选加速档**（需单独过 Test750 门控） |
| **P3b `fast_euler5`** (`Fast-5` 5 步) | P2b + Euler 5 步 (`5` 批 / `20` NFE) | 111.86 (`1.29×`) | 82.48 (`1.75×`) | **190.36 (`1.89×–1.96×`)** | **153.55 (`2.08×–2.10×`)** | 0.22 / **1.90** | 接近 2/255 像素漂移上限，需严测质量 |

*(注：`p2_torch_ablation.json` 内相对于该轮同步现测 `Reference` 基准计算即时加速比，与 P0 全量表 `372.73 ms` 相比 `P2b B=1` 达到 `1.70×`。)*

---

## 2. 四项核心纯 PyTorch Infra 优化详解

### 2.1 P-Core 轨迹级静态不变量代数外提 (`HoistedPCoreStep`)

在原始 6 步（7 次 4B 调用 = 28 次条件评估）轨迹整合里，`PCoreSLFUNet.forward` 对不随时间 $\tau$ 变化的张量反复进行了 7 遍重算。我们在 [`exp/deploy_v35/torch_opt.py`](../../../exp/deploy_v35/torch_opt.py) 中按精确线性/分支结构将其外提出循环：
1. **SPADE 与 `dino_proj` 从 `28B` 降至 `2B`**：CFG 四分支 `(sc, s0, 0c, 00)` 中 `s0` 与 `00` 的 `layout_scale=0`、`s16` 恒为零；`sc` 与 `0c` 共享同一非零 `s16_on`。因此仅在轨迹起点对 `[s16_on, 0]`（批大小 `2B`）运行一次 `dino_proj` + `mid1.spade` + `mid2.spade`，通过索引广播至 `4B`，直接消灭了 7 步循环内 **14 次 SPADE 卷积与 14 次 `dino_proj`**。
2. **`cond_mlp[0]` 线性拆分**：将输入 `[temb(64), onehot(5), u_style(417), flags(2), gamma_emb(16)]` 的第一层权重拆分为 `W_tau[:, :64]` 与 `W_rest[:, 64:]`；轨迹开始前仅计算一次 `cond_bias4 = F.linear(static_4b, W_rest, b0)`，每步仅对 `1B` 的 `t` 计算 `F.linear(temb_1b, W_tau)` 并广播相加，精度零损失（纯 FP32 验证 `max_err=0.0010, mae=0.00013`）。
3. **单组件实测提速**：仅外提与 SDPA 即让 P-Core 7 次等价耗时从 `230.99 ms` 降至 `183.82 ms`（`1.26×`），结合 `torch.compile(reduce-overhead)` 后进一步降至 **`107.61 ms`**（`B=1`，**`2.15×`**）与 **`97.68 ms/img`**（`B=4`，**`1.83×`**）。

### 2.2 Stiefel Polar 零同步替代 + 前端融合 (`FastConditioner`)

* 原始 `stiefel_polar_quotient` 调用 `torch.linalg.svd`（cuSOLVER），在 RTX 3060 Windows WDDM 下不仅引入额外同步，还阻断计算图捕获（`1.72 ms` 单步）。
* 改用 10 步纯 `bmm` **Newton-Schulz Polar 迭代**，并把共享 `rgb_to_gray`、预注册 Sobel/Gauss/Laplacian 静态卷积核、`ConvNeXt-S3 FP16` 与 `Y16 LCN` 合并进单一 `FastConditioner` 模块。
* `16x16` 正交特征 `q16` 余弦相似度高达 **`0.999980`**（`mae=0.00061`），整个 Conditioning 模块 `B=1` 耗时从 P0 的 `18.79 ms` 降至 Eager `9.75 ms`，编译后降至 **`5.98 ms`（`3.14×`）**。

### 2.3 注意力与上采样算子重写 (`SDPA` + `F.interpolate`)

* 将 `E-Core` 与 `P-Core` 内 `BottleneckAttention` 的 `nn.MultiheadAttention` 原位无损映射至 `F.scaled_dot_product_attention`（共享完全一致的 `in_proj_weight/bias` 与 `out_proj` 权重），去除 `_native_multi_head_attention` 调度税；并将上采样 `.repeat_interleave(2)` 统一为 `F.interpolate(..., scale_factor=2.0, mode="nearest")`。
* **`E-Core`** 单核在 `B=1` 从 `25.24 ms` 降至 **`6.51 ms`（`3.88×`）**，`B=4` 降至 **`1.46 ms/img`（`4.65×`）**。

### 2.4 `channels_last` 的编译/Eager 分化陷阱与 `torch.compile` 图缓冲安全

* **Eager vs 编译下的 `S10B channels_last`**：Eager 模式下开启 `channels_last` 会引发 GroupNorm 频繁触发 `nchwToNhwcKernel` 与额外拷贝，使 `E-only` 从 `144.9 ms` 劣化至 `236.5 ms`（`0.61×`）；而**仅在 `torch.compile(mode="reduce-overhead")` 下启用 `channels_last` 后，Triton 3.7.1 成功把 `Conv + GroupNorm + SiLU` 融合为原生 NHWC 内核**，使 `S10B` 在 `B=4` 从 P0 的 `68.68 ms/img` 大幅下降至 **`31.53 ms/img`（`2.18×`）**！
* **CUDAGraph 静态输出别名安全 + 磁盘缓存**：PyTorch 2.11 `CUDAGraphTrees` 在多步复用源特征或连续调用时易发生输出槽位覆写报错。我们在 [`pipeline.py`](../../../exp/deploy_v35/pipeline.py) 中为所有固定形状编译图统一注入 `torch.compiler.cudagraph_mark_step_begin()` 与输出 `.clone()`（微秒级拷贝），并启用 `TORCHINDUCTOR_FX_GRAPH_CACHE=1` 持久化缓存。
* **`A0 Encoder` 瘦身**：裁去推理丢弃的 `quant_conv` 后 4 通道 `logvar` 分支并用 `FP16` 编译，`B=4` 从 `65.88 ms/img` 降至 **`47.90 ms/img`（`1.38×`）**，`B=4` 无明显数值溢出现象（`mae_255=0.22`）。
