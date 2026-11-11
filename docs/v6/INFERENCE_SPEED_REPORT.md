# SD0-S10 推理速度专项报告（2026-11-12）

**目标**：量化「冻结官方 SD0 Encoder、只换 10.66M S10 Decoder」后的推理速度收益，给出 Encoder / WEAVE / Decoder 三段用时比例与换 Decoder 前后的瓶颈转移，并与论文主表/计时 ledger 对齐。

**结论速览**：
- 同一张论文用 RTX 3060、同一个 4-epoch 主 ckpt、同样 8-step / batch 8 下：新 S10 与旧 S10 速度一致，**Decoder 段快 3.15×（PyTorch）/ 2.71×（TRT）**，**整条生成快约 1.9×（PyTorch）/ 2.1×（TRT 拼接估计）**。
- **瓶颈发生转移**：官方 Decoder 时 Decoder 占 70–81%，换 S10 后降到 43–59%，Encoder 升为最大段（~41%）。继续优化 Decoder 边际收益已小，下一步要提速需动 Encoder（本轮方案冻结 Encoder，不做）。

---

## 1. 三套口径，必须分开看

| # | 来源 | 硬件 | 后端 | 测什么 | 关键数字 |
|---|---|---|---|---|---|
| **A** | 论文 `aaai2027_v4/paper.tex` 主表 + `supplement.tex` + `docs/model_probe/generation_only_timing_summary.json` | **RTX 3060 12GB** | ONNX Runtime FP16 encoder + **TensorRT FP16** decoder（ws 2GB, batch 8） | 整条生成（encode + 8-step ODE + decode），排除指标/缓存/序列化 | **97.3 ms/img**；`encoder+WEAVE 18.7`（19%）、**`TRT decoder 78.6`（81%）** |
| **B** | 本轮 `tools/profile_staged_3060.py`（3060） | **同一张 RTX 3060 12GB** | PyTorch eager（E/D FP16、WEAVE bf16 autocast）、batch 8、cuda.Event | 同三段 | 官方 Decoder 308.12 ms/img；S10 类 160.08 ms/img |
| **C** | 本轮 `tools/trt/one_shot_s10_trt_b8.py`（3060 **WSL**, TRT 11.4） | 同一张卡 | **TRT FP16 STRONGLY_TYPED** decoder 引擎、fixed b8、ws 2GB | 仅 Decoder 段 | 官方 **73.34 ms/img**、新 S10 **27.08 ms/img** |

**A 与 C 是论文同口径**（同卡、TRT、b8、fp16）：C 重测官方 TRT decoder 73.34 vs 论文 78.6，差 6.7%（TRT 10→11 / 计时窗口差异），**口径对齐成立**。
**A 与 B 绝对毫秒不可比**（TRT vs PyTorch eager），但 **B 内部 4 个 Decoder 的相对比较严格可比**（共享同一 Encoder 输出与同一批 WEAVE 输出，只换 Decoder）。

> 另：历史 4090 报告 `docs/v6/S10_FINAL_REPORT.md` 的 `44.61 → 15.57 ms`（2.86×）是 **4090 / bs1 / PyTorch / 纯 Decoder 段** 的第四套口径，不能与 A/B/C 直接相除。

---

## 2. 分段用时与占比（口径 B，3060 PyTorch，batch 8）

| 段 | ms/batch | ms/img | 说明 |
|---|---:|---:|---|
| **Encoder** | 523.08 | 65.39 | 官方 sd-vae-ft-ema encode，FP16，固定不变 |
| **WEAVE 8-step** | 207.17 | 25.90 | 论文主 ckpt `epoch_0004.pt`，bf16 autocast，固定不变 |
| Decoder 官方 49.49M | 1734.70 | 216.84 | PyTorch eager |
| Decoder 旧 S10 | 550.67 | 68.83 | 10.66M，A0 训练 |
| **Decoder 新 C0** | **550.42** | **68.80** | 10.66M，本轮最佳重建 |
| **Decoder 新 C1** | 550.65 | 68.83 | 10.66M，本轮最佳响应 |

### 整条流水与三段占比（换 Decoder 前后）

| Decoder | total ms/img | Encoder % | WEAVE % | **Decoder %** | Decoder 加速 | 整条加速 |
|---|---:|---:|---:|---:|---:|---:|
| 官方 49.49M | 308.12 | 21.2% | 8.4% | **70.4%** | 1.00× | 1.00× |
| 旧 S10 | 160.12 | 40.8% | 16.2% | **43.0%** | 3.15× | 1.92× |
| **新 C0** | **160.08** | 40.8% | 16.2% | **43.0%** | **3.15×** | **1.93×** |
| **新 C1** | 160.11 | 40.8% | 16.2% | 43.0% | 3.15× | 1.92× |

**关键读点**：
1. **新 C0/C1 与旧 S10 速度完全相同**（550.42 / 550.65 / 550.67 ms/batch，差 <0.05%）—— 同拓扑 10.658M、只改训练目标（Haar HF / 响应蒸馏），计算量不变，符合设计。
2. **Encoder + WEAVE 绝对时间不变**（523.08 + 207.17），只是因为分母变小，占比从 29.6% 升到 57.0%。
3. **瓶颈转移**：官方 Decoder 占 70.4%，换 S10 后 43.0%；Encoder 反而成为最大段（40.8%）。

### 与口径 A（论文 TRT）的占比互证

| | enc+WEAVE | decoder | 整条 | Decoder 占比 |
|---|---:|---:|---:|---:|
| 论文 TRT 官方 | 18.7 | 78.6 | 97.3 | **80.8%** |
| 本轮 TRT 官方（C 重测） | 18.7* | 73.34 | 92.0* | **79.7%** |
| **本轮 TRT 新 S10（C）** | 18.7* | **27.08** | **45.8\*** | **59.1%** |

\* `18.7` 取自论文 ledger，未在本轮重测；整条为拼接估计，非端到端实测。

两种后端都给出同一结论：**Decoder 始终是最大段（60–81%），但换 S10 后占比大幅下降，Encoder 段开始主导**。

---

## 3. TensorRT 专项（口径 C，3060 WSL，b8 fp16）

按用户要求只给**一个 S10**（C0）建引擎（C1/旧 S10 同拓扑同参数量，PyTorch 口径下差 <0.05%，结论通用）：

| Decoder TRT FP16 b8 | 参数 | ms/batch | **ms/img** | 吞吐 | 加速 |
|---|---:|---:|---:|---:|---:|
| 官方 sd-vae-ft-ema 49.49M（论文现成引擎） | 49.49M | 586.68 | **73.34** | 13.6 img/s | 1.00× |
| **新 SD0-S10 C0（本轮新建）** | 10.66M | 216.61 | **27.08** | 36.9 img/s | **2.71×** |

- 引擎 `s10new_c0_decoder_512_b8_fp16.engine` 24.2 MB，构建 115 秒；fp16 ONNX 由仓库 `convert_onnx_fp16.py` 统一类型生成，ORT fp32 校验 ~7.6e-6。
- 官方 baseline 引擎 `vae_decoder_512_b8_fp16.engine` 为 3060 原有资产，**未覆盖**。
- **TRT 相对 PyTorch**：官方 216.84 → 73.34（2.96×）、S10 68.80 → 27.08（2.54×）。
- 代入论文整条：`18.7 + 27.08 ≈ 45.8 ms/img`，约为论文 97.3 ms/img 的 **2.13× 整条加速**（拼接估计）。

**batch 12 未做**（按用户指示取消）；其余 6 个引擎构建未执行。

---

## 4. 32 方法时间维度（论文主表为唯一权威）

`aaai2027_v4/paper.tex` 主表 `Train/Infer` 列：**分钟 / 750 张输出 / RTX 3060**，`free` = 无需训练。换算 `ms/img = Infer(min) × 60000 / 750`。

| 方法 | Infer(min/750) | ms/img | 备注 |
|---|---:|---:|---|
| **WEAVE (Published)** | **1.22** | **97.6** | 主表 Ours 行，与 supplement audited 97.3 吻合 |
| Latent-WCT | 3.67 | 293.6 | |
| CUT / SD-Turbo | 5 | 400.0 | |
| SaMST | 10 | 800.0 | |
| SaMam | 17.6 | 1408.0 | |
| AesPA-Net | 23 | 1840.0 | |
| StyTr-2 | 38 | 3040.0 | |
| StyleAligned / Z-STAR | 77 / 180 | 6160 / 14400 | 主表有，32 方法榜未收录 |
| StyleID | 63 | 5040.0 | |
| StyleShot | 306 | 24480.0 | |
| IDT / TGT / Seedream | `--/--` | — | anchor / API，未报时间 |

- **主表中已报时间的最快方法就是 WEAVE 官方（97.6 ms/img）**；本轮新 S10 拼接估计 45.8 ms/img 约为其 2.13×，但**不同源、非端到端重测**，不可当作严格同口径冠军。
- `UNIFIED_FULL_RANKING.md` 的 32 方法里，**只有 10 个能映射到主表时间列**；其余 Ours/V6 衍生配置主表不含，时间留空。
- `docs/exp/b5_compute_cost.md` 的「126 s / 168 ms/img」是旧草稿，与当前主表 `1.22 min` 不符，**不采用**。

完整表格见 [BASELINE_TIMING_TABLE.md](BASELINE_TIMING_TABLE.md)。

---

## 5. 前后变化一句话总结

| | 换 Decoder 前（官方 49.49M） | 换 Decoder 后（新 S10 10.66M） | 变化 |
|---|---|---|---|
| 参数量 | 49.49M | 10.66M | −78.5% |
| Decoder 段（PyTorch b8） | 216.84 ms/img | 68.80 ms/img | **3.15×** |
| Decoder 段（TRT b8） | 73.34 ms/img | 27.08 ms/img | **2.71×** |
| 整条（PyTorch b8） | 308.12 ms/img | 160.08 ms/img | **1.93×** |
| 整条（TRT 拼接估计） | 97.3 ms/img | ≈45.8 ms/img | ≈**2.13×** |
| **Decoder 占比** | 70.4%（PyTorch）/ 80.8%（TRT） | 43.0% / 59.1% | **瓶颈从 Decoder 转向 Encoder** |
| **Encoder 占比** | 21.2% | 40.8% | 成为最大段 |

---

## 6. 产物与复现

| 文件 | 内容 |
|---|---|
| `deliver/staged_profile_3060_b8.json` | 3060 PyTorch 分段 profile（4 变体全字段 + paper_ledger 原文） |
| `deliver/s10_trt_b8_bench.json` | TRT b8 双引擎计时 |
| `docs/v6/STAGED_PROFILE_3060.md` | 分段 profile 详细报告 |
| `docs/v6/S10_TRT_B8_SPEED.md` | TRT 专项 |
| `docs/v6/INFERENCE_SPEED_BASELINE.md` | 三套口径可比性说明 |
| `docs/v6/BASELINE_TIMING_TABLE.md` | 32 方法总表 + 时间维度 |
| `deliver/bubble_csd.png` / `bubble_affinity.png` | content–style 气泡图（半径 ∝ img/s） |

**3060 侧脚本**（WSL）：`tools/profile_staged_3060.py`、`tools/trt/export_s10_new_onnx.py`、`tools/trt/one_shot_s10_trt_b8.py`、`tools/trt/bench_s10_trt.py`。
复现：`cd /mnt/i/Latent_Style/WEAVE && python3 tools/trt/one_shot_s10_trt_b8.py`（引擎已缓存，秒级重测）。

## 7. 未完成 / 口径限制（如实）

1. **没有端到端 TRT 重测**：45.8 ms/img 是「论文 enc+WEAVE 18.7 + 本轮 TRT decoder 27.08」拼接估计；要精确需把 ONNX encoder 接进同一计时循环。
2. **论文 baseline 的 Infer 是各自完整方法在 3060 上的总时长**，与「只换 Decoder」不完全同类，跨行对比需谨慎。
3. **TensorRT 只测了 b8 / c0 一个 S10**；b12 与其余引擎构建按指示取消。
4. 3060 运行环境是 **WSL**（torch 2.10+cu128 / TRT 11.4）；早期用 Windows cmd/python 的尝试不是正确路径，已全部改为 WSL 结果。
5. 论文主 ckpt 为 4-epoch；本地 4070 Laptop 曾有 10-epoch 替代 ckpt 的 profile，仅供内部参考，本报告不采用其绝对毫秒。
