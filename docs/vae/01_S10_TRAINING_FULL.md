# S10 Decoder 全流程训练过程

> 本文件是 S10 从「剪枝」到「可用」的**完整训练过程**权威记录，供复现与后续审计。
> 执行区间：2026-11-09 22:37 → 2026-11-10 02:10（原始 S10），
> 2026-11-11 13:15 → 11-12 00:20（SD0-S10 V2 二次训练）。
> 硬件：单卡 RTX 4090 24GB。

---

## 0. 一句话总览

把官方/A0 的 **49.490M** VAE Decoder 结构化剪枝到 **10.658M**（−78.5%），用**纯真实图像 `L1 + LPIPS@512`**（不用任何教师蒸馏、不用 CSD/HF Loss）在干净 78,661 张 WikiArt 上恢复重建质量，得到 `s10_p5_lowlr/decoder.pt`：LPIPS@512 0.0926、CSD Keep 0.8535、forward **15.57 ms（2.86× 加速）**。

后来在**原版 SD0 Encoder**下又完整重训了一版（SD0-S10 V2，含 Haar HF 与响应蒸馏），但 2026-11-12 的适配实验证明：**旧 S10 其实不需要重训，直接喂 SD0 latent 就可用且更好**。两条线的关系见 [09_S10B_DECISION.md](09_S10B_DECISION.md)。

---

## 1. 模型与接口（必须先固定，否则训练全错）

### 1.1 结构

Diffusers `AutoencoderKL` 的 Decoder 有 **4 条均匀通道流**（`decoder_prune.py` 实测）：

```
forward order block_out_channels = [c0, c1, c2, c3]，Decoder 内部反转：
  stream A = mid_block + up_blocks[0]   A0: 512
  stream B = up_blocks[1]               A0: 512
  stream C = up_blocks[2]               A0: 256
  stream D = up_blocks[3]（512px 输出） A0: 128
```

S10 目标宽度（forward order）：**`[64, 128, 192, 256]`**
→ 各流 `A=256 / B=192 / C=128 / D=64`，参数 **10,657,923**（A0 49,490,000 的 21.5%）。

### 1.2 剪枝方法（`decoder_build_s10.py` + `decoder_prune.py`）

1. **Taylor 重要性**：在 128 张校准图上，用 `Charbonnier + 0.5·多尺度 Haar 高频` 反传，按输出通道累加 `|Σ(W·dW)|`，得到每条流的重要性向量
2. **每流选一组索引**：`topk(imp, k)` 后排序；**一条流一个索引集**，保证残差捷径、GroupNorm、attention 一致
3. **权重继承**：按选中索引从 A0 的 conv/norm/linear 复制对应输入/输出通道，bias 同理
4. **零步 sanity**：用 A0 原宽度重建必须与 A0 输出 bit-exact；剪枝后未训练 L1 ≈ **0.5568**（极差，需恢复）

> 关键：**不能随意截前 K 个通道**。Cold-SD0 实验曾误把 A0 的索引数字套到官方 SD0 权重上，得到 LPIPS 0.808；正确做法是在目标 Decoder 上**重新计算 Taylor 重要性**。

### 1.3 `post_quant_conv` 接口（踩过最贵的坑）

```
AutoencoderKL.decode(z) = decoder( post_quant_conv(z) )
```

- `post_quant_conv` 是 **4→4 的 1×1 卷积**，属于 Decoder 接口，**随 Decoder 训练**
- `StudentDecoder(pqc, decoder)` 包装器内部**恰好应用一次 pqc**
- **禁止** `decoder(z)`（跳过 pqc）或 `vae.decode(pqc(z))`（双重应用）
- 错误路径实测：A0 自己 PSNR **5.34 / SSIM 0.013**（正确 29.05 / 0.812）

---

## 2. 数据：干净训练池的确立

原始 manifest `102,255` 条。`RawMemmapWikiArtDataset(is_training=True)` 早期**只按 `raw_index not in val_indices` 过滤，忽略 `in_train` 标志**：

```
total records 102,255 | in_train 78,661 | in_train=False 23,594
dataset(is_training=True) = 100,207  → 泄漏 21,546 张测试图
已有 12,991 缓存中 2,640 张（20.3%）是泄漏测试图
```

**影响判定**：评测用的 2,048 holdout 全部来自 `in_train=False` 且被 `val_indices` 排除，训练从未见过 → 已报告指标仍是在真未见数据上测的；但训练池不干净。

修正后（P1 起）：**显式 `in_train=True and raw_index not in val_indices` → 78,661**，缓存 `runs/vae_decoder_prune/full/latent_train_fp16.bin`（2.40 GB，34.1 min）：

```
[Split] 无重复、无 val 重叠
[Shards] problem shards: 0（无全零行、无 NaN/Inf）
[Holdout] 2,048 latents, matches val_indices
[Verify] re-encode cos(cached, fresh): min=0.99998 mean=0.99999
```

原 12,991 缓存与 split.json **未覆盖**。

---

## 3. 原始 S10 训练全流程（2026-11-09/10）

**联合目标**：`L1 ≤ 0.075`、`LPIPS@512 ≤ 0.10`、`CSD Keep ≥ 0.90`。
**Loss**：`L1 + 0.5·LPIPS@512`（**无教师蒸馏、无 CSD、无 HF Loss**）。
**优化器**：AdamW；有效 batch 64；512px；BF16；`torch.compile`；平台早停 `|ΔLPIPS|<0.002 且 ΔCSD<0.005`。

### 3.1 完整轨迹

| 阶段 | 初始化 | 数据 | 步数 | lr | L1 | LPIPS@512 | CSD Keep |
|---|---|---|---:|---:|---:|---:|---:|
| 剪枝零步 | Taylor 剪枝继承 | — | 0 | — | 0.5568 | — | — |
| **P1 L1 容量恢复** | 剪枝权重 | 13k（含泄漏） | 1000 | 1e-4 | 0.06586 | 0.390 | 0.608 |
| **P0 感知精修** | `s10_p1_l1` | 13k | 600 | 2e-5 | 0.07202 | 0.1099 | 0.8484 |
| **P2 全量精修** | `s10_p3_lpips512` | **78,661 干净** | 1400（原计划 8000，平台早停） | 3e-5 | 0.06859 | 0.0862 | 0.8710 |
| **P3 低 LR 续训** | `s10_p4_full` | 78,661 | 400（原计划 1200） | 1e-5 | 0.06829 | 0.0851 | 0.8713 |
| 最终 `s10_p5_lowlr` | — | — | — | — | 0.06727 | **0.0926**（测试集） | **0.8535**（测试集） |

（P2/P3 表内为 holdout 2,048；最终行为独立测试集 256 张，两者口径不同属正常。）

### 3.2 三个关键决策与验证

1. **跳过教师蒸馏是对的**：纯 `L1+LPIPS` 就把 CSD 从 0.608 自动拉到 0.871 —— **LPIPS 恢复后 CSD 会跟随，不需要专门风格 Loss**
2. **256px LPIPS 无效**：500 步只让 512px holdout LPIPS 0.3026→0.2888（−4.6%），训练侧 256px 已到 0.05；**必须用 512px**，同 500 步做到 0.390→0.110
3. **平台早停生效**：P0 @600、P2 @1400 触发，未耗尽预算
4. **L1 与 LPIPS 是 Pareto 冲突**：L1 0.0659→0.0686（+4.1%）换 LPIPS 0.390→0.086（−78%），验收线按方案放宽到 L1≤0.075

### 3.3 CSD 平台是稳健的（两次独立运行）

| 运行 | 配置 | 停止 | holdout L1 | holdout LPIPS | holdout CSD |
|---|---|---|---:|---:|---:|
| P2 | lr 3e-5 / 8000 步 cosine | 平台早停 @1400 | 0.06859 | 0.0862 | 0.8710 |
| P3 低 LR | lr 1e-5 / 1200 步 | 平台早停 @400 | 0.06829 | 0.0851 | 0.8713 |

高 LR 与低 LR 都停在 CSD≈0.871(holdout)/0.8535(测试集)，LPIPS 仍可微降但 CSD 不动 → **不是学习率未衰减造成的**。

### 3.4 三个 bug（全部修正，影响过早期训练）

| # | bug | 现象 | 修复 |
|---|---|---|---|
| 1 | `post_quant_conv` 被跳过 | 错误路径 A0 自己 PSNR 5.34 | 加 `StudentDecoder`，训练/评测/校准三处统一 |
| 1b | 双重应用 pqc | `vae.decode(pqc(z))`、对 `StudentDecoder` 再手动 pqc | P3 / L0 脚本修正 |
| 2 | 训练/测试泄漏 | 100,207 vs 78,661，缓存 20.3% 污染 | 显式 `in_train` 过滤，重建全量缓存 |
| 3 | 变量遮蔽 | 平台早停块 `dl = abs(...)` 覆盖 DataLoader，epoch 边界 `StopIteration` 崩溃且只在阶段末存盘 → 1200 步权重全丢 | 改名 `dlp`，加每 250 步 `decoder_partial.pt` |

另：checkpoint 选择守卫原为 `L1 ≤ L_prune×1.05`，把「L1 略升但 LPIPS 减半」的最优模型全拒掉（`best_sel=inf` 不存盘）；改为 `L1 ≤ max(L_prune×1.15, 0.075)` 并保证无通过时仍存最终权重。

### 3.5 最终指标（独立 256 张测试集）

| | A0 49.490M | **S10 10.658M** | 目标 | 达成 |
|---|---:|---:|---|---|
| L1 ↓ | 0.06456 | 0.06727 | ≤0.075 | ✅ |
| PSNR ↑ | 27.21 | 26.83 | 不明显退化 | ✅ |
| SSIM ↑ | 0.7660 | 0.7564 | 不明显退化 | ✅ |
| LPIPS@512 ↓ | 0.0784 | 0.0926 | ≤0.10 | ✅ |
| CSD Keep ↑ | 0.9155 | **0.8535** | ≥0.90 | ❌ 差 0.047 |
| HF corr L0 | 0.4504 | 0.4279 | 无新退化 | ✅ |

**联合目标 2/3 达成。CSD shortfall 被接受**（见 `docs/v6/S10_FINAL_REPORT.md`）。

### 3.6 效率（4090, bs=1, 512px, PyTorch）

| | forward | forward+input-grad | 峰值显存 |
|---|---:|---:|---:|
| A0 | 44.61 ms | 120.51 ms | 11.29 / 6.70 GB |
| **S10** | **15.57 ms** | **39.90 ms** | 6.36 / 4.13 GB |
| **加速** | **2.86×** | **3.02×** | 1.78× / 1.62× |

### 3.7 梯度代理限制

```
A0:  ||g||=0.003575  LPIPS=0.06044
S10: ||g||=0.002431  LPIPS=0.08177
cos(g_A0, g_S10) = +0.1946
```

方向未反转但仅 **19% 对齐** → **若需要通过 decoder 做图像域训练，S10 不能直接替代 A0 作梯度代理**。这个数后来常被误引为「Jacobian 不匹配」，实际只是 A0/S10 的 LPIPS 输入梯度，见 [07_INFERENCE_SPEED.md](07_INFERENCE_SPEED.md)。

---

## 4. SD0-S10 V2 二次训练（2026-11-11/12）

**动机**：官方 `sd-vae-ft-ema`（SD0）Encoder 才是生成流水线真正用的前端；旧 S10 是 A0 配对的。方案要求完全冻结官方 Encoder、在未缩放原生 4×64×64 latent 上训练 10.658M S10。

### 4.1 数据（重新编码，不复用 A0 缓存）

用官方 SD0 Encoder 重新编码：`78,661` train / `2,048` holdout / `256` test256，`latent_train_fp16.bin` 2.40 GiB。校验：ID 三组不重叠、全扫 `zero=0 nonfinite=0`、128 张重编码 `cos min 0.999950`、Encoder 确定性 `0.0`、双 sha256 记录。**不能用 RMS 缩放把 A0 缓存转成 SD0**（实测 `cos(A0,SD0)=0.992~0.996` 相近但不同）。

### 4.2 三初始候选与 P1 Gate

| 候选 | 构造 |
|---|---|
| W | Warm-A0：`s10_p5_lowlr` + A0 pqc |
| C | Cold-SD0：在**官方 SD0 Decoder 上重算 Taylor 重要性**再剪枝继承 |
| W-PQC | W 权重 + SD0 pqc（仅推理比较，LPIPS 差 0.00008 → 删除） |

各 800 步同配置：W LPIPS 0.09060 / CSD 0.85726；C LPIPS 0.12819 / CSD 0.79743 → **Gate 选 W**（Cold 等成本下仍落后 +0.038）。

### 4.3 C0：高保真重建（3000 步）

`Loss = Charbonnier + 0.5·LPIPS@512 + λ_h·两级 Haar DWT(LH/HL/HH) 逐系数 L1`
- 子带尺度 `a_lb` 从训练 GT **固定估计**（不由模型输出动态改，防 gameable）
- λ_h 经一次梯度标定到重建梯度的 ~15%（实测 `hf_grad_ratio≈11–13 → λ_h≈0.011–0.014`）
- 弱能量正则 λ_e 初始 0（P0 已确认 HF 能量比 0.694 有系统性衰减，但保持保守）
- **CSD/DINO/CLIP 不进 Loss**

结果：LPIPS **0.08588**、HF corr 0.47810、PSNR 25.42、CSD 0.85313。**C0 HF Gate 通过**（LPIPS −0.00476、HF corr +0.01751）。

### 4.4 C1：响应蒸馏（2500 步）

`Response = SmoothL1((Δy_S − Δy_T)/(a_δ+ε))`，扰动三类（多尺度相关高斯 50% / 真实 latent 小插值 30% / 后验式 20%）、两档幅度（小 0.02/0.05 近似 Jacobian；中 0.10/0.20 是 Secant，不混称完整一阶同构）、弱 off-manifold 教师一致性、25% 更新进双 latent 分支；λr≈0.22、λo=0.075。Teacher = 官方 49.49M eval+detach，且**必须用 `StudentDecoder(pqc, official.decoder)`**（曾修过 teacher 吃裸 latent 的 bug）。

结果：LPIPS **0.08504**、`resp_cos 0.6495→0.6710`、PSNR 25.49。**C1 响应 Gate 通过**（真实训练 0 次 OOM / 0 次二次 backward）。

### 4.5 两个 Gate 与四层验收

P5 用 `paper_rgb_v1` 同协议重算：official LPIPS 0.07836 / old_s10 0.09340 / c0 0.08588 / c1 0.08504；CSD Keep 0.961 / 0.876 / 0.853 / 0.848。新 C0/C1 相对旧 S10 LPIPS −0.008、PSNR +0.60、HF 相关 +0.062，但 **CSD Keep 与 HF 能量比仍明显低于官方**。

### 4.6 链异常与自动降级

链内 step-9 final eval 因 teacher(49.49M) 在 bs=8 带 autograd 做 LPIPS 输入梯度 **OOM（21.57GB）** 退出。`guard_p5.sh` 按降级路径写 `NEGATIVE_REPORT.json`、仍拉起保守 P5、权重零丢失；随后手动修复（input-grad 段 8→2 图、显式释放）全量重跑通过。

---

## 5. 最终结论（2026-11-12 适配实验推翻了「必须重训」的前提）

A/B/C/D 对照证明：**旧 S10 直接喂未缩放 SD0 latent 就可用且更好**（B: P0 DINO-C 0.9678、P2 Affinity 0.5616，高于 A0 配对 A 的 0.5587，也高于本轮新 C0/C1 的 0.5575/0.5576）。闭式 PQC 校准 C 反而显著更差。

因此：
- **旧 S10 = S10B，字节级相同**，零重训、零 A0 运行时依赖
- SD0-S10 V2 的 C0/C1 仍是有效的研究产物（更好的重建 + 响应一致性），但从「下游 Q2a+Oracle」视角不是必要路线
- 详细证据见 [09_S10B_DECISION.md](09_S10B_DECISION.md) 与 `docs/v6/S10_SD0_ADAPT_REPORT.md`

---

## 6. 复现入口

| 步骤 | 代码 |
|---|---|
| 结构化剪枝 | `tools/vae_suite/decoder_build_s10.py`、`decoder_prune.py` |
| 干净 A0 缓存 | `tools/vae_suite/decoder_cache_full.py` |
| 原始 S10 训练 | `tools/vae_suite/decoder_train.py` |
| SD0 缓存 | `tools/vae_suite/sd0_cache.py` |
| Cold 剪枝 | `tools/vae_suite/build_cold_sd0.py` |
| C0/C1 训练 | `tools/vae_suite/train_sd0_s10.py`、`sd0_losses.py` |
| 串行总控 | `tools/vae_suite/run_sd0_s10.sh`、`run_p5_final.sh`、`guard_p5.sh` |
| 评测 | `tools/vae_suite/eval_sd0_s10.py`、`eval_generator_swap_full.py`、`score_baseline_protocol.py` |
| S10B 导出 | `tools/vae_suite/export_s10b.py` |

原始报告：`docs/v6/S10_FINAL_REPORT.md`、`docs/v6/S10_12H_EXECUTION_LOG.md`、`docs/v6/SD0_S10_V2_EXECUTION.md`、`docs/v6/SD0_S10_V2_FINAL.md`。
