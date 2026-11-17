# WEAVE / PRISM 全局实验结论与全量数据总台账（Master Ledger · V6 → V19）

**更新日期**：2026-11-18 · **统一基准**：`eval_protocol/tasks_750.json`（Cross600 主读数 + Same150 恒等控制）· **解码口径**：`S10` 剪枝解码器（10.66M，全仓唯一正式口径）· **机读数据**：[`metrics.csv`](metrics.csv)（**232 行 × 25 列**）与 [`metrics.json`](metrics.json)（**88 个非支配帕累托点**）。

---

## 一、全局核心结论与决定性突破（Executive Findings）

### 1.1 `PRISM v2 / V19` 正交商—纤维对偶与各向异性谱传输（2026-11-18 实测终局）
立足源码级病灶剖析（见 [`../next1118/01_CRITICAL_DIAGNOSIS_AND_HISTORICAL_AUDIT.md`](../next1118/01_CRITICAL_DIAGNOSIS_AND_HISTORICAL_AUDIT.md)），`V19` 通过**正交商—纤维对偶 $(\Pi_\perp, \Pi_\parallel)$**、**潜空间亮度—色度正交分裂 $\R^4 = \operatorname{span}(\hat{w}_Y)\oplus\hat{w}_Y^\perp$** 与 **穿过可微 `S10` 的无归一化 `VGG-16+Oklab` 均衡判别器**，在完全**零测试期源标签 `sid`**、**零裁判泄漏**的前提下实现了全面历史性突破：

1. **纯单步毫秒级模型（`17.0 ms/img` = `58.7 img/s` @ 4090）无条件同时碾压 `Seedream 4.5` 与 `StyTr2`**：
   - **`V19.S2_prism1_v2`（`PRISM-1`, $\lambda=1.0$, `1.50M` 活跃潜参数）**：Cross600 **CSD Top-1 = `44.33%`**（`DINO-S = 0.4431`, `CLIP-S = 0.6880`, `Affinity = 0.5655`, `DINO-C = 0.7440`, `LPIPS = 0.5057`），以单次前馈同时在 **全部五项风格与内容指标** 上严格支配商业基座 **Seedream 4.5（`37.30%` Top-1, `0.7343` DINO-C, `0.5202` LPIPS）** 与 **StyTr2（`39.50%` Top-1, `0.7023` DINO-C, `0.7386` LPIPS）**，且提速 **`117×–178×`**！
   - **反事实错标签控制（`V19.S2_prism1_wrong`）**：Top-1 瞬间跌至 **`0.83%`**（因果特异性 **`+43.50pp`**）。
   - **全盲恒等门（`V19.S2b_prism1_blind_same`）**：即便不告知模型输入目标是否与源同派，仅靠连续弦模软门控 $\tanh^2(\|e\|_2/\sigma)$，Same150 准确率仍高达 **`99.3%`（149/150）**。
   - **单参考图零类别表 1-Shot 模式（`V19.S3_oneshot_exemplar`）**：仅用每派 **1 张独立留出集示例画作** $e = \phi(\Pi_\parallel(x_{\mathrm{ref}})) - \phi(\Pi_\parallel(x_s))$，单步 Cross600 Top-1 进一步达 **`45.50%`**！
   - **同网络连续强度旋钮（$\lambda = 1.30$）**：零重训直接推至 **`50.17%` Top-1**（`DINO-C = 0.7032` 仍高于 `StyTr2`）。
2. **彻底攻克 `Ukiyo-e` 与 `Early Renaissance` 少数派壁垒（`PRISM-Pro`）**：
   - **`V19.T4_sweet_vgg_nosid`**：Cross600 **CSD Top-1 = `59.00%`**，首次实现跨风格正值边际 **`Margin = +0.0265`**，`Style Affinity = 0.5791`（超越 `AesPA-Net` 的 `0.5784`，且 `DINO-C` 高 `+7.08` pts、`LPIPS` 低 `-10.81` pts），将 **`Ukiyo-e` 从旧潜空间 `R1` 的 `0.8%` 拉升至 `45.0%`**，**`Early Renaissance` 拉升至 `33.3%`**！

---

## 二、全仓库七条已验证方法论铁律

1. **Law 1 — Reference Retrieval $\neq$ Style Distribution Gradient（V10.1）**：内容条件近邻检索混入供体布局，恢复方向余弦呈负值（$-0.25\sim -0.53$）。
2. **Law 2 — Recovery Alignment $\neq$ Style Specificity（V10.2）**：低维小波 KDE 恢复收益中 95% 为通用“像画”先验，仅 5% 具画派特异性。
3. **Law 3 — Separation Margin $\neq$ Unconstrained Generative Direction（V11–V12 / V15 PT_C）**：无各向异性信任域约束直接刷判别器会摧毁结构；复杂内容 Query 门控与风格梯度近正交（不 binding）。
4. **Law 4 — Local DWT Token $\neq$ Semantic Style Carrier（V13–V14）**：小感受野局部小波判别力（`.544`）远低于全局自注意力 224px DINOv2 patch（`.793`）。
5. **Law 5 — Self-Paired Generative Capacity $\neq$ Cross-Style Editing（V15–V18）**：无去风格正交分裂的自配对流扩容 260×（$0.13\text{M}\to 34.2\text{M}$）使 $\tau_0=0$ 生成升至 `49.3%`，却使 $\tau_0=0.7$ 编辑退化至 `6.67%`；即便 $\tau_0=0$ 纯噪声配边缘控制流，自配对重建仍受困于二维笔触反转天花板（`18.67%`）。
6. **Law 6 — GroupNorm Latent Critics & Isotropic Trust Balls Blind Palette Transport（V18–V19）**：`GroupNorm` 架构性抹除通道均值/方差，各向同性欧氏球把三维潜色度迁移误判为结构损伤（致 `Ukiyo-e < 1%`）；一维亮度 LCN 严锁 + 三维色度放开 + 穿过 `S10` 的无归一化 `VGG+Oklab` 均衡判别器可将 `Ukiyo-e` 直接拉升至 `45.0%`。
7. **Law 7 — Orthogonal Quotient–Fiber Duality $(\Pi_\perp, \Pi_\parallel)$ Unlocks Label-Free 1-Step Amortization（V19）**：被结构商投影 $\Pi_\perp$（`18.3%` 风格探针）丢弃的互补统计量 $\Pi_\parallel$ 恰以 `97.3%` 精度编码源风格坐标，驱动免源标签通用弦与空间语义路由单步网络达 `44.3%–50.2%` Top-1。

---

## 三、帕累托前沿全景代表表（Cross600 · S10 四裁判协议）

| 方法 / 实验臂 | 家族 | 推理形态与延迟 | CSD Top-1 ↑ | Margin ↑ | DINO-S ↑ | Affinity ↑ | DINO-C ↑ | LPIPS ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **StyleShot** | 外部扩散 | 2.4B Diffusion (24.5s) | 92.60% | — | 0.5750 | 0.6949 | 0.3569 | 0.7697‡ |
| **StyleID** | 外部扩散 | 1.1B DDIM+Attn (5.0s) | 73.20% | — | 0.5143 | 0.6469 | 0.3885 | 0.5554 |
| **`V19.T4_sweet_vgg` (`PRISM-Pro`)** | **PRISM-V19** | **免 `sid` 2步 S10-VGG (137ms)** | **59.00%** | **+0.0265** | **0.4618** | **0.5791** | **0.6783** | **0.5166** |
| **`V19.S2_prism1_l130`** | **PRISM-V19** | **免 `sid` 1步 1.5M ($\lambda=1.3$, 17ms)** | **50.17%** | **-0.0041** | **0.4504** | **0.5705** | **0.7032** | **0.5338** |
| **`V19.S3_oneshot_exemplar`** | **PRISM-V19** | **1-Shot 单参考图 1步 (17ms)** | **45.50%** | **-0.0160** | **0.4446** | **0.5664** | **0.7362** | **0.5111** |
| **`V19.S2_prism1_v2` (`PRISM-1`)** | **PRISM-V19** | **免 `sid` 1步 1.5M ($\lambda=1.0$, 17ms)** | **44.33%** | **-0.0219** | **0.4431** | **0.5655** | **0.7440** | **0.5057** |
| **StyTr2** | 外部 #3 | 48M Transformer (3.0s) | 39.50% | — | 0.4074 | 0.5317 | 0.7023 | 0.7386 |
| **V18.A1b / A2b** | WEAVE-V18 | 含 `sid` 校准 LD4+ / 1步 Pro | 38.00% | -0.0368 | 0.4470 | 0.5674 | 0.7428 | 0.5018 |
| **`V19.S2_prism1_l075`** | **PRISM-V19** | **免 `sid` 1步高保真 ($\lambda=0.75$, 17ms)** | **37.33%** | **-0.0416** | **0.4370** | **0.5610** | **0.7715** | **0.4848** |
| **Seedream 4.5** | 外部商业基座 | 商业大模型 API (~2.0s) | 37.30% | — | 0.3885 | 0.5314 | 0.7343 | 0.5202 |
| **Q2a + R1 Oracle + S10** | WEAVE-V6 | 0.22M + 1步 R1 | 36.30% | — | 0.4360 | 0.5587 | 0.7345 | 0.5014 |
| **V9.LD4** | WEAVE-V9 | N2C + 2步掩码 LD4 | 35.17% | -0.0531 | 0.4401 | 0.5615 | 0.7718 | 0.4872 |
| **AesPA-Net** | 外部注意力 | 31M Pattern-Attn (1.84s) | 31.00% | — | 0.4661 | 0.5784 | 0.6075 | 0.6247 |
| **P1_B4_distill50 + S10** | WEAVE-V6 | 0.22M 单步基底 | 22.20% | — | 0.4242 | 0.5483 | 0.8152 | 0.4396 |
| **V18.B2_CCF_w35** | WEAVE-V18 | $\tau_0=0$ 去风格自配对流 | 18.67% | -0.1755 | 0.4515 | 0.5668 | 0.7222 | 0.4549 |
| **Hist2Style / CUT / SaMam** | 外部轻量 | 直方图 / 非配对 / Mamba | 12.0% / 5.7% / 2.5% | — | 0.364–0.383 | 0.507–0.520 | 0.824–0.908 | 0.388–0.619 |
