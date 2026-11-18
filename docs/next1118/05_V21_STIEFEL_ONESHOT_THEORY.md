# 05 — `V21 / PRISM` 终极形态：Stiefel 极分解正交商（二阶 Gram 零泄漏定理）与零类别表 1-Shot 参考跨注意力流

**日期**：2026-11-18 · **代码包**：[`exp/v21/`](../../exp/v21/) · **实测审计快照**：`runs/v21/stiefel_gram_leak_audit.json`

---

## 一、从 `V20` 迈向更大果实的决定性发现：隐藏在 $\tilde{P}_{16}$ 中的 `92.33%` 二阶 Gram 源风格泄漏！

在 `V20` 中，虽然我们将 `DINO-S` 推升至 `0.5789`（`Style Affinity = 0.6683`），但仍观察到一个陡峭的权衡斜率：当保持高结构保真（`DINO-C = 0.629`）时，`DINO-S` 回落至 `0.521`；要冲上 `0.579` 则不得不通过无内容引导分支 $v_{s,\emptyset}$ 牺牲部分 `DINO-C`（降至 `0.501`）。

为什么即使移除了 $64\times 64$ 像素边缘、仅保留 $16\times 16$ 对角白化灰度 DINO 特征 $\tilde{P}_{16}(x_s)$，高内容约束下仍残留显著的源风格牵引？

### 定理 1（对角白化的二阶 Gram 盲区与 Stiefel 极分解正交商定理）
设 $P(Y(x)) \in \mathbb{R}^{D \times N}$（$D=384, N=256$）为灰度图提取的 $16\times 16$ 视觉语义 token 矩阵。
1. **对角白化的局限**：此前 `V18–V20` 对每个通道独立做一维中心化与方差归一 $\hat{P}_d = (P_d - \mu_d)/\sigma_d$，这仅令矩阵对角线 $\operatorname{diag}(\frac{1}{N}\tilde{P}\tilde{P}^\top) = \mathbf{1}$，却**原封不动保留了全部 $\binom{D}{2} = 73,536$ 个跨通道二阶相关性（即 Gatys 意义下的通道 Gram 矩阵 $G = \frac{1}{N}\tilde{P}\tilde{P}^\top$）**！
   - **实测铁证**：在 `train_4400 → holdout_300` 严格隔离探针下，旧版 $\tilde{P}_{16}$ 的一阶均值探针虽为 `18.33%`，但其**二阶通道 Gram 矩阵的 5 分类源画派准确率竟高达 `92.33%`（384 维随机投影探针达 `96.33%`）**！任何 $1\times 1$ 卷积投影层都能轻易从通道协方差中读出源图究竟是文艺复兴铜版排线还是印象派油画！
2. **Stiefel 极分解正交商（Stiefel Polar Quotient）**：
   设 $W_k \in \mathbb{R}^{D \times k}$（$k = 32 \ll N = 256$）为在干净训练集上拟合的前 $k$ 个通用物体部件主轴（人脸、天空、建筑、水体、树木等），令中心化语义投影 $Z = W_k^\top P(Y(x))\,(I_N - \frac{1}{N}\mathbf{1}\mathbf{1}^\top) \in \mathbb{R}^{k \times N}$ 的紧奇异值分解为 $Z = U \Sigma V^\top$（其中 $U \in O(k)$ 为 **$k\times k$ 方正交矩阵**，$V \in \mathcal{V}_k(\mathbb{R}^N)$ 满足 $V^\top \mathbf{1} = \mathbf{0}$ 且 $V^\top V = I_k$）。定义 **Stiefel 正交商算子**：
   $$Q_k(x) = \sqrt{N}\, U V^\top \in \mathbb{R}^{k \times N}.$$
   则对**任意输入图像 $x$**，恒有：
   - **严格零一阶空间均值**：$Q_k(x)\mathbf{1} = \sqrt{N}\,U(V^\top\mathbf{1}) \equiv \mathbf{0}$；
   - **严格恒等二阶通道 Gram 矩阵**：
     $$\frac{1}{N} Q_k(x) Q_k(x)^\top = U V^\top V U^\top = U I_k U^\top \equiv I_k \quad (\forall x)!!$$
   - **Frobenius 最优几何保真**：在所有满足通道正交白化 $\frac{1}{N}Q Q^\top = I_k$ 的表示中，$Q_k(x) = \sqrt{N} U V^\top$ 是原语义场 $Z$ 在 Frobenius 范数下的**唯一最近邻正交投影**！
   - **4090 实测核验（`stiefel_gram_leak_audit.json`）**：在 300 幅留出测试画作上，$\max |\frac{1}{256}Q_{32}Q_{32}^\top - I_{32}| = 2.21\times 10^{-4}$，**二阶 Gram 5 分类源风格泄漏从 `92.33%` 骤降至精确均匀随机水平 `20.00%`**！

---

## 二、100% 零类别表、任意单图 1-Shot 直出的跨注意力流架构（`PRISMAttnUNet`）

为彻底回答“像 `StyleGallery` 一样给任意一张新风格参考图 $x_{\mathrm{ref}}$ 直接出图”并消除 `128-D` 全局向量的空间纹理压缩瓶颈，`PRISMAttnUNet`（`24.78M` 参数）**完全删除了任何离散 `style_id` 或类别数超参**：

1. **任意参考图双层表征 $(u_{\mathrm{ref}}, K_{\mathrm{ref}}, V_{\mathrm{ref}})$**：
   - 全局连续纤维：$u_{\mathrm{ref}} = \phi_\eta(\Pi_\parallel(x_{\mathrm{ref}})) \in \mathbb{S}^{127}$；
   - 局部语义键（Semantic Keys）：$K_{\mathrm{ref}} = Q_{32}(x_{\mathrm{ref}}) \in \mathbb{R}^{256 \times 32}$（去风格 Stiefel 角色坐标：天空、人像、水面、背景平面）；
   - **无归一化局部笔触值（Un-Normalized Brushstroke Values）**：$V_{\mathrm{ref}} = [\mathrm{PCA}_{128}^{\mathrm{raw}}(P_{\mathrm{RGB}}(x_{\mathrm{ref}})),\; \mathrm{Unfold}_{4\times 4}(z_{\mathrm{ref}})] \in \mathbb{R}^{256 \times 192}$，完整保留参考图每个局部块的笔触肌理、Oklab 色调与潜空间高频子带，且进入注意力 Value 线性层前**不施加任何 LayerNorm/GroupNorm**！
2. **双尺度 Stiefel 商跨注意力（$16\times 16$ 瓶颈 + $32\times 32$ 解码层）**：
   - 源图查询由 $[h_\tau, Q_{32}(x_s)]$ 构成，参考键由 $[K_{\mathrm{ref}}, \mathrm{LN}(V_{\mathrm{ref}})]$ 构成，使源图的天空自动检索参考图的天空笔触、源图的人物自动检索参考图的人物笔触（单次 `FlashAttention2` 仅 `<0.2 ms`，无需 `StyleGallery` 耗时 `30.4 s` 的 DDIM 反演 + KMeans + 逐步梯度优化！）。
   - 同时天然兼容 **1-Shot 单参考图模式**（$L = 256$ tokens）与 **Multi-Shot 画廊模式**（拼接 $M=4$ 张参考图为 $L = 1024$ tokens，网络权重零改动）。
3. **异图近邻去捷径训练（Anti-Shortcut Distinct-Neighbor Pairing）**：
   - 训练期重建目标画作 $z_i$ 时，**严禁将 $z_i$ 自身作为参考图输入**，而是从 `train_4400` 内部 `Core-kNN` 图谱中随机抽取**另一幅不同构图的同流派核心画作 $z_j$（$\mathrm{pid}_j \neq \mathrm{pid}_i$）**并随机打乱 `256` 个 token 的空间顺序，同时将 `StyleGallery` 评测所用的 5 张参考原图从训练池彻底物理剔除，确保 1-Shot 评测 100% 零泄漏、零捷径。
