# WEAVE V6：真实画作驱动的条件风格生成

下一阶段最值得验证的，不是更大的DiT，也不是新的风格损失，而是一个不同的学习问题：从真实画作中提取尽可能不含风格的内容表示，再让模型根据目标Style-ID重建完整画风。

建议继续采用单卡RTX 3060、现有18K五风格数据、24小时连续实验。主线使用真实画作作为监督终点；备用路线使用真实艺术分布的条件Score Difference。两条路线都不需要人工拼贴GT、GAN、RGB训练损失或新的大型外部模型。

## 一、核心问题形式化

$$P(z_{\rm artwork}\mid c,\;s)$$

- $c$：风格尽可能被消除、但内容仍然保留的表示。
- $s$：目标风格。
- $z_{\rm artwork}$：真实艺术作品的latent。

## 二、主模型：Content-Canonicalized Conditional Flow (C³-Flow)

$$x\xrightarrow{E_{\rm VAE}}z \xrightarrow{C}c \xrightarrow{G_\theta(c,s)}\hat z \xrightarrow{D_{\rm VAE}}\hat x$$

### 1. 内容规范化器 C
- **C1（频带规范化）**：LL/LH/HL 逐通道去均值、标准差归一；重设为 D5 全局统计，HH 衰减。
- **C2（强风格剥离）**：C1 基础上进一步消除局部纹理幅度差异（局部窗口标准差归一化/平滑），保留归一化梯度和多尺度结构。
- **随机外观视图**：$c_i^{(k)}=C(z_i;\xi_k)$，随机扰动颜色统计和局部纹理强度，不改变空间坐标。
- **两项检查**：
  1. 风格泄漏：轻量 probe 预测 $c$ 所属五种风格（与原始 latent 比较，并在内容匹配的跨风格子集上测试以排除题材偏差）。
  2. 内容充分性：局部结构、自相似关系，以及同风格重建时的 DINO-C。

### 2. 生成器 F1（C³-Flow）
- 起点：$z_0=c+\sigma\epsilon$，终点：$z_1=z_{\rm real}$，插值：$z_\tau=(1-\tau)z_0+\tau z_1$，目标：$v^*=z_1-z_0$。
- 归一化的频带损失，避免高能量子带主导梯度。
- 主干：2–4M 参数轻量多尺度 Conv U-Net，规范化内容特征注入各分辨率层，Style-ID Embedding + 逐层 FiLM/AdaLN，连续 $\tau$ Embedding，完整 4×64×64 速度（含 HH），输出残差零初始化，4/8 步 ODE 推理。

### 3. 确定性算子对照 D1
- $\hat z = G_\phi(c,s)$，同架构规模（1–2M/2–3M），直接监督 $\mathcal L_{\rm direct} = \|\hat z - z_{\rm real}\|^2$（含频带归一化权重）。

## 三、三项条件有效性硬性验收
1. **训练条件有效性**：留出集上比较 $L_{\rm correct} < L_{\rm wrong}$ 的配对差与 95% CI。
2. **目标风格选择性**：同一内容 $c$ 生成 5 种风格 $G(c, s_1..s_5)$，比较正确目标原型相似度 vs 错误风格输出相似度。
3. **内容条件有效性**：固定 Style-ID，交换两张源图的内容条件 $c$，检查输出与源图结构的对应性。

## 四、备用路线：Score Difference ($g_{s\rightarrow t}$)
- 从真实 D5 latent 随机裁剪 128px/256px 对应区域（16×16 / 32×32 latent patch 或全图多尺度），加噪 $z_\sigma = z + \sigma\epsilon$，训练条件去噪网络 $\epsilon_\theta(z_\sigma, \sigma, s)$。
- 差分方向：$g_{s\rightarrow t} = (\epsilon_\theta(z_\sigma, \sigma, s) - \epsilon_\theta(z_\sigma, \sigma, t)) / \sigma$，满足 $g_{s\rightarrow s} = 0$。
- 结合源图结构约束进行小步长多尺度更新。

## 五、模型验收标准
- 600 跨风格 CSD $\Delta T \ge +0.04$（或显著超越旧 WEAVE +0.0203）
- DINO-C $\ge 0.8688$
- LPIPS $\le 0.2753$
- Style-ID 具有显著目标风格选择性
- 150 同风格 CSD 风格身份不显著退化
- 5 个目标风格至少 4 类改善（关注 Ukiyo-e）
