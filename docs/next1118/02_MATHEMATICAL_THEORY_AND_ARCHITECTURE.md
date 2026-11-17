# 02 · 正交商—纤维对偶与各向异性谱传输：完整数学推导与网络细枝末节

本文档将 `PRISM v2 / V19` 的全部数学理论、张量形状、数值稳定项、可微解码器契约、无归一化多尺度判别器以及语义路由单步学生网络逐层展开至可直接逐行写成 PyTorch 代码的颗粒度。

---

## 一、正交商—纤维对偶定理（Orthogonal Quotient–Fiber Duality）

### 1.1 视觉信号的直和分解观
设输入 RGB 图像 $x \in [0,1]^{3\times 512\times 512}$，ITU-R BT.601 亮度权重 $w_Y = [0.299, 0.587, 0.114]^\top$，点态亮度场 $Y(x) = w_Y^\top x \in [0,1]^{1\times 512\times 512}$。我们定义一对严格互补的特征提取算子 $(\Pi_\perp, \Pi_\parallel)$：

1. **不变结构商算子 $\Pi_\perp(x) = \big(C_{64}(x),\; \tilde{P}_{16}(x)\big)$**：
   - **$C_{64}(x) \in [-1,1]^{4\times 64\times 64}$**：完全由灰度亮度 $Y(x)$ 经 $4\times$ 平均池化 + $3\times 3$ 高斯消厚涂平滑 + $2\times$ 池化得到 $Y_{64}$，拼接多尺度局部对比度归一化 $\mathrm{LCN}_5(Y_{64})$、$\mathrm{LCN}_{13}(Y_{64})$、95% 分位数归一化 Sobel 梯度模与 RMS 归一化拉普拉斯响应。其中方差下界 $\delta_0 = 10^{-8}$ 置于开方号内部：
     $$\mathrm{LCN}_k(Y_{64}) = \frac{1}{3}\mathrm{clip}\!\left(\frac{Y_{64} - \mathcal{A}_k(Y_{64})}{\sqrt{\max\big(\mathcal{A}_k((Y_{64}-\mathcal{A}_k(Y_{64}))^2), 10^{-8}\big)} + \varepsilon_k}, -3, 3\right).$$
   - **$\tilde{P}_{16}(x) \in \R^{384\times 16\times 16}$**：将三通道灰度图 $[Y,Y,Y]$ 输入冻结 DINOv2-small 提取 $16\times 16$ patch 网格 $P_{\mathrm{gray}} \in \R^{384\times 16\times 16}$，依次执行单图单通道空间标准化、逐位置球面投影与空间重中心化：
     $$\hat{P}_d = \frac{P_{\mathrm{gray}, d} - \mu_{\mathrm{sp}}(P_{\mathrm{gray}, d})}{\sigma_{\mathrm{sp}}(P_{\mathrm{gray}, d}) + 10^{-5}}, \quad \bar{P}_{:,u} = \sqrt{384}\frac{\hat{P}_{:,u}}{\max(\|\hat{P}_{:,u}\|_2, 10^{-6})}, \quad \tilde{P}_{16} = \bar{P} - \mu_{\mathrm{sp}}(\bar{P}).$$
   - **不变性保证**：$\Pi_\perp(x)$ 对任意满足 $w_Y^\top \Delta x = 0$ 的色度变换精确不变，且空间一阶矩恒为零 $\mu_{\mathrm{sp}}(\tilde{P}_{16}) \equiv \mathbf{0}$。

2. **互补风格纤维算子 $\Pi_\parallel(x) \in \R^{795}$**：
   恰收集被 $\Pi_\perp(x)$ 投影抹除的两类全局风格信息：
   - **三维色彩分布统计 $\psi_{\mathrm{color}}(x) \in \R^{27}$**：将 $x$ 转换至感知均匀的 CIE Oklab（或线性 RGB + 对手色空间 $[L, C_1, C_2]$），提取每通道均值（$3$）、标准差（$3$）、分位数 $[q_{05}, q_{25}, q_{50}, q_{75}, q_{95}]$（$15$）及 $3\times 3$ 通道协方差上三角（$6$），共 $27$ 维；
   - **原始彩色 DINOv2 空间一阶/二阶矩 $\psi_{\mathrm{dino}}(x) \in \R^{768}$**：将彩色原图 $x$ 输入同一冻结 DINOv2-small（无需额外加载新骨干！）提取 patch 网格 $P_{\mathrm{rgb}} \in \R^{384\times 16\times 16}$，拼接其空间均值 $\mu_{\mathrm{sp}}(P_{\mathrm{rgb}}) \in \R^{384}$ 与空间标准差 $\sigma_{\mathrm{sp}}(P_{\mathrm{rgb}}) \in \R^{384}$，共 $768$ 维；
   - 经训练集全局 z-score 标准化后拼接得 $\Pi_\parallel(x) = \mathrm{Norm}\big([\psi_{\mathrm{color}}(x);\; \psi_{\mathrm{dino}}(x)]\big) \in \R^{795}$。

### 1.2 连续风格流形嵌入与免源标签通用弦（Source-Label-Free Universal Chord）
构建轻量两层残差 MLP 编码器 $\phi_\eta: \R^{795} \to \R^{d_s}$（取 $d_s = 128$，输出归一化至超球面再乘可学习温度 $\alpha_s$），与 5 个可学习画派原型 $U = [\bar{u}_0, \dots, \bar{u}_4] \in \R^{5\times 128}$ 联合预训练（仅在 4,400 张训练画作上以 1:1:1:1:1 均衡采样最小化监督对比 / 余弦边际交叉熵，训练耗时 `<15 秒`）：
$$\mathcal{L}_{\mathrm{fiber}}(\eta, U) = \E_{(x, y)\sim\mathcal{D}_{\mathrm{train}}^{\mathrm{bal}}}\!\left[ -\log \frac{\exp(\tau^{-1}\langle \phi_\eta(\Pi_\parallel(x)), \bar{u}_y\rangle - m\cdot \mathbf{1})}{\exp(\tau^{-1}\langle \phi_\eta(\Pi_\parallel(x)), \bar{u}_y\rangle - m) + \sum_{k\neq y}\exp(\tau^{-1}\langle \phi_\eta(\Pi_\parallel(x)), \bar{u}_k\rangle)} \right].$$

由此定义**免源标签通用传输弦（Universal Chord）**：
- **类别条件迁移**（输入任意图像 $x_s$ 与目标画派 ID $y_t$，**完全不需要源标签 `sid`**）：
  $$e(x_s \to y_t) = \bar{u}_{y_t} - \phi_\eta(\Pi_\parallel(x_s)) \in \R^{128}.$$
- **单参考图零样本迁移**（输入任意图像 $x_s$ 与任意未见画派参考图 $x_{\mathrm{ref}}$）：
  $$e(x_s \to x_{\mathrm{ref}}) = \phi_\eta(\Pi_\parallel(x_{\mathrm{ref}})) - \phi_\eta(\Pi_\parallel(x_s)) \in \R^{128}.$$
- **自重构与同画派光滑门控**：
  - 在参考图模式下，若 $x_{\mathrm{ref}} = x_s$，则 $e(x_s \to x_s) \equiv \mathbf{0}$ **在代数上精确成立**；
  - 在类别模式下，定义连续软门控 $g_{\mathrm{id}}(e) = \tanh(\|e\|_2 / \sigma_e)^2$（并在已知 `tid == sid` 的协议任务上兼容精确硬门控），保证自然照片（$\phi_\eta(\Pi_\parallel(x_{\mathrm{photo}}))$ 远离任何油画原型）获得满幅弦驱动，而接近目标画派重心的样本平滑收敛。

---

## 二、潜空间亮度—色度正交分裂与各向异性信任域

### 2.1 四维潜空间的线性色度子空间剥离
在 4,400 张训练潜变量 $z \in \R^{4\times 64\times 64}$ 与对应真值降采样亮度 $Y_{64} \in \R^{1\times 64\times 64}$ 上求解最小二乘回归 $Y_{64} \approx w_{\mathrm{luma}}^\top z + b_{\mathrm{luma}}$，令单位向量
$$\hat{w}_Y = \frac{w_{\mathrm{luma}}}{\|w_{\mathrm{luma}}\|_2} \in \R^4, \qquad P_Y = \hat{w}_Y \hat{w}_Y^\top \in \R^{4\times 4}, \qquad P_C = I_4 - \hat{w}_Y \hat{w}_Y^\top \in \R^{4\times 4}.$$
则任意潜张量 $z \in \R^{4\times H\times W}$ 正交直和分解为**一维结构亮度分量** $z_Y = P_Y z$ 与**三维色度分量** $z_C = P_C z$（满足 $\langle z_Y, z_C\rangle_F \equiv 0$）。

### 2.2 潜色度目标预对齐起点（拔除 `z_0` 中 `71.3%` 的源色盘残留）
预先在训练集上按画派 $y \in \{0,\dots,4\}$ 统计三维色度子空间的通道均值 $\mu_C^{(y)} \in \R^{4\times 1\times 1}$ 与协方差平方根 $\Sigma_C^{(y)} \in \R^{4\times 4}$（限制在 $\operatorname{im} P_C$ 内）。给定源潜变量 $z_s$ 与目标 $y_t$（或参考潜变量 $z_{\mathrm{ref}}$）：
1. 对 $z_s$ 执行单层 Haar DWT 得 $(\ell_s, h_{1,s}, h_{2,s}, h_{3,s})$；
2. **仅对低频基底 $\ell_s$ 的三维色度分量 $P_C \ell_s$ 执行白化—重着色**（彻底洗掉古典油画暗褐底色，直接注入浮世绘/洛可可底色先验，同时**纹丝不动地保留一维亮度基底 $P_Y \ell_s$**）：
   $$\tilde{\ell}_{s\to t} = P_Y \ell_s + P_C\!\left(\Sigma_C^{(y_t)} \big(\Sigma_C(\ell_s) + \varepsilon I\big)^{-1/2}\big(P_C \ell_s - \mu_{\mathrm{sp}}(P_C \ell_s)\big) + \mu_C^{(y_t)}\right);$$
3. 高频子带执行温和方差对齐后逆变换得去色锚点 $z_s^\perp$，并经 `B4` 一步生成色度解耦起点 $z_0^\perp$。

### 2.3 各向异性结构锁定信任域投影算子 $\Pi_{\rho_Y, \rho_C}$
取代旧版惩罚色盘变化的各向同性 `project_rms(z, z0, rho)`，对候选潜变量 $z$ 定义正交分裂投影：
1. **色度子空间（宽半径 $\rho_C = 0.42$）**：允许全画布大跨度色盘与明暗基调迁移：
   $$\Delta_C = P_C(z - z_0^\perp), \qquad \tilde{\Delta}_C = \Delta_C \cdot \min\!\left(1,\; \frac{\rho_C}{\|\Delta_C\|_{\mathrm{RMS}} + 10^{-8}}\right);$$
2. **亮度子空间再分裂为“全局明暗基调（中半径 $\rho_{Y,\mathrm{low}} = 0.28$）”与“高频局部对比度边缘（紧半径 $\rho_{Y,\mathrm{edge}} = 0.14$）”**：
   用 $7\times 7$ 均值滤波将 $\Delta_Y = P_Y(z - z_0^\perp)$ 分解为平滑明暗趋势 $\Delta_Y^{\mathrm{low}} = \mathcal{A}_7(\Delta_Y)$ 与精细结构边缘残差 $\Delta_Y^{\mathrm{high}} = \Delta_Y - \Delta_Y^{\mathrm{low}}$，分别限幅：
   $$\tilde{\Delta}_Y = \Delta_Y^{\mathrm{low}}\cdot\min\!\left(1, \frac{\rho_{Y,\mathrm{low}}}{\|\Delta_Y^{\mathrm{low}}\|_{\mathrm{RMS}}+10^{-8}}\right) + \Delta_Y^{\mathrm{high}}\cdot\min\!\left(1, \frac{\rho_{Y,\mathrm{high}}}{\|\Delta_Y^{\mathrm{high}}\|_{\mathrm{RMS}}+10^{-8}}\right);$$
3. 合并得可行候选 $\tilde{z} = z_0^\perp + \tilde{\Delta}_Y + \tilde{\Delta}_C$，并以**真实亮度局部对比度结构偏移**
   $$\mathcal{C}_{\mathrm{LCN}}(\tilde{z}; z_s) = \big\|\mathrm{LCN}_5(\hat{w}_Y^\top \tilde{z} + b_{\mathrm{luma}}) - \mathrm{LCN}_5(\hat{w}_Y^\top z_s + b_{\mathrm{luma}})\big\|_2^2 \le \epsilon_{\mathrm{LCN}}$$
   作为唯一结构准入判据！这样一来，**把整片天空从深褐改成普鲁士蓝在 $\mathcal{C}_{\mathrm{LCN}}$ 下代价几乎为零，而任何扭曲人物五官轮廓的扰动都会被 $\mathcal{C}_{\mathrm{LCN}}$ 精准拦截**。

---

## 三、穿过可微 `S10` 的无归一化多尺度像素域均衡判别器（`V19-Critic`）

### 3.1 严格防泄漏契约（Zero Referee Contamination）
- **禁用名单**：在构建 `V19-Critic` 与离线教师时，**严禁导入或调用** `CSD-ViT-L`、`DINOv2`（仅在输入端静态提取一次无梯度源图结构 $\Pi_\perp(x_s)$ 与源坐标 $\Pi_\parallel(x_s)$，绝不对生成图 $\hat{x}$ 求梯度！）、`CLIP`、`LPIPS-Alex`，且严禁接触 `data/test`（150 源 + 150 参考）。
- **合法特征骨干**：采用 `torchvision` 标准 ImageNet-1K 预训练 **`VGG-16`（无 BatchNorm/GroupNorm 版！）** 前四段特征层 `relu1_2` ($64$ch), `relu2_2` ($128$ch), `relu3_3` ($256$ch), `relu4_3` ($512$ch)（输入缩放到 $256\times 256$ 以兼顾多尺度感受野与极速反向传播）。

### 3.2 为什么无归一化 `VGG-16` 多尺度矩 + 低秩 Gram 恰是 `R1` 的完美解药？
对 `S10` 可微解码输出 $\hat{x} = \mathcal{D}_\star(z) \in [0,1]^{3\times 512\times 512}$：
1. **无任何 `GroupNorm / InstanceNorm` 抹除幅值**：`VGG-16` 纯卷积 + ReLU 完整保留了通道激活均值 $\mu_l(\hat{x})$、通道标准差 $\sigma_l(\hat{x})$（共 $(64+128+256+512)\times 2 = 1920$ 维）；
2. **显式捕获跨通道笔触纹理协方差与三维色彩统计**：
   - 在 `relu2_2` 与 `relu3_3` 上经冻结正交随机投影 $R_l \in \R^{C_l \times 64}$ 计算归一化二阶 Gram 上三角特征 $\mathrm{triu}(G_l) \in \R^{2080\times 2}$；
   - 拼接可微 Oklab/RGB 色彩均值、标准差与色彩协方差（$15$ 维）；
3. **1:1:1:1:1 严格均衡训练 + 多视角裁剪增强**：
   在 4,400 张干净训练画作（及它们的 `S10` 重建图，消除编解码域差！）上，以每 batch 五画派严格等比例采样训练轻量两层分类头 $F_\psi: \R^{D_{\mathrm{feat}}} \to \R^5$（带 Cosine/Linear 双头集成与 Label Smoothing $0.05$），并在 300 张独立校准集上执行：
   - **Gate-1 预验收**：核查 300 张留出集上的 5 分类准确率（特别是 `ER` 与 `Ukiyo-e` 召回率须 $\ge 70\%$，远超旧 `R1`！）以及对数值先验校准偏置 $\hat{b}_{\mathrm{ho}}$。

### 3.3 可微 `S10` 双向切向求解器（`PRISM-Pro v2`）
封装可微版 `decode_s10_grad(s10, z)`（严格遵循 `AGENTS.md`：输入 scaled latent $z$，内部 `/ 0.18215` 并穿过 `StudentDecoder` 内置单次 `post_quant_conv`，FP16 前向/反向，峰值显存 `<2.5 GB` @ B16！）。
在每次迭代 $m = 1, \dots, M$（$M=4$）：
1. 计算像素域先验校准双向边际目标（注意：排斥项使用预测源分布或连续源伪标签 $\hat{p}_s = \mathrm{softmax}(\tau^{-1} U \phi_\eta(\Pi_\parallel(x_s)))$，**推理期完全无需真值 `sid`**！）：
   $$\mathcal{J}_{\mathrm{RGB}}(z; x_s, y_t) = \tilde{m}_{F_\psi}(\mathcal{D}_\star(z), y_t) - \gamma \sum_{k\neq y_t} \hat{p}_{s,k}\,\tilde{f}_{\psi,k}(\mathcal{D}_\star(z)) - \lambda_{\mathrm{proto}}\|\psi_{\mathrm{style}}(\mathcal{D}_\star(z)) - \bar{\psi}_{y_t}^{\mathrm{train}}\|_2;$$
2. 反向传播一次穿过 `S10` 得潜空间精确梯度 $g(z) = \nabla_z \mathcal{J}_{\mathrm{RGB}}(z)$；
3. 将 $g(z)$ 正交分解为亮度梯度 $g_Y = P_Y g(z)$ 与色度梯度 $g_C = P_C g(z)$，分别做 RMS 归一化（防止亮度或色度单方梯度淹没另一方！）：
   $$d_{\mathrm{aniso}}(z) = \beta_Y \mathcal{N}_{\mathrm{RMS}}(g_Y) + \beta_C \mathcal{N}_{\mathrm{RMS}}(g_C)$$
   并与旧潜空间快速方向构成候选集，在全画布无死角（不再冻结 65% 像素！）经各向异性投影 $\Pi_{\rho_Y, \rho_C}$ 与 $\mathcal{C}_{\mathrm{LCN}} \le \epsilon_{\mathrm{LCN}}$ 线搜索选出最优步。

---

## 四、语义路由空间弦单步网络（`PRISM-1 v2` 精确架构）

为彻底消除单步与多步之间的落差，我们在保持 **`<30 ms/img` 纯单步零梯度推理**的前提下，设计 `SpatiallyRoutedChordNet`（参数量约 **`1.25M`**）：

1. **输入张量（全 $64\times 64$ 对齐）**：
   - 去色度锚点基底 $z_0^\perp \in \R^{4\times 64\times 64}$ 与源潜亮度图 $P_Y z_s \in \R^{4\times 64\times 64}$；
   - Gate-0 多尺度亮度结构 $C_{64}(x_s) \in \R^{4\times 64\times 64}$；
   - 拼接共 $12$ 通道进入 $3\times 3$ Stem Conv 得 $H_0 \in \R^{96\times 64\times 64}$（宽度 $W = 96$）。
2. **语义—弦交叉门控模块（Semantic-Chord Spatial Router，核心精巧设计！）**：
   - 给定零均值球面 DINO 布局 $\tilde{P}_{16} \in \R^{384\times 16\times 16}$ 与免源标签通用弦 $e = e(x_s \to \tau) \in \R^{128}$：
   - 在低开销的 $16\times 16$ 分辨率上计算**语义角色感知的空间弦场**：
     $$Q_{16} = \mathrm{GN}(\mathrm{Conv}_{1\times 1}(\tilde{P}_{16})) \in \R^{96\times 16\times 16}, \qquad k_e = W_e e \in \R^{96},$$
     $$M_{16}(u) = \mathrm{SiLU}\big(Q_{16}(u) \odot k_e + b_e\big) \in \R^{96\times 16\times 16},$$
     再经 $3\times 3$ 深度可分离平滑卷积与双线性上采样得空间异质调制参数 $(\Gamma_{64}, B_{64}) \in \R^{192\times 64\times 64}$！
   - **直观物理意义**：每个 $16\times 16$ 语义块根据自身是“人脸皮肤”、“天空”、“树叶”还是“几何背景”，对同一根弦向量 $e$ 门控出不同的通道增益 $\Gamma_{64}(x,y)$ 与偏置 $B_{64}(x,y)$！
3. **6 层多尺度膨胀残差主干（扩大感受野覆盖全局构图）**：
   - 采用膨胀率序列 $d \in [1, 1, 2, 2, 4, 1]$ 的 6 个 `GroupNorm(16, 96) -> SpatialFiLM(Γ_64, B_64) -> SiLU -> DilatedConv3x3` 残差块，有效感受野从 `17×17` 扩大至 **`53×53`（覆盖全图 83% 潜空间宽度！）**，足以感知整块平涂色域与长线条轮廓，而计算量与普通 $3\times 3$ 卷积完全相同！
4. **输出头与复合蒸馏损失**：
   - 零初始化 $3\times 3$ 输出卷积生成残差 $\Delta_\theta \in \R^{4\times 64\times 64}$，乘连续弦幅门控 $g_{\mathrm{id}}(e)$ 加在 $z_0^\perp$ 上；
   - 在 **4,000 对**由 `PRISM-Pro v2` 离线生成的五画派严格均衡跨域轨迹 $(x_s, y_t, z^\star_{s\to t})$ + **500 对**自重构恒等样本 $(x_s, x_s \Rightarrow \Delta \equiv 0)$ 上联合训练（4090 上 batch=64，6,000 步仅需 **~6 分钟**！）：
     $$\mathcal{L}_{\mathrm{stu}} = \|\hat{z} - z^\star\|_H + 0.5\|P_H(\hat{z} - z^\star)\|_2^2 + \lambda_{\mathrm{rgb}}\,\mathcal{L}_{\mathrm{moment}}\big(\mathcal{D}_\star(\hat{z}), \mathcal{D}_\star(z^\star)\big),$$
     其中最后一项在每步随机抽 16 张子集穿过冻结 `S10` 直接对齐教师的 `VGG` 多尺度无归一化矩，消除潜空间 $L_2$ 回归带来的色彩发灰均值化效应！
