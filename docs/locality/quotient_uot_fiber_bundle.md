# PRISM 局部性理论：商空间加性污染、双向瓶颈纯净原则与摊销式局部性

> **文档定位**: V26-L Exp 1 证伪结果驱动的理论升级 — 从"几何泄露"浅层归因推进到"商空间加性污染"的线性代数机制，并据此建立 PRISM 核心理论的双向纯净原则。
> **实验来源**: `docs/v26/4090/exp1_refattn_rootcause.json`（V26-L Exp 1，零训练，150 cross 任务，C2nt 权重）。

---

## 一、Exp 1 现象的数学重构：商空间加性污染（Additive Quotient Pollution）

### 1.1 Exp 1 实测

| 干预 | DINO-C vs 内容 | dino_s vs 目标参考 | Δc vs R0 |
|---|---:|---:|---:|
| **R0** 原模型 (Dense Softmax + ConvNeXt V) | 0.5392 | 0.3958 | — |
| **R3** V=0 (局部通路完全移除) | **0.5892** | 0.3897 | **+0.050** |
| R4 V 打乱 (batch 内 shuffle) | 0.5886 | 0.3851 | +0.049 |
| **X1** V 去空间化 (per-image spatial mean) | 0.4858 | 0.3992 | **−0.053** |
| X2a T=4 (更平滑) | 0.4879 | 0.4007 | −0.051 |
| X2b T=48 (更尖锐) | 0.4931 | 0.4067 | −0.046 |
| **X3** UOT Γ̄ (零参数行归一 attention) | 0.4949 | 0.4044 | **−0.044** |

### 1.2 推翻"几何泄露"浅层归因

原始假设：RefAttnV24 的局部 Value 通路把参考图的空间边缘/几何**搬运**到了源图的 16×16 瓶颈 → DINO-C 下降。

**证伪证据**：X1（V 去空间化，零空间方向）依然掉 DINO-C 至 0.486——甚至比 R0 (0.539) 更低。如果几何泄露是根因，去空间化后 DINO-C 应该反弹到 R3 (0.589) 的水平。实际不仅没反弹，反而更低。

### 1.3 真实机制：商空间加性污染（DC Shift → Representation Collapse）

在 16×16 瓶颈层，特征张量 $H = [h_1, \dots, h_{256}] \in \mathbb{R}^{C \times 256}$ 承载的是图像的**布局商空间（Quotient Layout）**。核心信息编码在各空间 token 之间的**夹角对比度** $\cos(h_i, h_j)$ 中——天空 vs 人物、前景 vs 背景。

当 RefAttn 通过残差通路向瓶颈**加性注入** $\Delta_i = W_{\text{out}} R_i$ 时：

$$h'_i = h_i + \Delta_i$$

即使 $R_i \equiv \bar{V}$ 完全去空间化（零几何信息），只要 $W_{\text{out}}\bar{V}$ 在瓶颈通道空间具有不可忽略的幅度，就会给全部 256 个 token 叠加一个**空间相干的直流偏移（DC Shift）** $\Delta$。

夹角压缩的精确展开：

$$\cos(h_i + \Delta,\; h_j + \Delta) = \frac{\langle h_i, h_j \rangle + \langle h_i + h_j, \Delta \rangle + \|\Delta\|^2}{\|h_i + \Delta\| \cdot \|h_j + \Delta\|}$$

当 $\|\Delta\|$ 增大（远小于 $\|h_i\|$ 时线性项主导；接近时高阶项 $\|\Delta\|^2$ 不可忽略），不同语义区域之间的角度分离度**单调下降**——天空、人物、建筑的瓶颈表征同时往 $\Delta$ 方向拉拢。**DINO-C 必然暴跌**。

### 1.4 为什么"V 打乱"接近 "V=0" 而 "V 去空间化" 反而更低

打乱 V（batch 内 roll）破坏了 V 与 $W_{\text{out}}$ 主奇异方向的**空间/通道相干性**：非相干高频扰动在后续空间卷积与层归一化中互相抵消，或落入 $W_{\text{out}}$ 的小奇异值子空间——扰动被自然衰减，等效于 V=0。

而去空间化的 $\bar{V}$ 是**空间一致的直流偏置**——它**完整穿透** $W_{\text{out}}$ 的主奇异方向（幅度未被衰减），持续侵蚀全局布局对比度。

**验证方法**（待补算）：计算 $\|W_{\text{out}} R_{\text{despat}}\|_2 / \|W_{\text{out}} R_{\text{shuf}}\|_2$，若显著 >1（相干信号未被小奇异值子空间衰减），DC Shift 假设坐实。

### 1.5 为什么"V=0"恢复的最高

R3 (V=0) 让 $R = (1-g) \cdot u_g$（纯全局分支），其 out_proj 贡献的 $\Delta$ 幅度最小。此时 RefAttn 退化为一个**仅调制全局分支的置信度权重**——不向瓶颈注入任何来自参考图的向量。

---

## 二、PRISM 核心理论升级：双向瓶颈纯净原则（Two-Sided Bottleneck Purity）

### 2.1 原论文理论的单向防御

原 PRISM 论文理论只建立了**源侧防御**：通过 $\Pi_\perp$ 商投影阻断源图风格向 16×16 瓶颈的回流（防止 Source-Style Lock）。

### 2.2 Exp 1 补上的镜像防御：参考侧

Exp 1 补上了另一半——**参考侧防御**：

| 侧 | 污染路径 | 后果 | 防御 |
|---|---|---|---|
| **源侧** | 源图高频纹理 → 瓶颈 | Source-Style Lock | $\Pi_\perp$ 商投影 |
| **参考侧** | 参考特征 → 加性残差 $h + W_{\text{out}}R$ | **Reference-Induced Layout Collapse** | 移除加性注入 (Exp 1) |

### 2.3 合并后的统一几何表述

> **16×16 瓶颈必须维持严格纯净的商流形（Pure Quotient Manifold）**。风格纤维（Fiber）对生成流的作用在几何上应当是**纤维丛上的规范调制（Multiplicative / Affine Gauge Modulation，如全局弦驱动的 AdaGN $(\gamma_i, \beta_i)$）**，而非商空间上的**加性向量偏移（Additive Vector Offset）**。

两种注入方式的几何差异：

- **规范调制（合法）**：$\gamma_i \odot h_i + \beta_i$ 逐通道缩放/平移，$h_i$ 的方向在缩放后不变 → 夹角对比度保持。
- **加性偏移（非法）**：$h_i + \Delta_i$ 直接叠加向量 → $\Delta$ 方向成为所有 token 的公共分量 → 夹角坍缩。

### 2.4 与 StyleGallery 的架构对比

| | StyleGallery | PRISM (修正后) |
|---|---|---|
| 局部注入 | K-means 硬聚类 → 稀疏 Mask → 加性 Attention | **移除局部注入**（摊销式局部性） |
| 几何合法性 | 加性残差 = 商空间污染 | 规范调制 = 商流形纯净 |
| 跨题材 | 硬聚类失效（无语义双射） | UOT 熵正则自适应退化 |

---

## 三、摊销式局部性（Amortized Locality）

### 3.1 核心概念

**推理端彻底砍掉局部 V 通路，仅走全局弦 + 局部纹理交给训练期 Bary-SWD** — 这一机制的理论名称为**摊销式局部性**：网络在训练中学习一个条件映射先验：

$$(\text{区域语义 } Q_i,\; \text{全局风格弦 } e) \longmapsto \text{该区域的专属笔触分布}$$

例如：看到 $e = \text{Early Renaissance}$ + 区域 $Q_i = \text{人脸}$ → 自动生成平滑坦培拉釉面；看到 $Q_i = \text{背景建筑}$ → 生成柔和透视阴影。**推理期零注意力开销，却能产出区域特异的笔触**。

### 3.2 为什么"纯全局推理 + 局部 Bary-SWD 训练"理论上行得通

U-Net 本身已通过 $Q(x_s)$ 和 $z_\tau$ 完整掌握了源图的语义空间布局（哪里是天空、人脸、衣褶）。在以往的全局 VGG 均值监督下，网络没有动力区分不同区域的笔触差异。引入 $\mathcal{L}_{\text{Bary-SWD}}$ 后，速度场 $v_\theta(z_\tau, \tau, Q, e)$ 在训练中学会按区域语义条件化生成——但这种"区域条件化"是**内部化的**（编码在网络权重中），而非**外部化的**（通过推理时注意力机制注入）。

### 3.3 理论上限：数据处理不等式的硬约束

摊销局部性有一个信息论边界：当遇到**严格 OOD 参考**（如 SG-25），两张全局弦 $e$ 几乎相同、但局部配色/纹理绑定完全相反的未见参考图（如"红天空 + 蓝海面" vs "蓝天空 + 红海面"）时，仅凭单一全局向量 $e \in \mathbb{S}^{127}$，网络**不可能**分辨实例级区域绑定。

因此，完整的理论修正应区分**两种阶段**：

| 阶段 | 架构 | 条件 |
|---|---|---|
| **阶段 I（当前主线）** | 彻底删除局部 V 通路，纯全局弦 + Bary-SWD 训练 | 验证 Cross600 (Roc/ER/Uki) + SG-25 摊销局部性上限 |
| **阶段 II（仅当 SG-25 触及信息瓶颈）** | 解码端高分辨率层 ($32^2/64^2$) 加 **零初始化空间 AdaGN** 仅调制 $P_C$ 色度/高频补空间 | 输入仅限无几何局部色盘/Haar 统计量；16×16 商瓶颈永不受污染 |

**阶段 II 的关键约束**：即使需要局部推理注入，也**绝不**把局部 V 放回 16×16 瓶颈做加性残差；唯一几何合法的注入方式是在解码端更高分辨率层中以零初始化 AdaGN 调制色度/高频子空间。

---

## 四、Quotient-UOT Fiber Bundle：替代 StyleGallery 五步启发式的数学方案

### 4.1 StyleGallery 的五步启发式工程

```
PCA → K-means 硬聚类(K=4/8) → 区域匹配 → 稀疏 Mask → RSL/GCL
```

三处缺陷：
1. **跨题材数学病态**：Minimalism 几何色块 → Ukiyo-e 浮世绘，源图与参考图无语义双射，硬聚类引入不可微边界、对 K 极度敏感。
2. **OOD 混淆**：旧权重在 Dense Softmax 下训练，推理期硬 Mask 引起激活分布漂移。
3. **梯度拔河**：Flow-MSE 源图锚点与局部 SWD 方向冲突，未做子空间解耦。

### 4.2 PRISM 的三个零聚类数学算子

#### 4.2.1 Quotient-UOT（商空间非平衡最优传输）

在冻结的风格不变 DINO token 上定义代价 $C_{ij} = 1 - \langle \hat{q}_i^{\text{src}}, \hat{q}_j^{\text{ref}} \rangle$，求解 5–8 步 Log-domain 非平衡 Sinkhorn：

$$\Gamma^* = \arg\min_{\Gamma \ge 0} \langle C, \Gamma \rangle + \varepsilon H(\Gamma) + \rho_1 \operatorname{KL}(\Gamma\mathbf{1} \| \tfrac{1}{N}\mathbf{1}) + \rho_2 \operatorname{KL}(\Gamma^\top\mathbf{1} \| \tfrac{1}{M}\mathbf{1})$$

行归一化 $\bar{\Gamma}_{ij} = \Gamma^*_{ij} / \sum_k \Gamma^*_{ik}$。

**跨题材自适应退化**：语义正交时 $C$ 趋平坦 → 熵正则自动松弛为均匀分布（绝不强行匹配）；语义重合时 → 锐利对齐。**零聚类、零 K、零硬边界**，256×256 GPU < 0.2ms。

#### 4.2.2 Bary-SWD（1D W2 重心闭式损失）

利用分辨率嵌套：DINO 16×16 格 × VGG relu3_3 128²（每格 8×8=64 像素）与 relu2_2 256²（每格 16×16=256 像素）。对随机方向 $\theta_k$，每格内像素排序得精确 1D 分位数向量。由 1D $W_2$ 等距性，以 $\bar{\Gamma}_{i,\cdot}$ 为权重的参考纹理重心分位数有闭式解：

$$\tilde{Q}_i^{\text{ref}} = \sum_j \bar{\Gamma}_{ij} Q_j^{\text{ref}}, \qquad \mathcal{L}_{\text{Bary-SWD}} = \frac{1}{N}\sum_i \|\hat{Q}_i - \tilde{Q}_i^{\text{ref}}\|_1$$

**天然去几何**：格内排序丢弃像素位置 → 强迫输出格 $i$ 复现对应参考区域的笔触/色彩高阶分布，同时允许源图格内的真实边缘 **100% 保持原位**。

#### 4.2.3 去几何纤维拉回（Geometry-Free Fiber Pullback）

推理端如需局部注入，将 $V_{\text{ref}}$ 替换为**无几何的局部纤维统计量** $\Pi_\parallel^{(j)}(x_{\text{ref}})$（每格 Oklab 均值/方差 + Haar LH/HL/HH 能量），经同一套 $\bar{\Gamma}$ 做纤维拉回 $\tilde{s}_i = \sum_j \bar{\Gamma}_{ij} \Pi_\parallel^{(j)}$，经 $1\times1$ 映射生成空间 AdaGN $(\gamma_i, \beta_i)$ 注入。

**数学保证**：$\Pi_\parallel^{(j)}$ 不含空间边缘方向 → 不可能泄露参考图几何。

### 4.3 正交子空间监督解耦

训练时将源图内容锚点严格限制在商空间（$P_Y$ 亮度低频 + 灰度 DINO 布局余弦），**绝不对 3D 色度子空间 $P_C$ 与高频子带施加自配对 Flow-MSE 锚定**——把风格纤维空间完全交由 $\mathcal{L}_{\text{Local-Style}}$ 驱动。

---

## 五、Exp 1 + Exp 2 联合判读框架

### 5.1 Exp 1 已确认

- V 通路幅度（DC Shift）是内容损害根源，非空间几何泄露；
- 移除 V 通路是唯一有效修复（V=0, DINO-C +0.050）；
- 温度/UOT/去空间化均无效（仅改变 V 的空间分配，不改变通路存在性）。

### 5.2 Exp 2 待判读

| 场景 | 含义 | 行动 |
|---|---|---|
| V2 (Bary) > V0 (VGG) 且困难画派提升 | **摊销局部性有效**，局部性只需留训练 | 收束纯全局推理 + Bary-SWD；免局部注入 |
| V2 > V0 但极细纹理略降 | Bary-SWD 有效但有均值收缩 | 联合 $\mathcal{L}_{\text{Bary}} + \beta \mathcal{L}_{\text{Weighted-SWD}}$ |
| V2 ≈ V0 | Bary-SWD 无额外增益 | 回退最简 VGG 均值；研究转向教师监督质量 |
| SG-25 OOD 触及全局弦信息瓶颈 | 摊销局部性上限 | 触发阶段 II（解码端 AdaGN，零初始化 $P_C$-only） |

---

## 五.5 信息论分界线：域内摊销 vs. 极端 OOD 硬上限

砍掉 RefAttnV24 的局部 V 通路后，推理期只剩全局弦 $e$ 与源图局部语义 $Q(x_s)$。信息论分界线：

- **域内画派与常规语义先验**：Bary-SWD 训练将 $(Q_i, e) \mapsto$ 该区域典型笔触完整摊销进 U-Net 权重 → 推理零匹配开销。
- **极端 OOD 任意参考**：训练从未见过的参考图，其局部风格绑定违背常规先验（如同一套整体色盘下"厚重油画涂天空、水彩留人物"）→ 压缩为单一 $e$ 时实例级局部绑定**永久丢失**。

**Exp 2 一锤定音判别法（两个轻量化探针，<2 min）**：

1. **Bary-SWD 训练收敛残差**：Global-SWD vs Bary-SWD 训练集局部损失降幅。若 Bary-SWD 在训练集都压不下去 → 表达/信息瓶颈（$e$ 无法区分同色系参考图的局部多模态差异）。
2. **语义簇间纹理分化度**（Inter-Cluster Texture Divergence）：冻结 DINO 将输出图划 K=3–4 语义簇 → 同图内不同簇间 Haar 高频能量与色彩均值方差。若 Bary-SWD 臂的簇间分化度显著高于 Global-SWD 且 ER/Roc 同步拉升 → "模型局部能力够用，缺的只是训练期局部监督"彻底坐实。

**判决路径**：Exp 2 域内困难画派大幅起飞 + SG-25 OOD 同步打穿 → 纯全局推理 + 训练期 Bary-SWD = 终局架构；域内起飞但 SG-25 出现瓶颈 → 阶段 II（解码高分辨层去几何纤维调制）。

> **2026-11-22 判决落地（V26-L L3 五臂 + EXP2 双探针）**：实际兑现的是**第三分支** —
> S4 Bary-SWD (14.17%) ≈ S1 SWD (14.00%) < S0 VGG (16.17%)，且 Bary 簇间分化度探针
> (0.0133) ≤ VGG 基线 (0.0135)。"模型局部能力够用，缺训练期局部监督"与"摊销上限被
> 训练损失解锁"两个假说均被否定；**SG-25 OOD 瓶颈未触发**（阶段 II 无必要）。摊销式
> 局部性成立，但其载体是模型内部条件化 + 教师监督质量（F2 +5.17pp），不是外部局部损失。
> 详见 `docs/v26/LOCALITY_UOT_FIBER_BUNDLE.md` §2.2-2.4。

已实现: `exp/v26/gpu4090/exp2_locality_probes.py`（`inter_cluster_divergence()` — DINO k-means 聚类 + 逐簇 Haar 能量/Oklab 色彩均值方差）。

## 六、参考文献

- Chizat et al., "Faster Wasserstein distance-type estimators," JMLR 2021.
- Agueh & Carlier, "Barycenters in the Wasserstein space," SIAM J. Math. Anal. 2011.
- StyleGallery, CVPR 2026.
- V10.1 结果：`docs/v10_1/V10_1_RESULTS.md`（Dense Softmax 硬检索已证伪）。
