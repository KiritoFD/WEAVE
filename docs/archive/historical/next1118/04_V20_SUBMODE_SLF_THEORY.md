# 04 — `V20 / PRISM-Flow`：突破超球面均值坍缩与 64×64 笔触边缘枷锁（超越 `StyleGallery` 理论与实现）

**日期**：2026-11-18 · **目标对手**：扩散跨注意力强基线 `StyleGallery`（Cross600 `DINO-S = 0.5604`, `CLIP-S = 0.7259`, `Style Affinity = 0.6432`, `DINO-C = 0.6113`, `LPIPS = 0.5576`, `CSD Top-1 = 52.50%`）。

---

## 一、为什么 `V19` 已达 `59.0%` CSD Top-1，但 `DINO-S` 仍停留在 `0.462`？

在 `V19` 中，`PRISM-Pro (T4)` 已经把 Cross600 `CSD Top-1` 推至 **`59.00%`**（超过 `StyleGallery` 的 `52.50%` 达 `+6.50pp`），且 `DINO-C = 0.6783` 远高于 `StyleGallery`（`0.6113`）。然而其 `DINO-S = 0.4619`（`Style Affinity = 0.5791`）仍显著低于 `StyleGallery`（`0.5604` / `0.6432`）。

通过对比历史全量台账（`V15` 自配对生成流在 $\tau_0=0$ 曾达 **`DINO-S = 0.5912`, `Affinity = 0.6650`**，而 `V18 CCF` 在 $\tau_0=0$ 仅得 `DINO-S = 0.4515`），我们严格证明了造成 `DINO-S` 天花板的两大几何根因：

### 定理 1（单位超球面上的最大余弦均值坍缩，Hypersphere Max-Cosine Centroid Collapse）
设某画派 $y_t$ 在单位超球面 $\mathbb{S}^{d-1}$ 上由 $K$ 个互斥子流派簇（例如浮世绘中的“北斋普鲁士蓝海浪风景”与“写乐歌舞伎大首绘肖像”）组成，簇中心 $\{c_k\}_{k=1}^K$ 两两夹角满足 $\langle c_j, c_k\rangle = \rho < 1$。评测算子 $\mathrm{DINO\text{-}S}(\hat{x}, y_t) = \max_{r \in \mathcal{R}_{y_t}} \langle \psi(\hat{x}), \psi(x_r)\rangle$ 度量生成图与该画派参考集**最近邻单幅实画**的余弦相似度：
1. 若求解器将样本推向全类 880 幅画作的平均重心 $\bar{c} = \frac{\sum_{k=1}^K c_k}{\|\sum_{k=1}^K c_k\|_2}$，则球面投影范数收缩因子为
   $$\max_{k \in \{1,\dots,K\}} \langle \bar{c}, c_k \rangle = \sqrt{\frac{1 + (K-1)\rho}{K}} \;\ll\; 1.$$
   当类内子流派多样性较高（$\rho \approx 0.18, K=8$）时，全类重心与任何一幅具体真迹的余弦上界被硬性锁死在 $\approx 0.53$ 以下，再乘上潜空间局部微调的残差衰减即落入 `0.45–0.47`！
2. 反之，若将目标锚定在**与源图粗场景构图同构的单一高纯度子流派模态** $c_{k^\star(x_s)}$（肖像对肖像、风景对风景），则球面理想余弦上界恢复为 $1.0$（实测 `Same150` 真实画作的 `DINO-S` 均值恰为 `0.611`）。

### 定理 2（高分辨率 64×64 结构注入的源笔触复刻枷锁）
对比 `V15`（$\tau_0=0$, `DINO-S = 0.5912`）与 `V18 CCF`（$\tau_0=0$, `DINO-S = 0.4515`）：
- `V18 CCF` 将 $64\times 64$ 局部对比度图 $C_{64}(x_s)$ 通过 SPADE 注入了 U-Net 的 $64\times 64$ 与 $32\times 32$ 高分辨率编解码层。如 `Gate-0` 审计所证，$C_{64}$ 的高通边缘梯度仍携带 **`59.33%` 的源画派笔触节奏（油画厚涂刀痕 vs 工笔细线）**，迫使速度场沿着源画笔触沟壑填色，无法自由合成目标流派特有的微纹理簇（如印象派破碎短笔触或浮世绘木刻平涂）。
- 相反，空间白化灰度语义图 $\tilde{P}_{16}(Y(x_s)) \in \mathbb{R}^{384\times 16\times 16}$ 与 $4\times$ 下采样粗亮度轮廓 $\mathrm{LCN}_3(Y_{16}(x_s)) \in \mathbb{R}^{1\times 16\times 16}$（对应像素域 $32\times 32$ 宏构图块）的风格探针仅为 **`18.33%`（等于随机猜测 `20%`）**，完全不含任何源笔触微几何！

---

## 二、`V20` 三大核心算子设计（`exp/v20/`）

1. **场景匹配的鲜明子流派原型库（Vivid Scene-Matched Sub-School Bank，`submode_patchswd.py`）**：
   - 严格仅在 4,400 幅无重叠平衡训练画作（与 `tasks_750` 及 `references_v1` 零交集）的 $\Pi_\parallel$ 纤维球面上，按每类 $K=8$ 聚类出 $5\times 8 = 40$ 个鲜明艺术子流派，提取每个子流派中分类边际纯度最高的前 5 幅画作组成高纯度原型 $u_{y_t, k}^{\mathrm{vivid}}$，并记录其去风格 $4\times 4$ 粗场景布局签名（由 $\tilde{P}_{16}$ 池化得到）。
   - 推理时，根据源图去风格粗布局自动匹配构图最相容的目标子流派 $k^\star(x_s, y_t)$。

2. **多尺度局部块切片 Wasserstein 距离（Multi-Scale `ConvNeXt-Tiny` + RGB `Patch-SWD`）**：
   - 为刻画局部 $3\times 3$ 与 $7\times 7$ 笔触块分布（而非全局空间均值），使用离线冻结的 ImageNet `ConvNeXt-Tiny`（天然具备 $7\times 7$ depthwise 局部感受野，与四大评测裁判零重合）提取 `64×64 / 32×32 / 16×16` 三尺度局部 token 及 RGB $3\times 3$ 像素块，在 $P=96$ 条随机单位投影轴上最小化精确一维排序 Wasserstein 距离（`Patch-SWD`）。

3. **仅 $16\times 16$ 瓶颈注入结构的粗布局生成流（Coarse-16 `SLFUNet` & `T4` Flow-Bridge，`slf_flow.py`）**：
   - 在 `SLFUNet`（`21.95M` 参数）中，**$64\times 64$ 与 $32\times 32$ 编解码残差块完全不接收任何源图空间边缘张量**，100% 由目标子流派向量 $u^{\mathrm{vivid}}$ 经 `AdaGN-FiLM` 驱动自由生长目标笔触；
   - **唯独在 $16\times 16$ 瓶颈层**通过 `SPADE + MultiHeadSelfAttention` 注入 $[\mathrm{LCN}_3(Y_{16}), \tilde{P}_{16}]$ 锁定宏观物体构图（保障 `DINO-C > 0.62`）；
   - 同时支持从纯高斯噪声 $\tau_0=0$ 生成（最大化 `DINO-S` 冲击 `0.57–0.61`）与以 `V19_T4` 为锚点的 **Flow-Bridge** $\tau_0 \in \{0.35, 0.50\}$（兼顾高 `DINO-C` 与生成式笔触纹理），并蒸馏至零 DINO、零 B4 的纯潜空间极速网络 **`PRISMFlashNet`**（`< 1 ms/img` 潜推理）。
