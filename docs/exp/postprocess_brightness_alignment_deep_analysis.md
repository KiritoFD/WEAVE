# 后处理“看起来更好、指标却恶化”深度分析

## 1. 核心发现

对 D5-512 上 750 张图做 AdaIN s=0.5 后处理：

- **613/750 张图 LPIPS 改善**（像素/感知层面的内容保持变好）。
- **262/750 张图 CLIP-S 改善**，**488/750 张图 CLIP-S 下降**。
- **DINO-S 几乎不变**，说明 DINOv2 对亮度/对比度偏移不如 CLIP 敏感。

结论：**人眼觉得“不那么白化、更自然”≠ 机器指标认为的“更接近目标风格”**。后处理修正了亮度，却常常让图像在 CLIP 特征空间里离目标风格参考更远。

---

## 2. Per-image 指标分布

### 2.1 AdaIN s=0.5

| 指标 | 平均变化 | 改善张数 | 恶化张数 | 说明 |
|------|----------|----------|----------|------|
| CLIP-S | -0.01371 | 262 | 488 | 整体风格相似度下降 |
| LPIPS | -0.01552 | 613 | 137 | 内容保持整体提升 |

`corr(ΔCLIP-S, ΔLPIPS) = +0.164`：
LPIPS 改善时，CLIP-S 倾向于同步下降，但相关性较弱，说明两者衡量的不是同一回事。

### 2.2 按目标风格拆分（AdaIN s=0.5）

| 目标风格 | dCLIP-S | dLPIPS | 现象 |
|----------|---------|--------|------|
| Early_Renaissance | **+0.0122** | -0.0269 | 唯一 CLIP-S 提升的风格 |
| Impressionism | -0.0048 | -0.0013 | 几乎持平 |
| Minimalism | **-0.0341** | -0.0162 | CLIP-S 下降最重 |
| Rococo | -0.0242 | -0.0101 | 明显下降 |
| Ukiyo_e | -0.0176 | -0.0231 | 内容改善但风格分降 |

**Early_Renaissance 目标是唯一赢家**：后处理让生成图更接近该风格参考分布，CLIP-S 和 LPIPS 双赢。
**Minimalism 目标是最大输家**：identity 或跨风格迁移到 Minimalism 时，后处理把白色/浅灰内容强行压暗，CLIP-S 暴跌。

---

## 3. 视觉案例与像素统计

拉取了 4 组典型样本，见 `results/D5-512/_pp_visual_samples/`。

### Case 1：Impressionism → Minimalism（LPIPS 大改善，CLIP-S 下降）

| | 图像 | 均值 [R,G,B] | 标准差 | 亮度 |
|---|---|---|---|---|
| baseline | ![](file:///g:/GitHub/Latent_Style/WEAVE/results/D5-512/_pp_visual_samples/case1_baseline.png) | [0.746, 0.716, 0.662] | 0.104 | **0.719** |
| AdaIN s=0.5 | ![](file:///g:/GitHub/Latent_Style/WEAVE/results/D5-512/_pp_visual_samples/case1_affine_s05.png) | [0.685, 0.642, 0.565] | 0.206 | **0.646** |
| Minimalism 参考 | — | [0.639, 0.586, 0.495] | 0.339 | **0.583** |

**观察**：
- baseline 严重发白、对比度低（std≈0.10）。
- 后处理让画面变暗、对比度恢复，视觉上看更像“风格迁移结果”。
- 但色调偏暖/偏红，而 Minimalism 参考图的中性色占比更高；CLIP 认为它“不够 Minimalism”。
- LPIPS 改善是因为像素结构更清晰，但 CLIP-S 下降是因为颜色分布偏离了目标参考。

### Case 2：Impressionism → Early_Renaissance（双指标都改善）

| | 图像 | 均值 [R,G,B] | 标准差 | 亮度 |
|---|---|---|---|---|
| baseline | ![](file:///g:/GitHub/Latent_Style/WEAVE/results/D5-512/_pp_visual_samples/case2_baseline.png) | [0.762, 0.727, 0.678] | 0.119 | **0.731** |
| AdaIN s=0.5 | ![](file:///g:/GitHub/Latent_Style/WEAVE/results/D5-512/_pp_visual_samples/case2_affine_s05.png) | [0.642, 0.578, 0.520] | 0.191 | **0.590** |
| Early_Renaissance 参考 | — | [0.526, 0.433, 0.367] | 0.268 | **0.464** |

**观察**：
- Early_Renaissance 参考图整体偏暗、对比度高。
- 后处理把 baseline 从“洗白”状态压暗到接近参考亮度，视觉和指标都受益。
- 这是后处理最有效的场景：baseline 白化方向恰好与目标风格相反。

### Case 3：Minimalism → Minimalism（CLIP-S 暴跌，LPIPS 也恶化）

| | 图像 | 均值 [R,G,B] | 标准差 | 亮度 |
|---|---|---|---|---|
| baseline | ![](file:///g:/GitHub/Latent_Style/WEAVE/results/D5-512/_pp_visual_samples/case3_baseline.png) | [0.892, 0.881, 0.869] | 0.018 | **0.883** |
| AdaIN s=0.5 | ![](file:///g:/GitHub/Latent_Style/WEAVE/results/D5-512/_pp_visual_samples/case3_affine_s05.png) | [0.759, 0.730, 0.680] | 0.167 | **0.733** |
| Minimalism 参考 | — | [0.639, 0.586, 0.495] | 0.339 | **0.583** |

**观察**：
- 源图本身是“白色碗”极简作品，baseline 基本保持了这个外观（亮度 0.883）。
- 后处理按 Minimalism 全参考集的均值把它压暗到 0.733，背景变灰、出现色偏。
- 对于**这个特定 content**，目标参考里与它最接近的本来就是同样偏白的 Minimalism 作品；向全风格均值回归反而把它推离了正确参考。
- CLIP-S 从 0.91+ 暴跌约 0.17，LPIPS 也大幅恶化。

### Case 4：Early_Renaissance → Early_Renaissance（CLIP-S 改善）

| | 图像 | 均值 [R,G,B] | 标准差 | 亮度 |
|---|---|---|---|---|
| baseline | ![](file:///g:/GitHub/Latent_Style/WEAVE/results/D5-512/_pp_visual_samples/case4_baseline.png) | [0.719, 0.657, 0.606] | 0.108 | **0.669** |
| AdaIN s=0.5 | ![](file:///g:/GitHub/Latent_Style/WEAVE/results/D5-512/_pp_visual_samples/case4_affine_s05.png) | [0.620, 0.543, 0.484] | 0.184 | **0.559** |
| Early_Renaissance 参考 | — | [0.526, 0.433, 0.367] | 0.268 | **0.464** |

**观察**：
- baseline 仍然发白；后处理压暗并提升对比度。
- 目标 Early_Renaissance 参考本身偏暗，所以向参考对齐改善了 CLIP-S。

---

## 4. 根因分析

### 4.1 CLIP-S 不是“视觉质量”指标，而是“与参考分布的相似度”指标

CLIP-S 把生成图与目标风格的参考图分别送入 CLIP，取最大余弦相似度。它关心的是：
**生成图在 CLIP 嵌入空间里最接近哪一张目标参考图。**

如果后处理让图像变得更“自然”但偏离了参考分布的中心，CLIP-S 就会下降。

### 4.2 WEAVE baseline 的白化是一种系统性的亮度/对比度偏移

从样本统计可见，baseline 普遍：
- 亮度比目标参考高 0.1~0.3。
- 标准差（对比度）比目标参考低 30%~50%。

这种“洗白”可能是 VAE 解码、训练损失或 endpoint AdaIN 的副作用。

### 4.3 后处理在向“风格均值”回归，但均值不代表每个目标样本

后处理使用的参考统计量是整个目标风格测试集的平均：

```
μ_target = mean(all reference images of target style)
σ_target = std(all reference images of target style)
```

这带来两个问题：

1. **Identity 场景（src_style == tgt_style）**：
   源图本身就应该接近某张目标参考图。向全风格均值回归会把它拉向“平均脸”，反而远离真实匹配参考。
   Case 3 的白色碗被压暗，就是典型。

2. **跨风格迁移到 Minimalism**：
   Minimalism 参考集内部差异很大（std 高达 0.34），包含白色、灰色、黑色、彩色各种作品。
   向均值对齐可能让结果过饱和或过暖，偏离真正的目标参考。

### 4.4 AdaIN 的方差项会过增强

当前 AdaIN 同时调整均值和方差：

```
x_out = (x - μ_x) / σ_x * σ_target + μ_target
```

方差项会把低对比度的 baseline 强行拉伸到目标对比度。对于已经高饱和的色块，这会导致：
- 颜色过饱和。
- 噪声/伪影被放大。
- 局部色调向参考均值偏移，破坏原本的内容结构。

这正是 Case 1 中草地变成橙红色的原因。

### 4.5 DINO-S 为什么几乎不变？

DINOv2 特征对全局亮度、颜色均值相对不敏感，更关注局部结构和纹理。
后处理主要改变的是全局统计量，对局部结构影响小，所以 DINO-S 变化在 0.003 以内。

---

## 5. 为什么“亮度对齐”单独做可能更好

亮度对齐只调整均值：

```
x_out = x + strength * (μ_target - μ_x)
```

它不会放大噪声，也不会引入过饱和。预期效果：

- 对 Early_Renaissance / Rococo 等偏暗目标：压暗 baseline，改善视觉和 CLIP-S。
- 对 Minimalism identity：不会把白色碗强行压到灰褐色（因为没有方差拉伸）。
- 对跨风格到 Minimalism：只校正亮度，不引入额外色偏。

**建议下一步**：单独跑亮度对齐（`--std_strength 0.0`），看是否能保留 LPIPS 改善的同时减少 CLIP-S 下降。

---

## 6. 结论与建议

1. **后处理改善了视觉白化，但损害了 CLIP-S，因为 CLIP-S 衡量的是与目标参考分布的匹配，而非人眼舒适度。**
2. **Early_Renaissance 是后处理唯一双赢的目标风格**；Minimalism identity 是最大陷阱。
3. **AdaIN 的方差项是主要问题来源**：它过增强对比度和饱和度，导致色偏。
4. **推荐用纯亮度对齐（均值-only）替代完整 AdaIN**，并降低 strength（如 0.3），在保留视觉改善的同时减小对参考分布的偏离。
5. **根本解决仍需在生成端**：调整 endpoint_adain_scale、ll_partial_style_enabled / ll_partial_alpha，让训练/推理本身少产生白化。

---

## 7. 附件

- 样例图片：`results/D5-512/_pp_visual_samples/`
- per-image 分析日志：`results/D5-512/_pp_analysis_log.txt`
- 目标风格参考统计：`results/D5-512/_ref_stats.json`
