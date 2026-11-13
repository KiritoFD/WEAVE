# 像素空间后处理（亮度/AdaIN/WCT）效果报告

## 1. 背景与动机

在 D5-512 方法对比 teaser（`results/D5-512/teaser_d5_methods.png`）中可以直观看到：
**WEAVE 输出存在明显白化现象**——同一张 content/style 组合下，WEAVE 结果比 SaMam/SaMST 等基线更亮、更灰白。
这提示生成端点（endpoint）在亮度/对比度统计量上没有完全对齐目标风格。

为了量化并缓解这一现象，我们在**不重新跑生成模型**的前提下，在已生成的图像上实现了像素空间后处理：

- **亮度对齐（Brightness Alignment）**：通道均值对齐，仅调整生成图的整体亮度，不改变对比度/色度。
- **AdaIN（`style_rgb_affine`）**：通道级均值+方差对齐，可视为亮度对齐的扩展。
- **WCT（`style_rgb_wct`）**：完整 3×3 协方差 Whitening-Coloring Transform，对齐二阶统计量。

实现脚本：`exp/eval_all/apply_postprocess_to_images.py`

## 2. 实现方法

### 2.1 亮度对齐（Brightness / Mean-only AdaIN）

对生成图 `x` 和目标风格参考图集合 `R`：

```
μ_x     = mean(x)                       # 生成图通道均值
μ_R     = mean(R)                       # 目标风格参考图通道均值
x_mean  = x + strength * (μ_R - μ_x)    # 亮度对齐
```

在代码中对应 `apply_rgb_affine(..., mean_strength=1.0, std_strength=0.0)`，即 AdaIN 的方差项置零。

### 2.2 AdaIN（Mean + Std）

```
σ_x     = std(x)
σ_R     = std(R)
x_adain = (x - μ_x) / σ_x * σ_R + μ_R
x_out   = x + strength * (x_adain - x)
```

`strength=0.5` 表示只把生成图向目标风格推一半，保留更多内容结构。

### 2.3 WCT

通过特征值分解把生成图协方差矩阵白化后，再染上目标风格协方差矩阵的颜色：

```
Σ_x = cov(x)        # 生成图 3×3 协方差
Σ_R = cov(R)        # 目标风格 3×3 协方差
x_wct = Σ_R^(1/2) * Σ_x^(-1/2) * (x - μ_x) + μ_R
x_out = x + strength * (x_wct - x)
```

### 2.4 参考统计量计算

参考图按风格子目录分别聚合，最多取前 64 张（`--ref_limit 64`）。
为避免参考图尺寸不一致导致 `torch.stack` 失败，`load_reference_stats_affine` 采用**在线累积像素级和与平方和**的方式计算均值/方差，支持任意尺寸图像。

## 3. 实验设置

- **基线模型**：WEAVE with `structure_aligned_target=True`（align-tgt），checkpoint `runs/submission/canonical_oriented_epoch4/epoch_0004.pt`
- **数据集**：D5-512（750 张，5 风格）
- **评估指标**：CLIP-S、LPIPS、DINO-S、DINO-C、DINO-Structure、WFI
- **后处理强度**：AdaIN / WCT，各 0.5 和 1.0 两档
- **运行环境**：远程 RTX 3060 12GB；脚本 `exp/eval_all/run_postprocess_d5.bat`

## 4. 实验结果

### 4.1 align-tgt 基线（三数据集，已评估完成）

| 数据集     | n   | CLIP-S | 1-LPIPS | DINO-C | DINO-S | DINO-Struct | WFI    |
|------------|-----|--------|---------|--------|--------|-------------|--------|
| D5-512     | 750 | 0.7128 | 0.7406  | 0.8105 | 0.4915 | 0.0251      | 0.6483 |
| P2A-256    | 750 | 0.6491 | 0.7121  | 0.8971 | 0.4631 | 0.0096      | 0.6237 |
| R5-WikiArt | 750 | 0.7798 | 0.7679  | 0.8560 | 0.5725 | 0.0227      | 0.6352 |

*注：1-LPIPS = 1 - LPIPS，即内容保持分数。*

### 4.2 D5-512 后处理对比

| 配置              | CLIP-S | LPIPS ↓ | DINO-S | DINO-C | DINO-Struct | WFI    |
|-------------------|--------|---------|--------|--------|-------------|--------|
| baseline (无后处理) | 0.7128 | 0.2594  | 0.4915 | 0.8105 | 0.0251      | 0.6483 |
| AdaIN s=0.5       | 0.7015 | **0.2439** | 0.4901 | 0.7974 | 0.0251      | —      |
| AdaIN s=1.0       | 0.6944 | 0.2533  | **0.4931** | 0.7908 | 0.0251      | —      |
| WCT s=0.5         | 0.7020 | 0.2711  | 0.4913 | 0.7927 | 0.0251      | —      |
| WCT s=1.0         | 0.6915 | 0.2976  | 0.4913 | 0.7759 | 0.0251      | —      |

*注：CLIP-S 越高越好；LPIPS 越低表示内容保持越好；DINO-S 越高表示风格越接近目标；DINO-C 越高表示内容保留越好。*

## 5. 结论：后处理是否提升了整体性能？

**整体上没有显著提升，但 AdaIN s=0.5 在内容保持上有边际改善。**

逐项分析：

1. **风格指标（CLIP-S / DINO-S）**
   - AdaIN s=1.0 的 DINO-S 最高（0.4931），但相比 baseline 仅提升 +0.0032（+0.65%）。
   - 其他配置的 DINO-S 与 baseline 基本持平或略降。
   - CLIP-S 在所有后处理配置下均有下降（-0.010 ~ -0.021）。

2. **内容指标（LPIPS / DINO-C）**
   - **AdaIN s=0.5 的 LPIPS 从 0.2594 降到 0.2439（-6.0%）**，说明内容结构保持得更好。
   - 但 DINO-C 同步下降（0.8105 → 0.7974），意味着虽然像素级结构保持更好，但高层语义内容略有损失。
   - WCT 两个强度都导致 LPIPS 上升、DINO-C 下降，即内容损失更大。

3. **结构指标（DINO-Structure）**
   - 所有后处理配置下 DINO-Structure 均为 0.0251，与 baseline 完全一致。
   - 说明像素空间的颜色/亮度变换不影响结构自相似性。

4. **亮度对齐 vs 完整 AdaIN**
   - 当前实现的 AdaIN 包含均值+方差两项。
   - 若仅做亮度对齐（方差强度置 0），预期效果介于 baseline 和 AdaIN s=0.5 之间：
     亮度偏差被修正，但对比度不会向目标风格过度拉近，内容保持应优于完整 AdaIN。

5. **WFI（白化验收指标）**
   - 由于后处理降低了 CLIP-S 且对 DINO-S 提升有限，**WFI 不会显著改善**。
   - 从白化验收标准（WFI < 0.40）角度看，后处理没有帮助 crossing 该阈值。

## 6. 建议

- **不将 AdaIN/WCT 后处理作为默认链路**：指标提升微弱，且 CLIP-S 下降，不适合直接加入主表。
- **AdaIN s=0.5 可作为可选增强**：当任务更看重内容保持（低 LPIPS）而非绝对风格相似度时，可考虑使用。
- **优先在训练/推理端解决白化**：后处理只是“补救”，应继续优化 endpoint_adain_scale、ll_partial_style 等生成端参数，从源头减少亮度偏差。
- **如需纯亮度对齐**：可在 `apply_postprocess_to_images.py` 中调用 `style_rgb_affine` 并设置 `--std_strength 0.0`，实现零对比度损失的亮度修正。

## 7. 复现命令

```bash
# 亮度对齐（均值-only AdaIN）
python exp/eval_all/apply_postprocess_to_images.py \
    --input_dir results/D5-512/weave \
    --output_dir results/D5-512/weave_pp_brightness_s05 \
    --test_dir data/test \
    --mode style_rgb_affine \
    --strength 0.5 --mean_strength 1.0 --std_strength 0.0 \
    --style_subdirs "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e" \
    --device cuda --ref_limit 64

# 评估
python utils/run_evaluation.py results/D5-512/weave_pp_brightness_s05 \
    --test_dir data/test \
    --style_subdirs "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e" \
    --reuse_generated --batch_size 8
```

## 8. 原始结果位置

- 基线汇总：`I:\Github\Latent_Style\WEAVE\results\{D5-512,P2A-256,R5-WikiArt}\weave\`
- 后处理结果：`I:\Github\Latent_Style\WEAVE\results\D5-512\weave_pp_{affine,wct}_{s05,s10}\`
- 评估日志：各目录下 `_eval_log.txt`
