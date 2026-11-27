# Conditioning｜选择性信息过滤

> 公共内容、风格与路由模块。E 和 P 共享布局表征，但各自的结构注入与训练目标不同。核心结论不是“去掉所有风格信息”，而是阻断错误的二阶统计通路，同时保留对象关系。

## 1. 当前数据流

| 条件 | 生成方法 | 责任 |
|---|---|---|
| A0 latent | 冻结 SD-VAE Encoder，posterior mean ×0.18215 | 同一潜空间 |
| c64 | RGB 结构/局部对比度 | E 的高分辨率结构 |
| Y16 LCN | latent 亮度线性投影+局部对比度 | P 粗亮度布局 |
| Q32 | 256px gray3→ConvNeXt-Tiny Stage3→384×16×16→32维投影→极分解 | 去二阶泄漏的空间布局 |
| K8/417D | 每画派8场景子模态原型；场景匹配后选择 | 离散画法多模态 |
| tid | 5类目标 one-hot/ID | 闭集画派控制 |

冻结 ConvNeXt-S3（12.35M）替代 DINOv2；DINOv2/CSD/CLIP/LPIPS 仅作独立裁判，不用于当前生成条件、训练损失或逐图筛选。Zero-DINO 不意味着没有任何预训练视觉模型。

## 2. Stiefel：精确删除二阶通道相关

对经过同一冻结投影和空间预处理的特征 Z∈R^(32×256)，薄 SVD Z=UΣV^T，定义 Q=16UV^T。满行秩且数值计算正常时，QQ^T/256=I。逐通道白化只固定对角项，不能去除非对角协方差。Stiefel 不保证全部高阶统计、空间排列或类别语义独立于画法。

| V35 P3 探针 | Gram 5-way acc | Spatial acc |
|---|---:|---:|
| ConvNeXt-S3 Raw 384D | 91.33% | — |
| PCA-32 | 95.00% | 82.00% |
| 对角白化32 | 95.67% | 81.33% |
| Stiefel Q32 | **20.67%** | 76.33% |

Gram 接近20%随机，不代表空间风格泄漏为零。生产正交误差 V32 报1.95e-4。SVD/统计默认 FP32；近似替代必须检查正交误差、rank、小奇异值及生成质量。

## 3. 从表征探针到生成因果

同构44M、相同数据及3500步，P3/P4 双种子只改变归一化：

| 变体 | s100 C/S/Top-1 | S @C=.50/.61/.70 |
|---|---|---|
| Raw PCA-32 | .5960/.5374/63.00% | .5593/.5328/.5013 |
| 普通白化 | .5927/.5417/64.17% | .5660/.5355/.5013 |
| Stiefel | .5628/.5570/69.00% | .5708/.5388/.5010 |

相对普通白化：seed1 s100 +.0153 S/+4.83pp Top-1；seed2 +.0159/+6.67pp。等 C 的 S 增益是+.0048/+.0033/−.0003；所以真实生成收益集中在中强风格区，不能用固定旋钮的收益冒充等内容前沿收益。

44M Q32 六臂推理诊断：s085 正确 Q32 C=.6734，zero=.6340，shuffled=.6240；tau0=0 正确 C=.4155，zero=.2515，Same90.7→60.0%。说明删掉全部语义会使内容坍塌；真正有效的是选择性删除，不是最小化信息量。

## 4. 风格条件：多模态重要，更多维度不是答案

历史 V22 冻结权重中 K1/K4/K8 Top-1 46.0/60.8/61.5%；K8+Core-kNN 61.5%，随机实例61.0%。这支持保留多个目标模式，不证明 K=8 是普适最优。当前保留 K8、417-D 统计，目标标签与路由由训练侧先验提供；无需测试参考图的空间 Value。

- V26 局部 Value 注入污染内容，移除 V 后 C 约+.050；不否定所有局部性，但不重试该路径。
- V33 Q64+ 与32×32 Adapter 提高固定起点内容，却未提升等 C 风格前沿；V34 同容量联合训练仍失败。
- V34 连续417/1101-D混合参考响应提高4–4.8×，但同 C 前沿低约.0128 S。凸组合可能平均掉子模态对比，故保留离散原型。

开放域参考条件另见 [One-shot](../one-shot/README.md)，不能把闭集 K8 的性能等价为任意参考迁移。

## 5. 资产与源码

W_stiefel_cnx（384×32）来自 runs/v32_nodino/cache/cnx_stiefel_cache.pt；K8 来自 k8_cnx417_bank.pt；Y16 投影来自 runs/v18/cache/ccf_pool_cache.pt。当前 loader 仍需大缓存，部署包提取须先通过同输入 parity；训练缓存与 PCA 拟合数据要冷存。见 [cleanup](../../cleanup.md) 与 [System profile](../system/3060_PROFILE_OPTIMIZATION.md)。

原始证据：[V21 Stiefel](../archive/historical/next1118/05_V21_STIEFEL_ONESHOT_THEORY.md) · [V22](../archive/historical/v22/V22_P1_RESULTS.md) · [V32](../archive/historical/v32/NODINO_REPORT.md) · [V33](../archive/historical/v33/V33_SEMANTIC_CONTROL_REPORT.md) · [V34](../archive/historical/v34/V34_FRONTIER_REPORT.md) · [P3](../archive/historical/v35/data/p3_stiefel_poolc_report.json) · [P4](../archive/historical/v35/data/p4_seed2_report.json)。V32 特征提取/缓存源码当前 main 未齐全，实际运行环境必须归档。
