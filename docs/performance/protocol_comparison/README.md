# Cross600 与 All750：现役 CLEAVE 三联图对照

2026-11-28。只统计已有评分，不重新生成图片或运行裁判。Ours只保留现役CLEAVE-E/P：P用朱红实线、圆点，E用绿色实线、菱形，各自连接当前权重的全部实测档位；深色虚线连接两条曲线共同的实测非支配点。两条核曲线分别绘制，共同前沿是独立的包络表示。图内不画历史ours、One-shot、实验E、消融、错目标或未验收TRT候选。

## 两套三联图

- **Cross600**：[SVG](figures/cross600_triptych.svg) · [PNG](figures/cross600_triptych.png)
- **All750**：[SVG](figures/all750_triptych.svg) · [PNG](figures/all750_triptych.png)
- [同页对照](figures/comparison.html) · [聚合数据JSON](comparison.json) · [完整统计CSV](metrics.csv)

每套依次为DINO-C–DINO-S、DINO-C–CSD Top-1、吞吐–对数均衡综合得分 $\log C + \tfrac{1}{2}(\log\text{DINO-S} + \log\text{CSD Top-1})$（与前两面板各自补充的 Identity 参考底线：`DINO-S` 与 `CSD Top-1` `0.5%`/`20.0%`）。两套图使用相同模型、输出、颜色与自适应边界缩放，只改变任务聚合。E/P的旋钮不是新模型，不再混入旧权重。

口径、标题、紧凑图例和简短覆盖/测速说明全部放在子图内部。完整协议说明留在本文。第三栏按各自口径下的对数综合得分，为P、E各选一个最高实测得分点，标明对应旋钮；历史吞吐记录不变。Cross600选P s=.75、E λ=.90；All750选P s=.75、E λ=.65。

左栏将SaMST作为离群点标在图外左下角，并写出真实DINO-C/S坐标；它不参与该栏坐标范围计算，原始评分仍保存在JSON/CSV中，其余两栏照常显示。标签按渲染后的实际文字边框避让，必要时添加细引导线。

## 哪种口径对我们更有利

**DINO质量图上All750更有利；等内容Top-1优势则是Cross600略大。不存在三个面板一起全面变好的结论。**

以各自口径下StyleGallery的实测DINO-C为锚，对现役E/P各条曲线分别在相邻实测点内插值，取其中更好的估计：

| 量 | Cross600 | All750 |
|---|---:|---:|
| StyleGallery内容锚DINO-C | .6113 | .6304 |
| StyleGallery DINO-S | .5604 | .5797 |
| Ours等内容DINO-S，插值 | .5394 | .5645 |
| Ours−StyleGallery DINO-S | **−.0210** | **−.0153** |
| StyleGallery Top-1 | 52.50% | 61.20% |
| Ours等内容Top-1，插值 | 62.52% | 70.60% |
| Ours−StyleGallery Top-1 | **+10.02pp** | **+9.40pp** |

All750使DINO-S差距缩小约.0057；在现有有效样本均值点上，还新增了对StyleShot的DINO-C/S双轴支配。后者StyleShot只有745条有效任务，不能写成完整750的严格配对结论。C/Top-1面板中被ours实测点支配的公开基线名单未变。

Top-1绝对数字普遍上升不是模型变强：完整任务集的`All750 = .8×Cross600 + .2×Same150`。例如P s=1.00从69.67%到74.67%，StyleGallery也从52.50%到61.20%；只看ours的绝对涨幅会误判相对优势。

**主图建议保留Cross600，All750作为并列补充/附录。** Cross600直接对应跨风格迁移；All750更突出含同风格任务的整体表现，并对DINO前沿较有利。这里没有改动原主图口径或冻结论文文件。

## 数据来源与边界

- 基线按[docs/baseline首页](../../baseline/README.md)指向的`paper_rgb_v1`统一重评汇总读取，补入该目录的StyleGallery与Hist2Style逐任务评分，以及完整750任务的`Qwen-Image-2.1-Turbo`（[eval.json](../../baseline/QwenImage21Turbo/eval/eval.json)，600 cross + 150 same）。14个第三方/商业方法的当前聚合快照见[baseline聚合JSON](../../baseline/evaluation/paper_rgb_v1_aggregates.json)。Identity为控制、旧WEAVE为内部历史，不进入这张主图。
- 现役ours：P8最终P路径10档与P9原生E路径9档；不使用λ=0恒等档。原逐任务评分与SHA已汇入本地统计缓存，聚合结果及来源记录保存在`comparison.json`，不是从旧ours摘要补造All750。
- StyleShot为596 cross/149 same=745；Seedream为576/144=720。均为各自有效任务均值，以†标识，不填补缺失任务，不伪装成完整600/750。
- 每个二维面板单独计算非支配集合。外部文献前沿不含商业Seedream；SD-Turbo只显示为参照点，不改变仓库的文献来源白名单。
- 吞吐沿用已有时间元数据并在两图间保持不变：ours为4090已加载RGB→RGB B16（P14.3、E33.1img/s）；外部方法的记录条件不同。不能据此声称process-cold或严格同硬件加速比。缺少时间记录的方法只显示在前两个质量面板。
- 等内容结果为插值均值，不是新增实测，也未在本次计算新的配对CI。DINO-C/S与Top-1的最佳插值可能对应不同旋钮，不拼成一个不存在的联合工作点。

## 复现

从仓库根执行，默认用已归档聚合快照重画，不依赖GPU或远端：

```powershell
python docs/performance/plot_cleave_pareto.py --compare-aggregation
```

若本地已有原逐任务缓存，可加`--refresh`重新聚合；不自动启动模型评测。输出保留SVG与PNG，两图使用相同的自适应边距规则。
