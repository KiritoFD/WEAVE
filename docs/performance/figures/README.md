# metrics_750 跨指标可视化

数据源：[`metrics.csv`](../metrics.csv)（207+ 行 × 25 列）。全部从 CSV 读取，**零重算、零 GPU**。
脚本：[`../../../tools/plot_metrics_750.py`](../../../tools/plot_metrics_750.py) 与 [`../plot_master_suite.py`](../plot_master_suite.py)，重跑：

```bash
python tools/plot_metrics_750.py            # PNG only
python tools/plot_metrics_750.py --vector  # 额外 SVG + PDF
```

所有图**固定画布**：不用 `bbox_inches='tight'`，标题/图例/脚注全部在画布内，
没有越界文字撑出的大白边。配色沿用 `docs/v15` 的蓝(外部)/橙(本工作)/紫(V15)体系。

| 图 | 内容 |
| --- | --- |
| [correlation_heatmap.png](correlation_heatmap.png) | 12 个指标两两 Spearman ρ（上三角）+ 成对有效样本数 n（下三角）。↓ 指标在算 ρ 前取负号，故 ρ>0 恒表示"两个指标偏好同一臂"。 |
| [pairs_grid.png](pairs_grid.png) | 6 组关键指标对散点，每点一臂，按 4 个系列着色，虚线为最小二乘目视参考。 |
| [artfid_landscape.png](artfid_landscape.png) | 41 个同时有 DINO 与 ArtFID 的臂：位置 = DINO-C × DINO-S，颜色 = ArtFID（绿好红差）；灰点 = 只有 DINO 的臂；蓝环 = 外部 baseline 在 (content, DINO-S) 上的 Pareto 集。 |
| [artfid_ranking.png](artfid_ranking.png) | 44 臂 ArtFID 升序条形图，`*` = 图集不完整（Seedream 缺 30、StyleShot 缺 5）。 |

## 读图要点

- **风格簇内部高度共线**：DINO-S / Style Affinity ρ=0.98，CLIP-S / Affinity 0.96，DINO-S / CLIP-S 0.88；CSD Top-1 / Margin 0.95。
- **内容簇内部高度共线**：DINO-C / LPIPS 0.81（取负后），DINO-C / same_dino_c 0.99。
- **风格与内容是对立轴**：DINO-S / DINO-C ρ=−0.62（n=143，覆盖最全的一对）。
- **ArtFID 是第三根轴**：与 KID 0.90、与 DINO-C 0.77（内容越好分布越像），
  但与 DINO-S −0.16、与 Style Affinity +0.09 —— **几乎不排序风格能力**；
  与 ms/img 0.01，与效率也无关。即 ArtFID 主要惩罚内容破坏/分布漂移，
  不能单独当作风格迁移质量结论。
- **覆盖度必须看 n**：ArtFID 44、Style Affinity 63、LPIPS/CLIP-S 95、DINO 143、
  same_top1 仅 16。每个相关系数只用该两列都非空的臂，未做任何插补。

## 口径说明

- ρ 用 `Series.rank().corr()`（Spearman），`LOWER = {lpips, artfid, kid, ms_per_img}`
  在计算前取负；**坐标轴永远显示原始值**，箭头只标"越大越好/越小越好"。
- `ms_per_img` 是效率轴不是质量轴，纳入热图仅为呈现其与各质量指标的相关性，
  不参与任何"综合质量"排序。
- ArtFID = (1+FID)(1+LPIPS) 对 5 个目标画派取均值；KID 为 unbiased poly-kernel MMD。
- ArtFID 与 DINO 同时存在的 41 臂之外，3 个 `WEAVE.D5.*` 变体仅有 ArtFID，
  故出现在 ranking 而不在 landscape。

## 附：旧图修复

[`../../v15/plot_v15_dino_artfid.py`](../../v15/plot_v15_dino_artfid.py) 原先把 44 行
ArtFID/KID 明细表用 `fig.text` 堆在画布外，配合 `bbox_inches='tight'` 把
`dino_artfid_v15_zoom.png` 撑到 13008 px 宽。已改为图内一行摘要 + 固定画布，
重画后两版均为 3588×2116。完整明细改由 `metrics_750.csv` 承载。
