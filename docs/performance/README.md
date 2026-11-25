# docs/performance/ — WEAVE 统一性能台账、实验总结与可视化中心

本目录是全仓库**唯一的跨代性能数据与图表权威入口**，整合了原 `docs/metrics_750.*`、`docs/figures_750/` 与各阶段 `eval.json`，覆盖 **V6/Q2A → V8 → V9-LD → V10 → V11B → V15（Group A / Scaling / Style-Bridge）→ V16 → V17 → V18** 及 **19 个外部基线（含复现的 StyleGallery 与 Hist2Style）** 共 **200+ 个实验臂**。

## 核心文件索引

| 文件 / 目录 | 说明 |
|---|---|
| [**`MASTER_LEDGER.md`**](MASTER_LEDGER.md) | **实验结论与全量数据总台账（必读）**：六条方法论铁律、各阶段核心结论、帕累托前沿总表、外部基线全表、V6–V18 分代全量数据表、证伪路线登记表 |
| `metrics.csv` / `metrics.json` | 统一机读主表（25 列：`dino_s`, `dino_c`, `lpips`, `clip_s`, `style_affinity`, `target_top1`, `target_margin`, `dT`, `G`, `same_*`, `ms_per_img`, `artfid`, `kid`, `pareto`, `protocol`, `decoder`, `eval_source`） |
| `build_metrics.py` | 零 GPU 台账自动聚合与帕累托计算脚本（`tools/export_metrics_750.py` 已统一委托至此入口） |
| `figures/` | 全套矢量（`.svg` / `.pdf`）与位图（`.png`）可视化：<br>• `pareto_frontier_cross600.{svg,png}` — 跨代帕累托前沿双联图（CSD Top-1 vs LPIPS / DINO-S vs DINO-C）<br>• `correlation_heatmap.{svg,png}` — 12 维指标 Spearman 等级相关矩阵<br>• `pairs_grid.{svg,png}` — 6 组核心指标对散点矩阵<br>• `artfid_landscape.{svg,png}` / `artfid_ranking.{svg,png}` — ArtFID 分布景观与排序 |
| `images/` | 代表性基线与前沿模型的 750 任务输出图库（`baselines_D5_512` 17 方法 × 750、`WEAVE_published_reval` 750、`V15_W_dim96_conv_base` 2250、`B_adapter` 750；体积 ~2.8GB，已纳入 `.gitignore`） |

## 一键重建命令（零 GPU，纯本地秒级）

```bash
python docs/performance/build_metrics.py
python tools/plot_metrics_750.py --vector
```

## 跨协议比较红线

1. **`q2a_oracle_cross600`（现行主协议）**：`tasks_750.json` Cross600 主读数 + Same150 恒等控制，`S10` 解码器（`s10_p5_lowlr/decoder.pt`），冻结 `src/eval/measure.py` 四裁判（CSD-ViT-L / DINOv2-small / CLIP-ViT-B/32 / LPIPS）。
2. **`v8_sdvae_cross600`（V8 历史口径）**：V8 41 臂使用原版 SD-VAE 解码器评分，仅限 V8 内部机制对比，不与 S10 口径直接混排（同模型换 S10 平均 Top-1 +3.2pp、DINO-S −0.010）。
3. **诊断控制臂隔离**：`Identity`、`SD-Turbo`、`*_wrong`（错标签控制）、`*_swap_*`（强制单标签探针）以及 `t0=0.0`（无源图锚点的纯噪声无条件生成）均标记 `is_diagnostic=True`，不参与跨风格迁移帕累托前沿判定。

`Pareto` 图的外部名单、非支配计算与图例规范见 [pareto.md](pareto.md)。
