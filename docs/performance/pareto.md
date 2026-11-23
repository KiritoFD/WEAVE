# Pareto 画图口径

**外部文献前沿只使用独立于本项目的公开研究方法。** 按方法来源判定。历史表里的 `Baseline` 与 `Published` 标签不能决定资格。WEAVE 原版、V6–V27、PRISM-1、Flow-PnP、R12、HiC 和 Unified 都属于内部结果。

## 数据分组

| 分组 | 入选规则 | 图上表示 |
| --- | --- | --- |
| 外部公开文献 | 独立第三方公开研究方法。方法名进入显式名单，指标可追溯到同一评测协议下的来源和样本数。 | 外部散点。灰色或蓝色虚线只连接该组的二维非支配点。 |
| 本轮 WEAVE | 当前模型、旋钮档位及本轮消融。 | 每个模型按档位连实线，单独标明配置。 |
| 历史 WEAVE | 原 WEAVE、旧 PRISM 或其他本项目工作点。论文发表状态不改变归属。 | 用独立标记标作内部历史参照。内部前沿如需展示，单独计算和标注。 |
| 其他参照 | 商业产品、诊断任务、来源未核实或口径不一致的数值。 | 如需展示，用独立参照标记。缺指标或不同比口径的点留在表中。 |

当前名单在 [方法来源判定](../../src/eval/pareto.py) 中维护。公开文献提供方法出处，图上坐标来自本仓库 [指标台账](metrics.csv) 的同协议复评。新增方法前记录论文和复评数据来源。已有条目可核对项目 [文献库](../icme2027/refs.bib) 以及 [StyleGallery 论文](https://openaccess.thecvf.com/content/CVPR2026/html/He_StyleGallery_Training-free_and_Semantic-aware_Personalized_Style_Transfer_from_Arbitrary_Image_CVPR_2026_paper.html)、[SaMam 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_SaMam_Style-aware_State_Space_Model_for_Arbitrary_Image_Style_Transfer_CVPR_2025_paper.html) 和 [SaMST 论文](https://arxiv.org/abs/2503.20368)。

## 计算和绘制

1. 每张图先选任务集、目标设置、裁判版本和指标定义。当前质量图使用 750 任务中的 Cross600，要求每臂完整 600 条。闭集标签驱动和任意参考的结果并列画时，要在图例或图注写明设置，比较范围限于所选指标。
2. 先按来源过滤，再在每个子图里单独求非支配集合。DINO-C、DINO-S、CSD Top-1、吞吐量越大越好，LPIPS、延迟越小越好。如果一个点在两轴上都不差、至少一轴严格更好，它支配另一个点。坐标完全相同的点可共用一个包络顶点。两轴前沿从实测值计算，和台账历史四轴 `pareto` 列用途不同。
3. 外部虚线仅连接已观测的外部非支配点，不向观测范围外延伸。WEAVE 旋钮曲线连接实测档位，线段之间没有额外的实测工作点。恒等、错标签和 Same150 控制另行报告。
4. 报告“超越前沿”时，写清所用轴、工作点和相同内容或风格水位。二维图只支持对该二维指标的判断。速度图按硬件、批量、分辨率和缓存或 RGB-to-RGB 口径单独画。当前 4090 新结果不沿用历史 3060 速度。估计耗时和实测耗时分别标注。

## 复现入口

当前 PRISM-1-Unified 与外部文献的对照图来自以下命令。历史轨迹同图保留为内部参照。[图与结果](../exp1123/single_q32/unified/README.md)

```bash
python -m exp.v27_single.plot_flow_pnp docs/exp1123/single_q32/flow_pnp/results.json --prism docs/exp1123/single_q32/unified/results_s4000.json --editor-label PRISM-1-Unified
python docs/performance/build_metrics.py --plot-existing-ledger
python -m exp.v25.plot_ours_pareto
python tools/plot_pareto_frontier.py
python -m exp.v28_unified.plot_speed_score
```

## 均衡档位综合得分与推理速度图

线性组合（如 `DINO-C + DINO-S`）会在无编辑端点（`λ=0`, `C=0.959, S=0.3735`）或内容坍缩端点达到极值，压低均衡工作点。提供两类简洁、无门槛的综合指标（定义于 [pareto.py](../../src/eval/pareto.py)）：

1. **负对数平滑最小值** `composite_dino_lse`：$S_{\mathrm{LSE}}(C, S) = -\log\!\big(e^{-10 C} + e^{-15 S}\big)$，以 `2:3` 整数比对齐 `DINO-C (~0.70)` 与 `DINO-S (~0.47)` 的自然量级。Zoom-in 主图 [速度-负对数得分图](figures/speed_composite_score.png) 聚焦 `[5.20, 6.48]` 上簇区间（剔除下方离群点 `StyleID=3.86`, `StyleShot=3.56`, `SaMST=1.55`, `WCT=0.11` 以拉开纵坐标间距）。
2. **零参数负幂广义均值（无对数）** `composite_dino_power_mean`：$M_{-4}(C, S) = \big(\tfrac{1}{2}(C^{-4} + S^{-4})\big)^{-1/4} \in [0, 1]$，完全对称且无需任何超参或对数变换，在 [速度-负幂均值图](figures/speed_power_mean.png) 中使 `Seedream 4.5` 升至外部第 4（`0.4534`）、`λ=0` 垫底（`0.4417`）。

台账绘图命令读取现有 CSV，保留原始数据。V25 默认输出到性能图目录。冻结稿的图单独保管，需要明确指定 `--paper-figures` 才写入论文图目录。
