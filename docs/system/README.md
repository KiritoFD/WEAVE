# System｜CLEAVE 双核交付与证据总账

> V35 P8/P9：闭集五画派，Zero-DINO，纯前向。当前不再进行主线架构搜索。E/P 是两个独立训练的专门化模型，不是44M单核，也不要求每个请求执行两核。

**整体结果入口**：[2026-11-28 整体实验报告](OVERALL_REPORT.md)，统一列出闭集、One-shot、Cross600/All750及部署证据与限制。

## 1. 组件与运行路径

~~~text
RGB source -> A0 encoder + Conditioning
                       |
                       v
                  E-Core (11.13M)
                  /             \
     high-content                 A3 + P-Core (44.54M)
                  \             /
                    S10B (10.66M) -> RGB
~~~

| 模块 | 参数 | 权威说明 |
|---|---:|---|
| E | 11.13M | [单步编辑](../e-core/README.md) |
| P | 44.54M | [条件重绘/A3/CFG](../p-core/README.md) |
| A3 | 0 | P起点，不是额外模型 |
| S10B | 10.66M | [Decoder](../decoder/README.md) |
| A0 + ConvNeXt | 34.16M + 12.35M | [Conditioning](../conditioning/README.md) |
| 生成+解码 | 66.33M | E+P+S10 |
| 完整常驻 | 112.84M | 含公共编码器 |

E-only 高内容端生成只需11.13M E（另需公共编码器和 S10）；不加载 P 时活跃参数相应减少。P 路径用 E 草稿（s/tau不同，纯噪配置可跳过草稿的实际计算需实现并验收）。不要把生成参数、全链路参数和每步等价 NFE 混写。

## 2. 完整质量—效率谱

C/S 为 DINO-C/DINO-S Max，Top-1 为 CSD 画派判别。E 原生和 E+A3 分开；P8/P9a 的实测点如下。

| 路径 | 档位 | C | S | Top-1 |
|---|---|---:|---:|---:|
| E 原生 | l025 | .9166 | .3907 | 2.33% |
| E 原生 | l050 | .8141 | .4210 | 31.67% |
| E 原生 | l075 | .7204 | .4671 | 45.33% |
| E 原生 | l100 | .5675 | .5077 | 66.83% |
| E+A3 | l050 | .8086 | .4274 | 31.50% |
| E+A3 | l055 | .7792 | .4433 | 38.83% |
| E+A3 | l060 | .7684 | .4483 | 39.83% |
| E→P | s060 | .7813 | .4636 | 38.83% |
| E→P | s065 | .7443 | .4831 | 47.83% |
| E→P | s072 | .7133 | .5002 | 51.33% |
| E→P | s085 | .6514 | .5261 | 57.83% |
| E→P | s100 | .5502 | .5597 | 69.67% |
| E→P | s125 | .4678 | .5749 | 80.83% |
| E→P | s150 | .3435 | .5919 | 91.17% |

P9a 补齐高内容实测点，严格区间内插而非外推：

| C | V35 S | ΔS vs V32 | Top-1 |
|---|---:|---:|---:|
| .50 | .5690 | +.0092 | 76.47% |
| .61 | .5398 | +.0160 | 62.67% |
| .70 | .5053 | +.0245 | 53.57% |
| .75 | .4801 | +.0130 | 46.44% |

StyleGallery 历史 .6113 C/.5604 S/52.50% Top-1；CLEAVE 在同内容近邻领先约10pp分类，但 S 仍差约.021，闭集与任意参考任务亦不同。不能声称全面支配。极高内容区不能只画 P 曲线，参见[联合前沿](../performance/figures/cleave_pareto_triptych.svg)。

历史4090已加载未缓存 RGB→RGB：V32 E B1 41.68ms/B16 30.20ms/img；P8 完整 P B1 108.7ms/B16 69.82ms/img，峰值 allocated .83/6.52GiB。P8 B16 阶段计时混入 synchronize，名义6步 ledger 少计首步中点 NFE；不称进程冷启动。3060 双路径测速与质量门控已完成，见下方部署结果；该轮实测为7次四分支P调用，加E共29次条件等价评估。

## 3. 四条主要因果证据

| 主张 | 严格对照结果 | 边界 |
|---|---|---|
| Stiefel | Gram 95.67→20.67%；双种子 s100 +4.83/+6.67pp vs 白化 | 空间信息仍有泄漏，高内容等 C 收益很小 |
| E 路由 | s100 正确 P 条件下纠正草稿 +65.33pp；正确草稿下纠正 P +16pp | 半噪调度的简单效应，非全局必然 |
| A3/容量 | 固定44M A3 +.0086 S@s100；同配方扩容 s150 76→91.17% | 不混加不同实验收益 |
| S10B | 统一 P8 latent s100 +12.17pp；P2解码3.43×，参数−78.5% | 部分 C 下降；频谱机制为实证解释 |

P9e 等规模对照：Unified58M（58.04M 单 Flow）vs E+P（55.67M）；C=.50 S/Top-1 .5612/55.63% vs .5690/76.47%；s150 74.67% vs 91.17%；自草稿最高内容 C=.5419，速度≥75.09 vs69.82ms/img。支持当前任务专门化策略；因 E 的训练监督不同，不证明所有共享多任务网络都失败。

## 4. 协议、复现边界与当前任务

- test750 固定 Cross600+Same150，全部正式结果报告无 Same 旁路。冻结独立裁判；DINO-S Max 为目标画派库相似度，不是单参考忠实度。聚类配对 bootstrap 以源图为簇。
- s=0 精确 latent 恒等，P6 同重建参照 pixel max_err=0；tau0=0 A3 不变量。P9 E/P 交界 RMSE=.1935，大于内部 .033–.088，尚不能声称视觉平滑；另做固定输出过渡实验，不污染正式速度评测。
- Holdout-750 与训练/测试无作品重叠，但146/150源图、143/150参考图画家已见。s085 两训练种子 C=.6831/.6819、S=.5324/.5333、Top-1=58.50/58.17%。不是严格画家 OOD。P6 用 train_calib_300，A3/训练时长曾参考 test750。
- 当前锁定：C2 EMA step3500，ECore_NoDINO，S10B，A3=(1,1)，ws_mul=1,wc_mul=1.2,dtau0=-.05,6名义步。资产与 hash 见 [cleanup](../../cleanup.md)。训练/生成源码 [exp/v35_final](../../exp/v35_final/)，缺失的 V29/V32 执行模块必须补齐。
- **3060 纯 PyTorch 固定形状优化与全量门控已完成**：[RTX 3060 优化总报告](optimization/RESULTS.md)（[Profile](optimization/PROFILE.md) · [优化与消融](optimization/OPTIMIZATION.md) · [部署包](optimization/DEPLOYMENT.md) · 方案来源 [3060_PROFILE_OPTIMIZATION.md](3060_PROFILE_OPTIMIZATION.md)）。核心结论：`E-only` Host E2E 从 `6.92 -> 12.31 img/s`（`B=2`，`1.86x`），`E→P` 从 `3.10 -> 5.49 img/s`（`B=4`，`1.77x`，`Fast-5` `6.51 img/s`），全量 `test750` 配对 Bootstrap 非劣全部通过（`|ΔDINO-C| <= 0.0004`, `ΔTop-1 = -0.50pp ~ +0.83pp`）。One-shot 单独研发。
- **超参数与优化器专项**：[训练调参优先级与 Muon 评估](optimization/training/HYPERPARAMETERS_AND_MUON.md)：核对闭集/开放域配方、监督与梯度证据，给出待验证的局部候选及公平对照；当前保留 AdamW，权重继续冻结。
- **第一阶段执行准备**：[3060 · 16h训练定型计划](optimization/training/STAGE1_PLAN.md)：E代理Q32契约与B画法权重、One-shot P学习率/Q噪声、双EMA及可恢复串行代码已准备；尚未开始GPU执行，等待独立Qwen/F7任务结束后另行启动。S0复核将Fast-5 s100严格CI状态更正为不确定，原始记录保留。

原始完整报告：[V35 Final](../archive/historical/v35/V35_FINAL_REPORT.md) · [专家资产与调用链](../archive/historical/v35/EXPERT_OPTIMIZATION_DOSSIER.md) · [P8 JSON](../archive/historical/v35/data/p8_final_report.json) · [P9 JSON](../archive/historical/v35/data/p9_paper_audit_report.json)。历史文件保持原件，跨目录失效链接可从[归档前快照](https://github.com/KiritoFD/WEAVE/tree/0e314629099b18d7f4a91ec4ca69507549eb279f/docs/v35)查阅。
