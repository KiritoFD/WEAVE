# Performance｜统一评测与前沿

当前 CLEAVE Final 的模型与因果结论见 [System](../system/README.md)。本目录维护统一性能证据和绘图，不再承担 E/P、Conditioning 或 Decoder 的原理说明。

## 当前正式结果

| 入口 | 内容 |
|---|---|
| [Pareto 三联图](figures/cleave_pareto_triptych.svg) | E-only 与 E→P 的完整前沿、目标分类率与速度 |
| [交互图](figures/cleave_pareto_latest.html) | 历史及定稿工作点 |
| [P8 原始 JSON](../archive/historical/v35/data/p8_final_report.json) | 最终44M系统的质量、效率与参数 |
| [P9 原始 JSON](../archive/historical/v35/data/p9_paper_audit_report.json) | 高内容补点、同 latent Decoder、单核/双核、Holdout |
| [P7 原始 JSON](../archive/historical/v35/data/p7_biglittle_report.json) | 原生 E 四档、P-only、E→P |
| [3060 执行计划](../system/3060_PROFILE_OPTIMIZATION.md) | 尚待最终双路径实测、实际 NFE 与质量门控 |

## 历史台账与绘图

- [MASTER_LEDGER](MASTER_LEDGER.md)、[metrics.csv](metrics.csv)、[metrics.json](metrics.json)：历史跨代台账，不保证涵盖所有 V35 点；最终结果以 P8/P9/P7 原始 JSON 为准。
- [build_metrics.py](build_metrics.py)：历史聚合入口；[plot_cleave_pareto.py](plot_cleave_pareto.py)：最新图表，读取归档的原始 JSON。
- [pareto.md](pareto.md)：非支配计算、外部基线与图例规范；历史协议不得冒充当前质量。
- 代表性输出图在执行端的 images / runs 等未跟踪目录；不能因 README 提及就假定 Git 内存在。

从仓库根运行：

~~~bash
python docs/performance/plot_cleave_pareto.py
~~~

历史台账重建（需相应历史资产）：

~~~bash
python docs/performance/build_metrics.py
~~~

## 统一比较约束

正式评测完整 test750（Cross600 主读数 + Same150 无旁路），S10 解码（显式 decoder-only 对照除外），冻结裁判。DINO-S Max 是目标画派库相似度，不能代替指定参考图忠实度。诊断臂与纯噪独立配置单列。等 C 比较只在真实工作点夹逼范围内插值。计时分清 E-only、E→P、实际模型调用、单条件等价 NFE、硬件、batch、后端及缓存范围。B16 的 ms/img 是吞吐折算，不是 B1 交互延迟。历史 V8 原版 decoder 口径不能直接混入 S10 对比。新的3060数据写入 performance/data，固定原始记录后再更新组件结论和图。

