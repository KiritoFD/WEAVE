# 研究档案｜结论有首页，证据有出处

WEAVE → PRISM 的研究不是一份越积越长的实验流水账。我们把文档分成四层：**研究结论、可复现模型、原始证据、未完成假设**。原始实验报告不改写；后续实验推翻了旧结论时，在索引中标明。

> **快照：2026-11-21。** [根 README](../README.md) 负责回答“发现了什么”；本页负责回答“证据在哪儿”。V26-L 长队列仍在执行中，不能把排队、代码完成写成质量收益。

## 从这里进入

| 目标 | 权威入口 | 阅读建议 |
|---|---|---|
| **研究主张与定量跃升** | [根 README](../README.md) | 两条主线：潜空间传输与 S10 Decoder |
| **模型与复现** | [模型资产库](ours/README.md) · [实验源码](../exp/README.md) | 先确认 checkpoint、推理锚点、CFG、Decoder |
| **Decoder 为什么改变了全系统？** | [VAE 专题](vae/README.md) | Taylor 剪枝 → LPIPS 恢复 → C0/C1 蒸馏 → S10B |
| **V26 的阶段裁决与长队列** | [V26 导航](1120/README.md) · [执行日志](1120/PROGRESS.md) | 结果 / 作废 / 待测分别看 |
| **局部性真的有用吗？** | [K1](v26/4090/K1_REFATTN_CAUSAL_AUDIT.md) · [Exp 1/2 专题](v26/LOCALITY_UOT_FIBER_BUNDLE.md) | 现有局部 Value 污染内容；Bary-SWD 尚待实训 |
| **模型是否真的推前沿？** | [V25 消融台账](v25/ABLATION_LEDGER.md) · [模型血缘](v25/MODEL_LINEAGE.md) | 区分新权重、锚点、解码器与 Bug 修复 |
| **历史数据与负结果** | [历史 Master Ledger](performance/MASTER_LEDGER.md) · [metrics.csv](performance/metrics.csv) · [负结果](negative_results/README.md) | Ledger 正文是 V6–V21 快照；CSV 后续有追加 |
| **论文与早期 WEAVE** | [ICME 论文工程](icme2027/README.md) · [早期复现](reproduction/baseline_reproduction.md) | 正式发表前需复核最新证据 |

## 历史拐点：找到最关键的原始实验

| 版本 | 真正改变认识的结果 | 原文 |
|---|---|---|
| **V6–V14** | 去源风格≠到目标风格；通用艺术恢复梯度≠目标画派方向 | [V6](v6/V6_CONCLUSIONS.md) · [V10.2](v10_2/V10_2_RESULTS.md) · [V14](v14/V14_RESULTS.md) |
| **V15–V18** | 扩大自配对 Flow 的生成能力，可能反而损害编辑能力 | [V15](v15/WEAVE_V15_SCALING_RESULTS.md) · [V18](v18/V18_RESULTS.md) |
| **V19–V20** | **各向异性内容预算＋16×16 粗尺度结构条件**，带来真正的跨风格质量跃升 | [V19](next1118/V19_RESULTS.md) · [V20](next1118/V20_RESULTS.md) |
| **V21–V23** | 去泄漏不等于生成；早停、锚点、跨风格监督和 CFG 实现至关重要 | [V21](next1118/V21_RESULTS.md) · [V22](v22/V22_P1_RESULTS.md) · [V23](v23/V23_RESULTS.md) |
| **V24–V25** | 不重训，仅靠锚点与潜变量组合仍能推动高内容 Pareto | [V24](v24/PROGRESS.md) · [R12](ours/02_PRISM_HighContent_R12/RECIPE.md) |
| **V26** | C2nt 局部 Value 判负；SR-CFG、跨画派采样单独使用都没有解决根因 | [六问总结](1120/V26_RESULTS_SUMMARY.md) · [L0](v26/4090/L0_CROSSSTYLE_CAUSAL.md) |
| **V32–V33** | V32 全链路去 DINO（ConvNeXt-Stage3）；V33 验证路由/Y16/Q64+/32×32控制均未抬升冻结 22M 主干 Pareto 上限，锁定真正瓶颈 | [V32 Zero-DINO](v32/NODINO_REPORT.md) · [V33 实验报告](v33/V33_SEMANTIC_CONTROL_REPORT.md) |
| **V34** | **解析起点 `A3` ($L_{\rm low}+P_C$) + `44.54M Q32` 扩容 (`C2`) 打破容量瓶颈**，在 `test750`（`+0.011–0.020 S`）与 `holdout-300` 盲测集显著推升 Zero-DINO Pareto 前沿 | [V34 实验报告](v34/V34_FRONTIER_REPORT.md) |
| **V35** | **CLEAVE 终稿定型（Zero-DINO / P0–P9 全因果归因 + 等参数共享主干对照）**：`44.54M P-Core + 11.13M E-Core + 10.66M S10B`（Zero-DINO），`P9a` 补齐 `s=0.60–0.72` 后严格区间内插较 V32 在 `C=[0.50,0.61,0.70,0.75]` 均增 **`+0.0157 S`**（`C=0.70` **`+0.0245`**，`s100 69.67% / s150 91.17%`），`4090` 冷启 **`14.3 img/s` (`69.82 ms`)** 全面优于 `58.04M` 共享单主干；Holdout 修正为严格作品级留出（146/150 画师已见，0 画作重叠） | [V35 终稿报告](v35/V35_FINAL_REPORT.md) · [CLEAVE-One](v35/CLEAVE_ONE_RESULTS.md) |

## Decoder 独立研究线

从 [VAE 总览](vae/00_OVERVIEW.md) 阅读完整决策；再看 [原始 S10 训练](vae/01_S10_TRAINING_FULL.md)、[3060 端到端实测](vae/08_STAGED_PROFILE_AND_TRT.md)、[固定 latent 的 Decoder 对照](v6/Q2A_ORACLE_DECODER_RANKING.md)、[蒸馏与 C0/C1](vae/05_SD0_S10_V2_FINAL.md)、[S10B 最终交付](vae/10_S10B_FINAL.md)。

**务必分清：旧 S10 的成功主要来自剪枝＋真实图感知恢复；C1 才是响应蒸馏，且未取得更高的下游风格 Top-1。**

## 证据层级

**已测数据**以原始输出、对应 RESULTS/审计报告为准；**计划**以 [V26-L 长队列](1120/08_v26_long_plan.md) 为准，不能当成实测；**历史观点**可在 [旧交付总结](DELIVERY_20261118.md) 和 [legacy_root](legacy_root/EXPERIMENTS.md) 中追溯，但“最新”“全面超越”须重新检查任务、硬件与参考条件。

评测基准：**test750 = Cross600＋Same150**、S10 解码和冻结裁判。test750 已用于反复模型选择，**不是全新盲测集**。计时必须明确 GPU、batch 和 Decoder 分段还是完整 RGB→RGB。[工程约束](../AGENTS.md)
