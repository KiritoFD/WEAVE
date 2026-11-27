# V26：研究诊断已收官，任意参考迁移仍在攻关

**C2nt 不是最终方案，StyleGallery 式局部注意力也不是加上就能奏效。** V26 把几条高成本直觉逐一检验：Q32 能去 Gram 泄漏，却未证明生成收益；SR-CFG 无法凭空造出没学到的风格方向；C2nt 现有局部 Value 在受控干预下拖累内容。V26-L 因此转向 **PRISM-Flow＋正确的跨风格教师＋最小参考条件**。

> 本目录 `01–07` 是 2026-11-20 前后方案快照；[08 长队列](08_v26_long_plan.md) 是执行计划；[PROGRESS](PROGRESS.md) 才是实测和作废记录。**代码写好 ≠ 实验跑好。**

## 已完成的科学裁决

| 证据 | 主要结果 | 原始出处 |
|---|---|---|
| **K0 Q32** | 二阶 Gram 风格探针 **92.3%→20.0%**；生成必要性仍 **INCONCLUSIVE** | [K0](../v26/4090/K0_Q32_FEATURE_AUDIT.md) |
| **SR-CFG** | C2nt 基线 Top-1 **42.17%**；排斥项没在相近内容区实现净提升 | [六问总结](V26_RESULTS_SUMMARY.md) |
| **K1 与 Exp 1 局部性** | 在 C2nt 上，局部 Value 置零 DINO-C **0.5392→0.5892**；去空间化、调温、UOT 均无救 | [K1](../v26/4090/K1_REFATTN_CAUSAL_AUDIT.md) · [Exp 1](../v26/LOCALITY_UOT_FIBER_BUNDLE.md) |
| **K2 风格损失** | 更正后的判别 AUC：VGG 均值 **0.673**、SWD **0.661**；互补梯度余弦 **0.179**，生成收益待验证 | [K2](../v26/4090/K2_STYLE_LOSS_AND_SUBJECT_AUDIT.md) · [勘误](PROGRESS.md) |
| **L0 跨画派采样** | 1200 步 Top-1：原近邻 **16.17%**，强制跨画派 **11.33% / 12.00%**；仅修采样无效 | [L0](../v26/4090/L0_CROSSSTYLE_CAUSAL.md) |
| **4-step & SG-25** | 4 步 Euler Top-1 **43.17%**；OOD 响应 **+0.2216**，但内容 DINO-C **0.3566** | [PROGRESS](PROGRESS.md) |

**C2nt 原训练配方已冻结**，但其参考响应与诊断实验保留。[降级决定](07_c2nt_demotion_decision.md)

**L1 的教师条件反置已修复；修复前 F2=11.00% 彻底作废。** 教师应按 `(source, target_style)` 生成，之后必须重新评测。[修复日志](PROGRESS.md)

## 文档与版本

| 文档 | 身份 |
|---|---|
| [V26_RESULTS_SUMMARY](V26_RESULTS_SUMMARY.md) | K0/K1/K2 的阶段结果快照；**K2 的 13× 说法已被后续更正** |
| [PROGRESS](PROGRESS.md) | 最新执行、修复、已测/作废状态；当前判断优先看这里 |
| [08_v26_long_plan](08_v26_long_plan.md) | L0–L6 实验设计与停机准则，不是结果报告 |
| [局部性 Exp 1/2](../v26/LOCALITY_UOT_FIBER_BUNDLE.md) | Value 根因复核；Bary-SWD 仍未有可归档的训练对照 |
| [07_c2nt_demotion_decision](07_c2nt_demotion_decision.md) | C2nt 从主线转为诊断探针的依据 |
| [01 诊断](01_diagnostics_and_srcfg.md) · [02 风格损失](02_swd_and_mechanisms.md) | 初版研究问题与设计 |
| [03 双卡排程](03_schedule_and_dual_gpu.md) · [06 K 队列](06_4090_diagnostic_queue.md) | 历史工程排程 |
| [04 工程契约](04_engineering_and_tests.md) · [05 验收](05_validation_and_deliverables.md) | 实验协议、验收与交付物 |

## 下一阶段必须解决的不是“再加 Attention”

**先证明跨风格目标监督真的正确，再判断局部纹理损失是否有效。** 现有 C2nt 局部 Value 的负收益不能外推为“局部性永远无用”；但足以支持 **Global Fiber / 连续风格弦先行**。L2 Full Decoder matched fine-tune 与 Exp 2 Bary-SWD 仍是待验证实验，不得提前写进性能排行榜。

返回 [研究档案导航](../README.md) · [根 README](../../README.md)。
