# WEAVE / Art-VAE 全路线复盘 · 2026-11-11

这是一份面向继续研究、组内交接与结果复核的总档案：**3060的WEAVE主线、4090的VAE/Decoder主线，以及两条线交叉后的评测、归因与纠错**。
不是将旧报告串起来的结论集合；原始测量、当前有效口径、未完成事项和被修正的解释分别记录。

## 先读这几条

1. **WEAVE有过可保留的进展。** 从较重的ODE/Direct缩到218K单步模型；R1逐图方向能提高风格判别；B3/B4把部分教师收益摊销进单步模型。但风格与内容仍有明显交换，R2新教师/投影/再蒸馏没有通过相对B3的增益门槛。
2. **VAE训练并非没有好结果。** Art-A0改善艺术域重建；S10把Decoder参数减少78.46%、4090的Decoder段forward加速2.86×；联合训练恢复了一部分压缩损失。这些不等于风格几何或所有下游迁移指标都改善。
3. **S10的下游收益是生成器与指标相关的。** 真正固定Encoder的A0→S10，Oracle/B3/B4的CSD Top1分别+3.67/+4.00/+5.00pp；五个候选DINO-S都提高。ScoreDiff的CSD却下降，DINO/CLIP及margin提高。不能统称“Decoder训练使风格都变差”。
4. **全套换VAE与只换Decoder是两种实验。** 相对SD0，A0/C_edit在现有冻结WEAVE上CSD整体变差；B4的2×2按CSD显示Decoder负效应更明确，按DINO却显示Encoder负、Decoder正。该归因仅适用于测过的B4链路。
5. **最新4090两项门控未过。** ArtEncoder Stage A完成500步，AUROC仅+0.0031，B/C依门控跳过；正式1500步DeepSets/CNN读取器均未突破困难AUROC/检索门槛。局部读取器训练不是又训了一套VAE。
6. **旧高分与旧失败都要检查口径。** 输入[0,1]/[-1,1]、raw RGB/往返参照、cross600/same150、TRT/BF16、统计bug、零梯度损失、checkpoint版本均发生过变化。最新核对发现R2的cluster bootstrap仍有重复簇被消掉的问题；均值可用，小差异显著性暂不能继承。这里不做跨口径“冠军榜”。

## 阅读导航

| 要解决的问题 | 阅读位置 |
|---|---|
| 从起点到现在发生了什么 | [研究时间线](01_TIMELINE.md) |
| 为什么同名模型分数不同、指标怎么定义 | [评测协议与命名](02_EVALUATION_PROTOCOLS.md) |
| 哪些路线有效、为什么没过、下一步依据 | [综合判断](03_CONCLUSIONS.md) |
| 全部主线/分支执行状态快速查找 | [54条路线索引](04_ROUTE_INDEX.md) |
| 早期主干、V4伪GT、V5监督端点 | [WEAVE 1](weave/01_early_model_and_v4_v5.md) |
| V6 Direct/ScoreDiff与结构消融 | [WEAVE 2](weave/02_v6_direct_scorediff_ablation.md) |
| V7–V11去噪、分布匹配、CFM/OT、DCAE、EDM等 | [WEAVE 3](weave/03_v7_to_v11_distribution_routes.md) |
| R1、Q2、Oracle、算子蒸馏与最新R2 | [WEAVE 4](weave/04_r1_oracle_distill_r2.md) |
| 更早考古、未执行路线、范围边界 | [WEAVE 5](weave/05_archaeology_and_unexecuted_routes.md) |
| Art-A0/A1/A2艺术后训练 | [VAE 1](vae/01_art_vae_a0_a1_a2.md) |
| SWD、CSD/ALADIN、REPA、坍塌、子空间/Flow | [VAE 2](vae/02_geometry_teachers_and_failures.md) |
| S10压缩恢复、训练目标、速度与梯度 | [VAE 3](vae/03_s10_decoder_pruning_and_recovery.md) |
| 联合StyleVAE A/B/C与可编辑性 | [VAE 4](vae/04_joint_style_vae_abc.md) |
| 最新ArtEncoder和局部/CNN读取器 | [VAE 5](vae/05_art_encoder_and_local_readers.md) |
| 外部方法与现有WEAVE工作点 | [分协议横向结果](comparison/01_rankings_and_protocol_families.md) |
| 20格均值、CI、same、逐风格、重建 | [完整换VAE矩阵](comparison/02_full_vae_swap_matrix.md) |
| DINO/CLIP/CSD分歧、纯Decoder、2×2归因 | [多指标与归因](comparison/03_style_metrics_and_attribution.md) |
| 哪些旧结论撤回或收窄 | [纠错台账](comparison/04_bugs_corrections_and_invalid_claims.md) |
| 权重、代码、机器、依赖与复现 | [产物与复现](comparison/05_artifacts_and_reproduction.md) |
| 机器可读路线、原表与来源指纹 | [路线CSV](appendix/route_ledger.csv)、[原始表格CSV](appendix/arm_ledger.csv)、[来源清单](appendix/source_inventory.csv) |
| 远端结果冻结与最后状态 | [采集快照](appendix/live_snapshot.json) |
| 未决问题与下一步最小验证 | [未决项](appendix/open_questions.md) |

## 阅读规则与证据等级

- **配对原始JSON**：本次20格、多指标与2×2均值/CI从本地逐任务JSON直接提取；最新R2/ArtEncoder/Reader从远端聚合JSON冻结。
- **历史正式报告**：保留原协议原数值，不等于本次重跑；有后续修正时在本复盘收窄判断。
- **日志/部分checkpoint**：只证明执行到可见位置；例如REPA日志到220/500，不能标为500步完整失败。
- **计划/建议**：没有正式结果就写未执行/未验证，不将方案文本计为实验。
- **更早archive**：有来源索引与既有考古报告；不声称本次逐一重验已清理的早期模型。

## 本次采集边界

本地仓库与两端文件均实际读取。3060最后状态刷新在11月11日10:16左右（CST），未见WEAVE训练/评测任务。
4090远端Reader1500步结果先冻结于09:58附近；随后本地回传了修复初始化顺序后的最终1500步结果，local/CNN AUROC为0.4645/0.4770，本复盘主结论采用后者，并保留前版0.4654/0.4726。
最终报告另记两次300步复现验证通过。10:16远端进程刷新时SSH连接超时，故不推断其后任务状态；已有回传结果仍可确认。
各版本原文件hash、大小、mtime与保留字段存于快照，后续同名目录覆盖不会改变本次档案。

文件系统时间只是采集/产物时间；不以Git提交时间或文件mtime推导训练耗时。部分历史实际日期不可复原时按研究先后叙述。
本次只生成文档，没有新训练/推理评测，也没有停止、修改其他会话的工作。

## 附录格式

route_ledger每行是一条研究路线；arm_ledger每行是一张来源原表，`columns_json`保留表头、`rows_json`保留全部行，避免将不同协议同名臂强行合并。
原表包含的“PASS/FAIL/建议”等是当时记录；本次判读以正文和纠错台账为准。source_inventory区分`used`与`catalog_only`；后者仅编目，不冒充逐项事实核验。
