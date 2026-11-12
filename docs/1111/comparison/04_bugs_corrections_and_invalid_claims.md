# 纠错台账与结论失效范围

旧报告既保存历史，也可能包含当时未识别的bug、口径标签错误或过强解释。本章不删除历史，而是说明哪些数值还可用、哪些结论不能继承。

| 问题 | 旧状态/说法 | 本复盘采用 | 影响边界 |
|---|---|---|---|
| 原始SD-VAE与艺术A0混称 | LEADERBOARD把sdvae写A0 SD-VAE | 两者权重/Encoder不同；统一SD0与Art-A0 | 整套替换与纯Decoder差分不能混淆 |
| VAE encode输入范围 | 旧eval_tiers给[0,1] | 正确链路给[-1,1] | 旧表保留历史值；新链路基准重测 |
| 源图参照 | 部分paper_rgb_v1标签实际用VAE往返 | 主表相对原始RGB；往返作诊断 | 零点、DINO-C与LPIPS不再同义 |
| Identity命名 | R2 identity返回latent后decode | R2标latent no-edit/roundtrip | 不能与RGB恒等1/0混一行 |
| EvaluatorV2占位/不一致字段 | DINO-SSM整列0、rawVAE No-op LPIPS0 | 保留历史异常，采用已核验链路的真实往返误差 | 不能解释成完美结构保持或无损编解码 |
| cross筛选 | 缺is_cross时默认True | 比较tgt_style与src_style | 否则750混成600，成功率会虚高 |
| Minimalism误判 | pred_class整数与字符串比较 | 按整数类别索引 | 旧0.000不是没有偏置 |
| 聚类bootstrap | np.isin选簇抹掉重复抽样；本地/3060的R2导入模块仍如此 | 4090attribution保留簇重复；R2的CI仍需修复重算 | R2均值可用，细小差异不能沿用其CI宣称显著 |
| V10 SWD梯度 | 在no_grad中计算 | V11移出no_grad并验证有效更新 | 原exp24不能当分布损失机制阴性 |
| 条件分布统计 | 五类不分target混合距离 | 按target分组再聚合 | 整体近不代表每个条件分布正确 |
| V10机制编号漂移 | 早期IMPLEMENTATION分组与正式suite不同 | 按具体报告配置读exp24 SWD/exp27在线SDE | 不能跨版本用组号代替机制说明 |
| 在线桥与参数化 | 在线loss混方向、score_current反向符号、Monge独立端点 | 方向专属漂移损失、速度反号、Monge要求有效配对 | 修正前阴性不能否定原理论 |
| V11监督强度 | 将d2的ρ=0.802套到d1 | d1仅0.0249，d2为0.8020 | d1不足以评判分布监督；d3/d4未见正式实测 |
| V8 Sinkhorn | 历史实现缺陷 | 修复后再训练/审计 | 修复前阴性不作为最终OT结论 |
| V6两因素交互 | 旧2×2计算得+0.0105 | 正确交互−0.00288 | “正协同”撤回 |
| V6方向纯度 | 把ΔM=ΔT−ΔS称几何方向 | 真正latent cosine另测 | 分类margin变化不等于方向纯度 |
| ScoreDiff噪声门槛 | 0.0113经验线被扩大解释 | 按后续复核撤回强阈值 | 不用于跨方案准入 |
| post_quant_conv | 漏掉或双重调用 | wrapper内外恰好一次 | 错误路径重建值不代表模型质量 |
| 训练数据泄漏 | 只排val未查in_train | clean78661显式过滤 | 旧独立holdout与外部训练域污染分开说 |
| LPIPS分辨率 | 训练256、验收512 | 恢复训练改512 | 先前纹理失败不是纯容量证明 |
| S10最终checkpoint | 页首写p4 | 正文/最终JSON为p5_lowlr | 加载时核对元数据与manifest |
| “Decoder KD” | 把恢复训练统称蒸馏 | 最终实际真实RGB L1+LPIPS | 训练对象/损失必须说清 |
| 梯度冲突代表性 | quant_conv单层−0.61 | 全Encoder−0.074 | 不能单层概括全网 |
| 排序有限差分 | 将67.4%误差归咎不可微 | sort多数位置分段线性；问题未完全解开 | 保留数值审计疑点 |
| 随机子空间结论 | 宣称完全各向同性 | 仅一次随机768D能量≈维数比 | 不排除学习子空间 |
| 联合Encoder checkpoint | quant_conv requires_grad但不在optimizer | 未更新参数未保存不等于丢训练成果 | 不可杜撰失败根因 |
| 梯度预算 | ArtEncoder标定成1.5% | 改15%并核对GradientCache | 旧配置不能当正式有效预算 |
| 重建门控参照 | StageA与A0比 | 固定S10背景下对照S10 | 避免归咎Encoder已存在Decoder损失 |
| 读取器版本覆盖 | 旧diagnosis与新gate混用 | 09:58远端版与初始化修复后回传1500版分开 | 报告时间/steps/seed/hash都要留 |
| 读取器初始化种子 | 先build模型再manual_seed | 修复顺序；最终报告记两次300步一致 | 前版seed字段相同不保证初始化相同 |
| 检索flat基线 | 固定计划0.0708当本轮值 | 正式1500本轮0.0912 | 用matched基线做科学比较 |
| 风格判官解耦 | 真实图分类高便说无内容混淆 | 保留content-matched与margin | 校准通过不等于生成图上无偏 |
| 容量上限 | 整latent插值当stylecode/FiLM证明 | 它会携带目标内容 | 未做过的受限编辑测试仍未验证 |
| 容量永久排除 | 218K/845K近便说与容量无关 | 仅当前模型/目标/预算未见明显宽度收益 | 不构成架构普遍结论 |
| 速度口径 | Decoder2.86×套整个WEAVE | 仅4090bs1/512Decoder段 | 须另计VAE编码/生成器/梯度与IO |
| StyleAffinity权威性 | 人工平均当论文验证主指标 | 各原指标分列、组合仅辅助 | 不支持“权威全面胜出” |

## 判定原则

真正实现修正与仅提出怀疑分开；有错误不等于整份报告全部无效。若错误影响了评分链路，绝对值只留历史表；若只影响解释，保留测量但收窄推论。
bug修复后的新阴性必须单列，不能用修复前错误结果反复累计“失败次数”。

本次确认仍存在于当前源码的R2 cluster bootstrap问题仅写入文档，未擅自改动另一实验线或覆盖结果；门槛未过的均值结论仍成立。

## 主要证据入口


**证据来源：** [tools/v6_24h/eval_tiers.py](../../../tools/v6_24h/eval_tiers.py)；[tools/v6_24h/stats.py](../../../tools/v6_24h/stats.py)；[tools/v6_next/r2_eval.py](../../../tools/v6_next/r2_eval.py)；[tools/vae_swap_eval/run.py](../../../tools/vae_swap_eval/run.py)；[tools/vae_swap_eval/attribute.py](../../../tools/vae_swap_eval/attribute.py)；[docs/v6/V6_NEXT_P0_CORRECTIONS.md](../../v6/V6_NEXT_P0_CORRECTIONS.md)；[docs/v6/V6_24H_P0_AUDIT.md](../../v6/V6_24H_P0_AUDIT.md)；[docs/v11/V11_P1_RESULTS.md](../../v11/V11_P1_RESULTS.md)；[docs/v6/V2_FAILURE_DIAGNOSIS.md](../../v6/V2_FAILURE_DIAGNOSIS.md)；[docs/v6/S10_FINAL_REPORT.md](../../v6/S10_FINAL_REPORT.md)；[docs/v6/ART_VAE_ENCODER_REPORT.md](../../v6/ART_VAE_ENCODER_REPORT.md)；[docs/v6/LOCAL_STYLE_READER_REPORT.md](../../v6/LOCAL_STYLE_READER_REPORT.md)；[LEADERBOARD.md](../../../LEADERBOARD.md)。
