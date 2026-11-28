# CLEAVE｜组件文档

**当前基线：V35 P8/P9，2026-11-27。** 文档按最终系统的责任边界维护，不按实验版本组织。每个组件首页都是当前权威说明；原始报告和 JSON 保留在证据库，历史结果不冒充当前配置。

**2026-11-28 整体报告**：[CLEAVE / WEAVE 整体实验报告](system/OVERALL_REPORT.md)，汇总闭集19档、Cross600/All750、One-shot、外部基线、独立审计与3060部署，并标明未完成项和无效指标。

| 组件 | 解决的问题 | 文档 |
|---|---|---|
| E-Core | 1-NFE 高保真编辑；把 P 的起点推向目标画法 | [e-core](e-core/README.md) |
| P-Core | 粗尺度条件下的多步重绘；A3 起点与 Flow | [p-core](p-core/README.md) |
| Conditioning | 内容/风格信息的选择性保留；Q32 与 K8 | [conditioning](conditioning/README.md) |
| Decoder | 10.66M S10B；编辑 latent 的频谱响应 | [decoder](decoder/README.md) |
| System | 双核分工、完整前沿、评测、部署接口 | [system](system/README.md) · [3060 优化](system/3060_PROFILE_OPTIMIZATION.md) |
| One-shot | 未见画派与单参考图；独立研发 | [one-shot](one-shot/README.md) |

公共入口：[性能台账](performance/README.md) · [三联图](performance/figures/cleave_pareto_triptych.svg) · [论文](icme2027/) · [外部基线](baseline/) · [早期协议复现](reproduction/) · [资产清理](../cleanup.md) · [原始证据](archive/historical/README.md)。

## 文档维护原则

每个组件只维护一份简明、可复现的当前说明：职责 → 架构及接口 → 训练 → 独立贡献 → 失败路线 → 资产与来源。新实验先写原始报告和逐任务数据，再更新相关组件的结论；跨组件实验统一更新 System/Performance。版本号仅作为证据标识，不再创建 docs/v36、docs/v37 等顶层目录。未测内容明确标待验证，原始报告、配置、JSON 与历史负结果禁止覆盖。保留历史 Markdown 原件；若其跨目录链接已失效，可从[归档前固定快照](https://github.com/KiritoFD/WEAVE/tree/0e314629099b18d7f4a91ec4ca69507549eb279f/docs)查看。
