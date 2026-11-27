# CLEAVE 研究与部署文档

> 2026-11-27。当前闭集交付以 V35 P0–P9 为准；历史研究进入 archive，不再以 Vxx 目录作为工作入口。One-shot 是独立研究线，未达到闭集生产质量。

| 任务 | 唯一入口 |
|---|---|
| 当前模型、完整消融、质量与效率 | [V35_FINAL_REPORT](v35/V35_FINAL_REPORT.md) |
| 性能数据与最新 Pareto | [performance](performance/README.md) · [三联图](performance/figures/cleave_pareto_triptych.svg) · [交互图](performance/figures/cleave_pareto_latest.html) |
| 3060 部署与优化 | [执行方案](3060_PROFILE_OPTIMIZATION.md) · [现有资产/源码简报](v35/EXPERT_OPTIMIZATION_DOSSIER.md) |
| 模型资产保留与冷归档 | [cleanup.md](../cleanup.md) |
| 论文 | [icme2027](icme2027/) |
| S10B 与 Decoder 证据 | [vae](vae/README.md) |
| 外部基线 | [baseline](baseline/) |
| 早期协议复现 | [reproduction](reproduction/) |
| 开放域研究 | [CLEAVE-One 阶段报告](v35/CLEAVE_ONE_RESULTS.md) |
| 历史实验与失败记录 | [archive/historical](archive/historical/README.md) |

## 当前结论与边界

- E-Core 11.13M，P-Core 44.54M，S10B 10.66M；生成/解码 66.33M，完整系统 112.84M。不是 44M 单一网络。
- E 独立路径覆盖高内容，E→A3→P 覆盖中强风格；P9 实测两条路径重叠，但切换 RMSE=.1935，尚不能宣称无跳变。
- P8 s085 C/S/Top-1=.6514/.5261/57.83%，s150=.3435/.5919/91.17%；P9 matched C=.61 时 S=.5398（插值），未超过 StyleGallery 的 .5604。
- Stiefel 同构消融、E/P 正误条件、固定 latent Decoder 与等规模单 Flow 对照均已有数据；后者训练目标并不完全相同。
- 4090 B16 已加载模型 RGB→RGB 历史报告 69.82ms/img；当前实际 NFE 与异步计时待纠正，不将该数字外推为 3060 实测。Holdout 是作品隔离而非画家隔离。

## 目录维护

只在本层维护当前交付、论文、公共协议、性能索引与正在执行的优化方案。已结束阶段放入 archive/historical 或其后续按主题的归档目录；保留原始 JSON、配置与负结果。当前脚本使用归档后的证据路径；历史文档原字节保留，旧链接可通过归档前固定提交查阅。不要移动 exp、src、tools 的模块或训练资产来实现文档整理。

正式评测固定 test750=Cross600+Same150、S10 解码、独立裁判；test750 已用于开发。任何新性能结论须区分 E/P 路径、实际步数、条件等价 NFE、GPU、batch、计时范围及缓存状态。原始报告不得随叙事更新而改写。
