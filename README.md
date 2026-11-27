# WEAVE → CLEAVE

**More capacity can preserve the wrong information.** 风格迁移不是单纯的重建或纹理匹配：生成器必须删除源画法，同时保留对象关系。CLEAVE 用选择性信息过滤、非对称 E/P 传输和轻量解码实现完整质量—效率谱。

[文档首页](docs/README.md) · [系统总览](docs/system/README.md) · [E-Core](docs/e-core/README.md) · [P-Core](docs/p-core/README.md) · [Conditioning](docs/conditioning/README.md) · [Decoder](docs/decoder/README.md) · [论文](docs/icme2027/) · [性能前沿](docs/performance/figures/cleave_pareto_triptych.svg) · [3060 优化](docs/system/3060_PROFILE_OPTIMIZATION.md) · [资产清理](cleanup.md)

## 当前系统（V35 P0–P9，2026-11-27）

| 模块 | 作用 | 参数 |
|---|---|---:|
| E-Core | 1-NFE 高内容编辑与目标风格草稿 | 11.13M |
| A3 | 低频亮度/色度解析起点 | 0 |
| P-Core | Q32 瓶颈条件 Flow，6 名义积分步 | 44.54M |
| S10B | 轻量解码 | 10.66M |
| 冻结 Encoder / ConvNeXt | 编码与布局 | 46.51M |
| 完整系统 | 无 DINO 推理依赖 | 112.84M |

E 独立路径不调用 P；E→A3→P 为高质量重绘路径。当前 P9 补测覆盖高内容区，但交界视觉连续性仍需验证。

| 实测点 | DINO-C | DINO-S | CSD Top-1 |
|---|---:|---:|---:|
| E-only l025 | .9166 | .3907 | 2.33% |
| E+A3 l050 | .8086 | .4274 | 31.50% |
| E→P s060 | .7813 | .4636 | 38.83% |
| E→P s085 | .6514 | .5261 | 57.83% |
| E→P s100 | .5502 | .5597 | 69.67% |
| E→P s150 | .3435 | .5919 | 91.17% |

来源：[V35 P7–P9](docs/archive/historical/v35/V35_FINAL_REPORT.md)，E-only 与 E+A3 配置不同。4090 B16 P 路径已加载 RGB→RGB 报告 14.3 img/s，3060 最终权重尚待统一实测。StyleGallery 在 C≈.61 处 S=.5604，CLEAVE 插值 S=.5398、Top-1=62.67%；质量取舍与计时协议见报告，闭集画派条件不是任意参考。

## 证据与复现

- Stiefel 消除二阶 Gram 泄漏（95.67%→20.67%），双种子同构消融支持中强档生成收益；空间风格信息仍可能存在。
- 固定 P-Core 交换 E 草稿，目标识别率变化最高 +65.33pp，支持目标流形路由；A3 在固定权重下额外增加约 .009 S。
- 固定最终 latent 更换 S10B，P9 的 s100 Top-1 +12.17pp；参数 −78.5%，历史 P2 解码提速 3.43×。
- 58M 单 Flow 对照不及双核当前配方，支持专门化系统价值，不证明任意共享网络不可能成功。

最终实验入口见 [exp/v35_final](exp/v35_final/)；旧模型、原始报告与负结果见 [历史档案](docs/archive/historical/README.md)。旧 WEAVE 复现指南 [REPRODUCE.md](REPRODUCE.md) 不代表当前 CLEAVE 推理接口。模型与缓存未随 Git 全部交付，按 [cleanup.md](cleanup.md) 的资产清单核验；远端缺失源码/权重时不得假装已能完整复现。

One-shot 独立开发：[One-shot 研究](docs/one-shot/README.md)。不把高参考跟随率当作高质量内容保持的证明。
