# One-shot｜开放域单参考迁移（独立研究线）

**2026-11-28 05:31 更新**：V37 已完成固定 12 小时实验。E0 原配方续训至 12,902 步，E1 反向真实配对同曝光至 12,902 步；完整 Core750、Artist-OOD 1000、Movement-OOD 500 和配对置信区间已归档于 [V37 结果](V37_RESULTS.md)。E1 在等内容比较中 DINO-S 约低 0.0054–0.0065，Top-1 低 7.7–13.0pp，P4 延长训练门槛未通过；正式 One-shot 仍保持 P+A3。与重型基线的全面量化差距拆解见 [One-shot 与 StyleGallery 差距专项总结](ONE_SHOT_VS_STYLEGALLERY_SUMMARY.md)，执行证据与训练审计见 [V37 计划与审计](V37_PLAN_AND_AUDIT.md)。

闭集 CLEAVE 已定型；任意参考不等于换一个风格 Encoder。当前 V35 CLEAVE-One 只证明参考响应，尚未解决高内容重绘。不要把它作为当前生产模型或将其 OOD 跟随率当作生成质量。

## 1. 已完成与真实瓶颈

V35 以22.3M闭集 P 主干为起点，训练19,997张/29画派；留出5派 Pointillism、Cubism、Pop Art、Art Nouveau、Fauvism（500张），另500 Artist-OOD。StyleEncoderC 2.38M，1561D统计+无序 Patch，经 AdaGN 注入。O3 Reference Swap 训练1400步，3060 峰值5.18GB，约2小时。

| 相同 tau=.30 | C | S | Top-1 | Movement follow | SG-25 C |
|---|---:|---:|---:|---:|---:|
| O2 | .4941 | .5491 | 34.50% | 84.17% | .2410 |
| O3 | .4177 | .5583 | 42.67% | 92.50% | .1647 |

Swap 提升参考跟随 +8.33pp，却损失 .0764 C。O3 tau=.35 SG-25 C=.1878；纯噪 O1 follow93.33% 但 SG-25 C=.0927。不能证明等内容前沿进步。未见画派 raw417 1-NN=.866，高于 EncC=.798，复杂编码器不是已证瓶颈。历史3060 4-step B1 237.82ms、B4/8约5.24–5.49 img/s 属旧 One-shot 权重，不代表当前闭集部署。

## 2. 下一阶段（计划，未见完整新结果）

利用全部 WikiArt 训练数据建立艺术生成先验；按此前拟定的核心五画派留出设定，原库103,250张、136非空画派，非核心131派原始85,651张（用户数据统计，实际清洗后需核验）。若训练端选择 V35 的另外五个 OOD 留出，则必须更新 manifest，不能混淆两种协议。

- 44M Q32 P-Core 从随机初始化；Style MLP 从随机初始化；仅训练画派拟合 PCA/先验。若 S10/公共编码器见过留出画派，仅称生成器层未见，不称整系统未见。
- 真实艺术 GT、同画家跨作品参考及去风格代理，Noise-to-Data FM；先全量长训，再增强跨画法代理。默认关闭昂贵逐步 Reference Swap。
- 3060 BF16、缓存 latent/Q32/风格统计，effective batch64，micro4/8/16 按 profiling；36h预算只是建议，实际按吞吐和数据曝光决定。不要为了画派均衡而丢掉大部分作品。
- Reference-A3 从单图估计低频亮度/色度统计，无画派硬编码，不复制参考布局。固定源/参考/噪声，以完整 test750、Artist-OOD、Movement-OOD、SG-25 的内容与风格联合指标验收。需要同44M/同曝光20K对照才能将收益归因全量数据。

当前仓库尚未提交 exp/v35_cleave_one 的核心脚本与原始 OOD JSON；全量预训练结果也未在本次可访问快照中。先补齐 manifest/权重与实际结果再更新本页，不根据旧报告推断新模型已完成。

- **2026-11-27 V36 CLEAVE-One 专项实测已归档**：全量 WikiArt（`76,319` Seen-Train）Zero-DINO 训练、P3 Seen-Dev 校准、`I0/I1/I2` 起点消融与 Core-5 完整 `test750` / 4090 冷启动吞吐评测见 [V36 专项技术报告](V36_CLEAVE_ONE_REPORT.md)（含 [PDF 版](V36_CLEAVE_ONE_REPORT.pdf)）。
- **21:47 Reference E-Core 更新**：4,500 步训练及 16 臂各 750 条评测完成；E 独立 Top-1 13.00%，E+P 最高已测 52.67%，尚未证明等内容前沿改善或训练充分。实际源图曝光仅训练集规模的 0.826 倍，后续长训是待执行计划；完整结果和训练诊断见[报告第 8 节](V36_CLEAVE_ONE_REPORT.md#8-reference-e-core-完整结果与训练充分性2147-实测)。

原始证据：[V35 CLEAVE-One](../archive/historical/v35/CLEAVE_ONE_RESULTS.md)；闭集条件的研究结论见 [Conditioning](../conditioning/README.md)。
