# ICME Q32 单模型零训练裸测

使用 RTX 4090、`icme_q32_flagship.pt` 的 EMA、固定 test750 与 S10。汇报 Cross600 和 ER/Uki 各 120 张的 CSD 命中。Same150 经过模型。各臂共用源图条件、目标标签、噪声种子以及六步 midpoint 积分。运行链读取现成 Q32、亮度和少量训练参考统计，不调用 T4、B4、PRISM-1 或 Reflow 权重。

- M1 从纯高斯噪声出发，分别使用 cs_j 和 e2full（前两物理步四分支 CFG，后四步 cs_j）。
- M2 与 M1 e2full 同预算，只调整前两物理步全 CFG 中的内容权重，扫 1、2、3。
- M3 从 t=.20 开始，初始 latent 为 .80 高斯噪声加 .20 解析 latent。解析 latent 为源 latent 的亮度投影加目标类色度均值。
- M4 在 M1 e2full 上把 128 维 u_vivid 置零，模型风格输入仍有目标 one-hot。复用原内容路由和 CFG，以隔离去掉向量的影响。现有模型训练时用过 u_vivid，结果仅衡量冻结权重的移除干预。

旗舰权重来自前 3,000 步训练加同目标的后 3,000 步续训，续训日志为 15.5 分钟。两段都以纯高斯噪声到真实 latent 为训练桥，损失为 FM MSE 加 .10 亮度结构损失，并训练于缓存的 u_vivid 和五类均衡采样。15.5 分钟对应续训半程。新一轮单阶段候选需独立训练，旧权重的 M4 提供去掉向量的起点证据。

```bash
python -m exp.v27_single.bare_q32
```

运行入口见 [裸测脚本](../../../exp/v27_single/bare_q32.py)。训练配方见 [Q32 初训](../../../exp/v27/icme_step3_q32_vs_dino16.py) 和 [旗舰续训](../../../exp/v27/icme_step3b_q32_flagship.py)。

- [Flow-PnP 零训练推理](flow_pnp/README.md) 记录 12 档正交端点引导。
- [PRISM-1+ 单阶段训练](prism1plus/README.md) 记录随机初始化后的单步评测。
- [PRISM-1-Unified](unified/README.md) 记录 150 条教师验收及 4,000 步停止点的单步评测。

## Q32 裸测结果

| 配方 | DINO-S ↑ | DINO-C ↑ | CSD Top-1 ↑ | ER / 120 | Uki / 120 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 cs_j | .6347 | .3766 | 55.00% | 39 | 20 |
| M1 e2full | .6247 | .3839 | 52.33% | 21 | 22 |
| M2 wc=1 | .6350 | .3641 | 54.50% | 38 | 13 |
| M2 wc=2 | .6214 | .3930 | 48.67% | 14 | 20 |
| M2 wc=3 | .6157 | .4064 | 43.00% | 7 | 10 |
| M3 解析起点 | .5513 | .4112 | 39.67% | 28 | 0 |
| M4 零 u_vivid | .5458 | .4305 | 33.67% | 9 | 0 |

[原始记录](results.json)

公开文献前沿和内部历史点的口径见 [pareto.md](../../performance/pareto.md)。
