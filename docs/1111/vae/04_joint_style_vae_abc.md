# 联合 StyleVAE A/B/C：重建、风格组织与可编辑性

这里的 A/B/C 是 **A_joint / B_style / C_edit**，与前面的 Art-VAE A0/A1/A2，以及后面的通用 ArtEncoder Stage A 完全不同。
起点为 A0 Encoder + S10 Decoder，试图用联合训练恢复重建，并让低维风格统计可替换。

## V14 · A_joint：联合重建恢复

| 项 | 记录 |
|---|---|
| 执行侧 / 状态 | 4090 / **完成400步** |
| 要回答的问题 | 联合更新 Encoder 与 S10 Decoder 能否弥补压缩损失？ |
| 实际改动 | partial Encoder+Decoder 重建训练，基准比较 frozen/partial/full 与手工 gradient checkpointing。 |
| 任务与评测 | 同系列128图评测；不能直接减去S10另一个256图表。 |
| 主要结果 | 128图 A0 LPIPS0.06789，A_joint0.07805；B/C约0.0745/0.0742。后来同150图链路 C_edit0.07605优于S10 0.08419。 |
| 本次复盘判断 | 联合训练有恢复压缩后重建的证据；这是E+D联合效应，不能称为单独Decoder收益。 |


**证据来源：** [docs/v6/STYLE_VAE_ABC_REPORT.md](../../v6/STYLE_VAE_ABC_REPORT.md)。

## V15 · B_style：风格统计组织

| 项 | 记录 |
|---|---|
| 执行侧 / 状态 | 4090 / **完成1500步 / 编辑门槛失败** |
| 要回答的问题 | 类别可读性提升能否产生可用编辑坐标？ |
| 实际改动 | full Encoder+Decoder，低维统计风格读出、SupCon/CE和多裁剪；一次梯度标定到15%。 |
| 任务与评测 | 20类探针；另5类跨内容style-swap，与WEAVE D5不是同一集合。 |
| 主要结果 | 完整latent探针13→15%；8维统计11.5→14.12%；跨内容替换Top1 A0 19.67%、A19.67%、B18.8%、C19.0%；判官自身61.5%。 |
| 本次复盘判断 | 有微弱判别收益，未转成编辑；不能把18.8%直接与WEAVE600的B4 21.33%比较。 |


**证据来源：** [docs/v6/STYLE_VAE_ABC_REPORT.md](../../v6/STYLE_VAE_ABC_REPORT.md)。

## V16 · C_edit：显式编辑目标

| 项 | 记录 |
|---|---|
| 执行侧 / 状态 | 4090 / **记录到1120步 / best元数据upd1000 / 评测完成** |
| 要回答的问题 | 加入swap/edit损失能否强迫风格坐标可编辑？ |
| 实际改动 | 在B后继续编辑与重建训练，训练侧 edit_style 与 SupCon下降。 |
| 任务与评测 | 原定1500；可见日志1120，best checkpoint upd1000；不可写成1500完整训练。 |
| 主要结果 | 训练edit_style约0.386→0.10，SupCon2.76→0.90；独立跨内容替换Top1仍19.0%；后来5个WEAVE生成器整套替换均降低CSD Top1。 |
| 本次复盘判断 | 训练损失收敛没有形成泛化编辑；整套latent分布显著漂移，旧WEAVE兼容性不足，但不能把全部责任归给Encoder。 |


**证据来源：** [docs/v6/STYLE_VAE_ABC_REPORT.md](../../v6/STYLE_VAE_ABC_REPORT.md)；[deliver/vae_ranked_eval/STYLE_ATTRIBUTION.md](../../../deliver/vae_ranked_eval/STYLE_ATTRIBUTION.md)。

## 吞吐与数值结果原表


**证据来源：** [docs/v6/STYLE_VAE_ABC_REPORT.md](../../v6/STYLE_VAE_ABC_REPORT.md)。

### 原报告表 1：1. 基础设施实测（方案要求的第一步）

| 模式 | 最优 bs | 峰值 alloc | img/s | Encoder 可训 |
|---|---:|---:|---:|---:|
| frozen（参考） | 8 | 19.28 G | 14.43 | 0 |
| **partial**（后两级 Down+Mid+quant_conv） | **6** | 17.92 G | **12.22** | 30.7M |
| **full** | **4** | 19.89 G | **9.05** | 34.2M |
| full + 梯度检查点 | 8 | 20.52 G | 7.69（更慢） | 34.2M |

### 原报告表 2：2. 训练结果

| 阶段 | 步数 | L1 | LPIPS | CSD Keep | `s_ratio` | `z_mean` |
|---|---:|---:|---:|---:|---:|---:|
| S10（起点） | — | 0.0673 | 0.0926 | 0.8535 | — | — |
| **A** 联合重建（partial） | 400 | 0.0650 | 0.0781 | 0.8789 | 2.72 | 0.13 |
| **B** 风格组织（full） | 1500 | 0.0644 | 0.0745 | 0.8720 | 3.15 | 2.66 |
| **C** 可编辑（full） | 1120 | 0.0648 | 0.0742 | 0.8816 | 3.66 | 2.05 |

### 原报告表 3：3.1 重建

| 模型 | L1 | LPIPS@512 | CSD Keep |
|---|---:|---:|---:|
| **A0（原 VAE，参考）** | **0.06235** | **0.06789** | **0.94972** |
| A | 0.06496 | 0.07805 | 0.87886 |
| B | 0.06435 | 0.07447 | 0.87196 |
| C | 0.06480 | 0.07423 | 0.88159 |

### 原报告表 4：3.2 风格线性探针（top-20，随机 5%）

| 模型 | 全 latent | μ-only | logσ-only | s_full(8维) |
|---|---:|---:|---:|---:|
| A0 | **13.00%** | 10.00% | 9.25% | 11.50% |
| A | 13.12% | 10.25% | 8.87% | 11.63% |
| B | 13.75% | 11.25% | 7.87% | 13.25% |
| C | **15.00%** | 12.00% | 9.62% | 14.12% |

### 原报告表 5：3.3 跨内容风格替换（最关键）

| 模型 | CROSS Top-1 | dprob | DINO-C | LPIPS | SAME Top-1 |
|---|---:|---:|---:|---:|---:|
| A0 | 19.7% | +0.0297 | 0.8271 | 0.6462 | 58.0% |
| A | 19.7% | +0.0582 | 0.8611 | 0.5215 | 54.7% |
| B | 18.8% | +0.0575 | 0.8616 | 0.5554 | 55.3% |
| C | 19.0% | +0.0539 | 0.8600 | 0.5966 | 55.3% |

### 原报告表 6：5. 与之前工作的关系

| 尝试 | 风格方向可复现 | 跨内容可迁移 |
|---|---|---|
| 固定 768 维随机投影 + Ranking | 弱 | ✗ |
| 风格质心差 | 部分（cos 0.04–0.62） | ✗（L0，分类检验 P(style)=0.000） |
| **本次：A0+S10 联合重训 + 8 维统计坐标** | 探针 13.00→15.00% | ✗（19.0% vs 随机 20.0%） |

### 原报告表 7：7. 产物

| 文件 | 内容 |
|---|---|
| `tools/vae_style_vae/model.py` | StyleVAE、`style_content`/`recombine`/`style_swap`、手动 encoder 检查点、StyleHead |
| `tools/vae_style_vae/data.py` | 实时 RGB 数据集（不用 latent 缓存）、`BalancedStyleBatchSampler` |
| `tools/vae_style_vae/losses.py` | 重建+KL、SupCon、裁剪一致性、edit 循环一致性 |
| `tools/vae_style_vae/train.py` | A/B/C 三阶段、lam_s 梯度标定与 ramp、latent 监控 |
| `tools/vae_style_vae/eval.py` | 四组评测（top-20 探针、重建、750 任务替换、μ/σ 分解） |
| `runs/vae_style_vae/*_best.pt` | A/B/C 最佳权重 |
| `runs/vae_style_vae/eval_*.json` | 四组评测原始数据 |
| `runs/vae_style_vae/joint_infra_bench.json` | 联合训练吞吐/显存实测 |

## V17 · 容量上限探针：完整latent插值

| 项 | 记录 |
|---|---|
| 执行侧 / 状态 | 4090 / **完成 / 解释范围受限** |
| 要回答的问题 | Decoder是否能在latent输入改变时输出另一目标风格？ |
| 实际改动 | E(source)向某一真实target图的整个latent插值t=0/.25/.5/.75/1。 |
| 任务与评测 | 独立5类judge准确率54.5%；完整latent，不是纯风格code。 |
| 主要结果 | crossTop1 11.25%→12.25%→20.0%→39.5%→60.0%。 |
| 本次复盘判断 | 证明输入接近目标图的完整latent时可恢复目标类；t=1同时复制目标内容，不是保内容迁移，也不是FiLM/低维stylecode容量的正式验收。 |


**证据来源：** [docs/v6/STYLE_VAE_ABC_REPORT.md](../../v6/STYLE_VAE_ABC_REPORT.md)。

| 插值t | crossTop1 | sameTop1 |
|---|---|---|
| 0.0 | 11.25% | 55.00% |
| 0.25 | 12.25% | 47.00% |
| 0.5 | 20.00% | 40.00% |
| 0.75 | 39.50% | 40.00% |
| 1.0 | 60.00% | 60.00% |

## C_edit 的分布与兼容性

在同150源图的swap兼容性审计中，SD0 raw latent RMS4.870，A0/S10 4.862，C_edit6.260；C_edit均值2.599，SD0约0.577。
这些分布差异与旧WEAVE失配假说一致，但RMS差异本身不构成充分因果解释。B4的2×2实验显示按CSD是Decoder效应更大，按DINO-S是Encoder负效应与Decoder正效应抵消，详见归因章。

## 实现审计的一个边界

`quant_conv` 被设为可求梯度，但实际optimizer参数组仅收model.encoder等子模块，未包含它。其权重没有实际更新，checkpoint未保存它不等于“丢失已训练权重”。
不能把这一点编造成已经定位的失败根因；未来需要明确哪些参数应该更新并保存。


**证据来源：** [tools/vae_style_vae/train.py](../../../tools/vae_style_vae/train.py)；[deliver/vae_ranked_eval/vae_swap_compat.json](../../../deliver/vae_ranked_eval/vae_swap_compat.json)。


远端原始 JSON 的位置、大小、SHA256 与压缩内容保存在 [采集快照](../appendix/live_snapshot.json)。这里的“压缩”只省略逐任务大数组，不改聚合数值。
