# A05：128D/795D 风格特征有效维度诊断（2026-11-20）

> **TASK**: A05（Slot A，第三批）｜**状态**: DONE（本地 CPU 部分）+ 两项 **BLOCKED**（128D u 空间、ConvNeXt v16——远程导出命令已登记）
> 工具：[`tools/v25_cpu/a05_feature_geometry.py`](../../../tools/v25_cpu/a05_feature_geometry.py)（一键复算：`python -m tools.v25_cpu.a05_feature_geometry`，CPU-only，`OMP_NUM_THREADS=2`，无 ssh/CUDA）
> 数值：[`a05_feature_geometry.json`](a05_feature_geometry.json)（全量统计，本报告所有数字均可溯源至该文件）
> 验收依据：[agent2.md §五 A05](../../19_pull_together/agent2.md)；特征使用链见 [a02_teacher_lineage.md](a02_teacher_lineage.md)
> **DATA-03 边界**：本轮只分析、不删任何特征分支；所有"可简化"判断均为建议，需 eval-side 配对实验裁决。

## 〇、结论速览（分级）

| # | Finding | 分级 |
|---|---|---|
| F1 | 795D fiber 的**方差**极度集中（PR=6.92、熵有效维 32.5），但**判别信息**需要 ~64 维：logistic 5-way 在 PCA-k=64 达 92.0% ≈ 全维 91.8%，k=32 已 88.0% | confirmed |
| F2 | 判别几乎全部由 **DINO768 块**承载（单块 92.5% ≈ 全维 91.8%）；Oklab27 单块仅 55%（chance 20%）但**与 DINO 块互不可预测**（ridge CV R² dino→oklab 0.034、oklab→dino 0.164）——它是独立的低秩调色通道（PR=1.6，k95=5），不是判别冗余 | confirmed |
| F3 | 795D 空间内最相似的画派对是 **ER–Rococo（质心余弦 +0.42）**，最对立是 **Min–Rococo（−0.71）**；train-proxy 与 eval150 两个种群一致 | confirmed |
| F4 | 质心几何**不能**预测逐目标迁移成功率：Ukiyo-e 质心在 fiber 空间近正交孤立（最大质心余弦 +0.02），却是 PRISM-1 最差目标（Top-1 10.0%）——"分离度 ≠ 生成方向"在特征空间层面的又一例证 | suggestive（观测性对照） |
| F5 | **SG-5 One-Shot 条件向量有 1/5 失位**：ER 参考（jacopo-bellini 素描）在标准化 795D 空间最近质心是 Minimalism（0.124）而非自身（0.013），margin −0.11；其余 4 个 margin 健康（0.21~0.64）。references_v1 的 150 张参考 own-acc 93.3% | confirmed（几何事实）/ 影响待行为验证 |
| F6 | 训练代理种群 PCA-k95 基只解释 **65.4%** 的 eval 侧方差——train→eval 在外观统计空间存在实质分布位移（One-Shot 的 `fib_n` 用 train 统计标准化 eval/SG-5 特征时处于 off-distribution） | suggestive |
| F7 | 原图 JPEG 与生产路径 `decode_s10(z_src)` 的 fiber 几何几乎一致：逐图余弦 0.968（中位 0.973）、质心矩阵最大漂移 0.024、Fisher 0.2353 vs 0.2346；fp32-vs-fp16 解码 MAE 0.018/255 | confirmed |
| F8 | **128D u 空间有效维度：BLOCKED**（`fiber_state`/u 缓存在 4090）。既有间接证据：encoder holdout 5-way 97.33%（4400 训练 300 holdout） | blocked → 导出命令已登记 |
| F9 | **795D vs ConvNeXt v16 冗余：BLOCKED**（convnext_tiny 权重本地不存在）。行为侧旁证：O1→O2→O3 Top-1 .3067→.3350→.3533@t45（V24 实测）显示 V 通路与全局门各自增益 | blocked → 导出命令已登记 |

## 一、特征与消费链（为什么分析这些）

795D `Pi_parallel` fiber = `[Oklab27 | DINOv2-small patch-token moments 384|384]`（[`exp/v19/fiber_duality.py:43-73`](../../../exp/v19/fiber_duality.py)），三个消费点：

1. **StyleFiberEncoder 795→128**（`fiber_duality.py:146-178`）：PRISM-1 的 `u_vivid` 与原型 logits 的输入；4400 训练 + 300 holdout 上 holdout 5-way **97.33%**（[runs/v19/gate1_critic_and_fiber_audit.json](../../../runs/v19/gate1_critic_and_fiber_audit.json)）。
2. **One-Shot `u_glob`**（[`exp/v24/p0_o1_o3_value.py:54,76,258`](../../../exp/v24/p0_o1_o3_value.py)）：`nn.Linear(795, attn_dim)` 直接吃 **train 统计标准化的原始 795D**（`fib_n = (fib-mean)/std`），与 Q/K（灰度 DINO 16×16 布局）、V（RGB ConvNeXt v16）两条局部位通路并联，经置信度门 `R = g·R + (1-g)·u_g` 混合。
3. **BalancedVGGCriticHead** 的 27D Oklab ×3 加权分量（`fiber_duality.py:132`）。

agent2.md 的质询（"795D 与局部特征是否信息冗余、可否简化 One-Shot 条件模块"）分解为：795D 内部有效维（§3/§4，本地 done）、参考侧几何是否失位（§6，本地 done）、795D vs ConvNeXt 冗余（§8，BLOCKED）。

## 二、数据种群与保真度

全部本地（未 ssh、未 CUDA）：

| 种群 | 构成 | n | 用途 |
|---|---|---:|---|
| P1_eval150 | test750 的 150 张唯一源图；**主变体** = `n2c_z.pt` z_src 经 S10（`deliver/.../s10_p5_lowlr/decoder.pt`，CPU fp32）解码；**副变体** = `data/test` 原图 resize512 | 150×2 | 评测侧 fiber 分布 + 路径代理差实测 |
| P2_train600 | **合并池**（`weights_local/wikiart_new_512` 10,545 + `datasets/distinct5_512/train` 新增 4,694，canonical-pid 去重、wikiart 先序——镜像 [`tools/pseudo_gt/build_assets.py:47-92`](../../../tools/pseudo_gt/build_assets.py) 的池构造），排除 banned（test750 源 ∪ references_v1，151 文件）后 seeded 120/style（seed 20261120） | 600 | 训练代理（**近似** train_4400：确切成员表在远端池 manifest，本地不可复算） |
| P3_oneshot | 精确 `SG_5_STEMS`（[`exp/v21/stiefel_ref_cache.py:19`](../../../exp/v21/stiefel_ref_cache.py)）+ references_v1 全部 150 张 | 155 | One-Shot 条件侧 |

协议保真：Oklab27 块直接 `import exp.v19.fiber_duality.extract_color_fiber_27d`（逐位同源）；DINO768 块复刻 `extract_dino_rgb_moments_768d`（DINO_TRANSFORM Resize224/BICUBIC+CenterCrop224、patch token 逐 token L2 归一后取 mean|std；本地快照为 257-token 无 register 版）。**代理差已实测**（F7）：两变体几何结论一致，原图 JPEG 可作为 fiber 分析的可靠代理；CPU fp32 DINO/解码 vs 生产 fp16 CUDA 的影响在分析分辨率以下（fp16-vs-fp32 解码 MAE 0.018/255，单 latent fiber 余弦 0.999999）。

## 三、795D 有效维度（P2 训练代理，600×795）

**PCA 谱（标准化后）**：PR（participation ratio）= **6.92**，熵有效维 = **32.5**，累计方差 k90/95/99 = **90 / 153 / 317**。前 12 特征值 0.166 / 0.038 / 0.022 / 0.021 / 0.015 / 0.012 / …——方差被少数全局方向（亮度/色度尺度类）支配。

**但方差集中 ≠ 判别集中**。5-way logistic（5 折 CV，标准化）：

| PCA-k | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | full 795 |
|---|---|---|---|---|---|---|---|---|---|
| acc | .548 | .757 | .837 | .863 | **.880** | **.920** | .925 | .920 | .918 |

→ **约 64 维（8% 的维度）承载全部可测的 5 画派线性判别**；32 维已到 88.0%。逐维 ANOVA F 检验：710/795 维单独显著（89%，F>7.15），top F=294——信号弥散存在但可被低秩投影捕获。

> 对 128D 问题的含义：fiber encoder 把 795D 压到 128D 并保住 97.33% holdout 判别（gate1）——从输入空间看，**128 的容量上限不是瓶颈**（判别饱和在 ~64 维输入子空间）；128D 内部有多少维真正被 5 类原型使用，仍需 §7 的 u 空间导出。

## 四、分块结构：Oklab27 vs DINO768

| 块 | dim | PR | k95 | Fisher | Sil(cos) | 5-way acc（full） | 单独显著维 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oklab27 | 27 | **1.6** | **5** | 0.136 | −0.028 | .550 | 24/27 |
| DINO768 | 768 | 22.12 | 199 | 0.203 | 0.103 | **.925** | 690/768 |
| 全 795D | 795 | 6.92 | 153 | 0.201 | 0.102 | .918 | 710/795 |

- **Oklab27 是 ~5 维流形上的调色通道**（k95=5，PR 1.6——L/A/B 均值 + 1-2 个协方差方向主导），单独判别弱（55%）但对 5-way 无增益（.925 → .918，600 样本噪声内）。
- **块间互不可预测**：ridge CV R² dino768→oklab27 = **0.034**，oklab27→dino768 = **0.164**；折外 CCA 典型相关仅 **[0.425, 0.145, 0.073, 0.135, 0.056]**（5 折 CV；样本内 CCA 会因 27 维 vs 768 维过拟合饱和到 1.0，已弃用）。→ Oklab 块**不是** DINO 块的冗余副本，它携带 DINO moments 不含的低秩调色信息（latent chroma_head 辅助任务的输入端正是它）。
- 工程含义（建议级，非本轮改动）：One-Shot `u_glob` 若要瘦身，"PCA-64 白化"或"DINO768+低秩 Oklab"在**判别信息**上无损；但 Oklab 通道承担调色/色度功能，删除需 eval-side 配对验证（DATA-03：本轮不动代码）。

## 五、类质心几何与混淆结构

P2 标准化空间 5 类质心两两余弦（行=类）：

|  | ER | Imp | Min | Roc | Uki |
|---|---|---|---|---|---|
| **ER** | 1 | −.149 | −.636 | **+.422** | +.019 |
| **Imp** | −.149 | 1 | −.400 | +.295 | −.289 |
| **Min** | −.636 | −.400 | 1 | **−.706** | −.290 |
| **Roc** | +.422 | +.295 | −.706 | 1 | −.229 |
| **Uki** | +.019 | −.289 | −.290 | −.229 | 1 |

最相似对 **ER–Rococo（+0.42）**，最对立 **Min–Rococo（−0.71）**；eval150 同构（most-similar ER–Rococo +0.32/+0.32 原图/解码两变体，most-opposite Min–Rococo −0.61/−0.60）。Fisher 0.201 / Sil(cos) 0.102：线性可分但低裕度。

**与实测迁移成功的对照（observational）**：PRISM-1（`V19_S2_prism1_v2`，[runs/v19/measured](../../../runs/v19/measured)）逐目标 Top-1 = ER 14.2 / Imp 60.0 / Min 79.2 / Roc 58.3 / **Ukiyo 10.0%**——而 Ukiyo-e 质心恰是 fiber 空间最孤立的一类（最大质心余弦仅 +0.02）。**特征空间分离度与生成端可迁移性脱钩**，与 V10.2 铁律（Recovery Alignment ≠ Style Specificity）同向：不能用 fiber 几何替代配对评测做风格可编辑性决策。

## 六、One-Shot 条件侧几何（P3）

`u_glob` 的输入分布 = train 统计标准化后的 795D。把 SG-5 五张参考与 references_v1 的 150 张参考投到 P2 质心上（标签无关最近质心）：

| 集合 | own-acc | own-cos 均值 | margin 均值 | margin p10 |
|---|---|---:|---:|---:|
| SG-5（精确 5 张） | **0.80** | 0.368 | 0.327 | **0.018** |
| references_v1（150） | 0.933 | 0.449 | 0.383 | 0.046 |

SG-5 逐张（行=参考自身画派，列=5 类质心）：

| 参考画派 | ER | Imp | Min | Roc | Uki | 判定 |
|---|---|---|---|---|---|---|
| **ER**（jacopo-bellini 素描） | 0.013 | −.180 | **.124** | −.166 | .067 | **失位→Min**（margin −0.11） |
| Impressionism | −.335 | **.326** | .116 | −.220 | −.008 | ✓ |
| Minimalism | −.503 | −.183 | **.709** | −.501 | −.249 | ✓（最强） |
| Rococo | .049 | .051 | −.132 | **.443** | −.239 | ✓ |
| Ukiyo-e | −.071 | .093 | −.142 | −.110 | **.347** | ✓ |

**诊断**：ER 的 One-Shot 参考是一张单色素描——在 Oklab27（调色）与 DINO moments 上都不携带 ER 的颜色签名，几何上落入 Minimalism 半空间。这与 ER 同时是 PRISM-1 最差迁移目标（14.2%）相容：**该参考为 One-Shot 提供的全局外观条件是退化的**。影响（是否实际拉低 ER One-Shot Top-1）需行为配对实验；替换参考或对参考做 palette 门控是候选修复，均超出本轮边界。

分布位移（F6）：P2 的 PCA-k95 基在 P1 上只解释 **0.654**（0.6536）方差——eval 源图的 fiber 有 ~1/3 方差在训练代理主子空间之外；`fib_n` 的 train 统计在 eval/SG-5 侧是 off-distribution 的，这给"u_glob 在 eval 时的行为漂移"提供了一个量化入口。

## 七、128D u 空间：BLOCKED（导出已登记）

本地不存在 `fiber_state`/u 缓存（`runs/v19/cache/v19_foundation.pt`、`runs/v21/cache/v21_stiefel_ref_cache.pt` 均 4090/3060 独占；本地 v19/v21 目录无 .pt）。既有间接证据：gate1 fiber holdout 97.33%（§一）。

**G 队列待办（4090，一条命令）**：

```bash
ssh 4090 "cd /root/Workspace/WEAVE && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -m tools.v25_cpu.a05_feature_geometry --export-u > runs/v25/cache/a05_export_u.log"
```

产物 `runs/v25/cache/a05_u_export.npz`（u_tr/u_ho/u_750 + sid/ids + tr_fib/ho_fib + protos，约 8 MB；复用已缓存的 `v19_foundation.pt`，不重跑特征提取）。回传本地后一键补全：

```bash
python -m tools.v25_cpu.a05_feature_geometry --u-npz runs/v25/cache/a05_u_export.npz
```

判读计划（脚本已实现）：u128 的 PCA 谱/PR/熵维、5 类质心余弦、Fisher、acc-by-k 曲线、与 fiber795（tr_fib 精确 4400）Fisher 对照、protos 5×5。判定"128 维里多少维承载判别"看 acc-by-k 饱和点与 PR。

## 八、795D vs ConvNeXt v16：BLOCKED（导出已登记）

本地无 `convnext_tiny-983f1562.pth`（hub/eval_cache/weights 三处均无），无法在 CPU 复算 v16 特征。行为侧旁证（[runs/v24/p0_o1_o3.json](../../../runs/v24/p0_o1_o3.json)）：t45 下 O1（V=灰度 DINO）Top-1 .3067 → O2（V=RGB ConvNeXt）.3350（+2.8pp）→ O3（+795D 全局门）.3533（+1.8pp）——两条通路各有增量，但这不能区分"互补"还是"同一信息的两次使用"。

**3060/4090 待办**：

```bash
python -m tools.v25_cpu.a05_feature_geometry --export-cnx   # 产出 runs/v25/cache/a05_cnx_export.npz（train_4400 池化 192-D + sids/pids）
```

回传后 `--cnx-npz` 自动执行：按 canonical pid 对齐 P2 种群 → CCA/ridge R²、Fisher 对比、fiber-only vs cnx-only vs concat 的 logistic 对照（脚本已实现，合成数据自测通过）。判读：concat 增益 < 1pp ⇒ 两通路信息高度重叠，One-Shot 的 V 或 u_glob 之一可简化；≥ 2pp ⇒ 互补保留。

## 九、局限与风险

1. **P2 是分布级代理**：确切 train_4400 成员由远端池 manifest 决定（含 `_h10` 抽样与 seeded randperm），本地不可复算；600 样本对 PCA/logistic 的抽样噪声 ~±1-2pp。§7 导出的 `tr_fib` 可提供精确 4400 版复算。
2. **DINO 快照为 257-token 无 register 版**；若 4090 侧 eval_cache 是 261-token register 版，patch token 有细微系统差（协议均有兼容分支），跨机器比对 fiber 绝对值时需注意。
3. fp16 探针仅 1 个 latent（CPU fp16 解码 ~100× 慢）；MAE 0.018/255 支撑"fp32 代理可行"，但非逐图统计。
4. F4/F5/F6 均为**几何→行为**的单向观测，不能反向归因；任何"删通路/换参考"决策必须走 A01 配对 CI。
5. refs150 仅聚合统计（逐画派 own-acc 未入 JSON，如需可由脚本再跑一次获得——`p3_oneshot` 段的 `_centric` 输入已按画派分组）。

## 十、复算

```bash
# 本地全量（~28 min，CPU-only；--skip-fp16-check 省 15 min 探针；--skip-s10-decode 省 9 min）
python -m tools.v25_cpu.a05_feature_geometry
# 守卫
python -m pytest tests/test_no_absolute_paths.py tests/test_markdown_links.py -q
```

输出：`docs/v25/audits/a05_feature_geometry.json`（含 meta.protocol_notes 全部协议声明、p2_pids/p1_source_stems 种群清单，供与远程导出对账）。
