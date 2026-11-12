# 最新进展：通用 ArtEncoder Stage A 与局部风格读取器

本章同时保留远端采集和后来本地回传的**修复初始化种子顺序后的最终1500步版本**。正式数字优先后者，旧版本不覆盖删除。
快照保存各版本JSON SHA256，避免同名目录重跑后误拼数值。

## 数据与验收问题

Stage A针对通用风格表征，而非只优化D5：78,661 clean训练图、20个监督风格、31个未见风格；冻结S10，只训练Encoder部分。
配对困难负样本使用DINO内容相近项，跨画家/严格画家不交叠另验；固定统计读出 `[μ, logσ, G0, G1]`。
Stage A和读取器比较的是latent风格判别/检索，不是新一轮WEAVE迁移750任务。

## V18 · 通用 ArtEncoder Stage A

| 项 | 记录 |
|---|---|
| 执行侧 / 状态 | 4090 / **完成500步 / Gate A未过 / B、C跳过** |
| 要回答的问题 | 固定读出与内容匹配负样本下，Encoder能否改善通用风格几何？ |
| 实际改动 | 冻结S10；SupCon+VICReg，style梯度预算15%，GradientCache逻辑批；DINO缓存78,661×768。 |
| 任务与评测 | 500updates，原始summary107.1min、4.98img/s；GradientCache与常规梯度cos=1，relL2≈6.84e−7。 |
| 主要结果 | ref_style4.23647→4.22624（降0.01022）；AUROC0.4535→0.4566（+0.0031，低于要求+0.005）；LPIPS相对S10降0.0009；Gate=false。 |
| 本次复盘判断 | 训练与数值健康，未达几何增益门槛。B/C依门控未执行，不能列为训练失败；FINAL_REPORT.verdict为null，须按gate判读。 |


**证据来源：** [docs/v6/ART_VAE_ENCODER_REPORT.md](../../v6/ART_VAE_ENCODER_REPORT.md)；[tools/art_vae/train.py](../../../tools/art_vae/train.py)。

## Stage A同链路原始聚合值

| 模型 | AUROC | 5-shot | flat latent检索 | unseen检索 |
|---|---|---|---|---|
| a0 | 0.45350 | 0.02950 | 0.07080 | 0.12900 |
| s10 | 0.45350 | 0.02950 | 0.07080 | 0.12900 |
| A | 0.45660 | 0.02060 | 0.06780 | 0.14190 |

重建读数来自本系列同一集合：A0 LPIPS约0.0678，S10 0.08312，StageA0.08222；CSD Keep约0.95057/0.87319/0.87404。
不能拿这组值覆盖独立256图的0.07837/0.09264；样本协议不同。

### 为什么Gate没过

固定A0读出的同风格异画家正对余弦约0.7993，内容匹配异风格负对约0.8140，负对反而更近；中心化AUROC仍约0.4316，StageA约0.4336。
latent RMS4.656→4.665，未发生整体缩零。当前证据更支持读出对内容/全局统计的偏好，而不是简单尺度坍塌。
但一次500步、固定目标/梯度配额下的阴性结果，不能排除其他权重、预算或读出。旧报告“不是权重/步数问题”的强说法应收窄。

### 实施中已修正

早期梯度预算标定误用1.5%而非15%；后改为正确比例。Gate重建对照从A0改为S10，以隔离固定Decoder背景。
全量ref batch用于稳定季度均值；训练style loss与独立几何必须同时报告。

## V19 · 局部DeepSets风格读取器

| 项 | 记录 |
|---|---|
| 执行侧 / 状态 | 4090 / **正式1500步完成 / Gate R未过** |
| 要回答的问题 | 固定全局统计读出不佳，是否小型局部非线性读取器即可解决？ |
| 实际改动 | 冻结A0latent；3×3 patch(36D)共享MLP96→均值池化→rho64→L2；参数20,848；SupCon+内容匹配hard negatives。 |
| 任务与评测 | 最终回传protocol：steps1500、lr3e−4、seed0、logicalbatch64、held painters250、严格PD子集768；修复先build后manual_seed的初始化问题。 |
| 主要结果 | AUROC0.4645、PD0.4651、5shot0.0353（本轮flat0.0912）、10shot0.0353、unseen0.1395、linearprobe0.0872；Gate=false。 |
| 本次复盘判断 | 相对固定reader的配对gap改善显著，但绝对AUROC仍低于0.5，检索低于matched flat基线；不能称为通过。此路线不训练VAE/Decoder。 |


**证据来源：** [docs/v6/LOCAL_STYLE_READER_REPORT.md](../../v6/LOCAL_STYLE_READER_REPORT.md)；[tools/local_reader/readers.py](../../../tools/local_reader/readers.py)；[deliver/local_reader_gate_R.json](../../../deliver/local_reader_gate_R.json)。

## V20 · CNN读取器容量/空间对照

| 项 | 记录 |
|---|---|
| 执行侧 / 状态 | 4090 / **正式1500步完成 / 未突破** |
| 要回答的问题 | 局部不变均值聚合是否丢失空间信息，CNN能否纠正？ |
| 实际改动 | 三层卷积加全局池化，34,624参数；相同冻结latent与训练/评测协议。 |
| 任务与评测 | 与V19配对的同一正式版本。 |
| 主要结果 | 最终回传AUROC0.4770、PD0.4788、5shot0.0529、10shot0.0412、unseen0.1395、linearprobe0.0765；本轮flat0.0912。 |
| 本次复盘判断 | 困难判别略优于DeepSets，但绝对水平与检索仍未达标。不能证明所有局部/卷积读取器均不可行。 |


**证据来源：** [docs/v6/LOCAL_STYLE_READER_REPORT.md](../../v6/LOCAL_STYLE_READER_REPORT.md)；[tools/local_reader/readers.py](../../../tools/local_reader/readers.py)。

## 修复初始化后最终1500步读取器全指标

| 指标 | DeepSets | CNN |
|---|---|---|
| auroc | 0.46450 | 0.47700 |
| auroc_painter_disjoint | 0.46510 | 0.47880 |
| retrieval_5shot | 0.03530 | 0.05290 |
| retrieval_5shot_flat | 0.09120 | 0.09120 |
| retrieval_10shot | 0.03530 | 0.04120 |
| unseen_retrieval | 0.13950 | 0.13950 |
| linear_probe_final | 0.08720 | 0.07650 |
| mean_gap | -0.00352 | -0.00281 |

## 逐项Gate

| 条件 | 满足 |
|---|---|
| auroc_ge_0.55 | False |
| retr5_gt_flat_latent_0.0708 | False |
| unseen_ge_0.129 | True |
| paired_bootstrap_excludes_zero | True |
| content_matched_negatives | True |

配对gap local−fixed = +0.017418 [0.015283,0.019654]；local−CNN = −0.000711 [−0.001005,−0.000423]。
这是正负相似度间隔的差分，不是AUROC的置信区间，也不是5shot增益。

## 同步诊断：随机负样本与内容匹配负样本

| 读出 | hard AUROC | easy AUROC | 同画家异风格AUROC | 正对平均余弦 | hard负对平均余弦 |
|---|---:|---:|---:|---:|---:|
| 固定统计 | 0.4535 | 0.5375 | 0.4735 | 0.79929 | 0.81399 |
| DeepSets | 0.4645 | 0.5976 | 0.4447 | 0.95334 | 0.95533 |
| CNN | 0.4770 | 0.6098 | 0.4502 | 0.95807 | 0.95899 |

easy判别明显改善而hard仍弱，支持原生latent有可读的画风相关信息、内容匹配仍是难点。不能把hard<0.5解释成latent没有风格信息，也不能从easy提升证明已解耦题材。
最终报告记载初始化顺序修复后两次300步验证数字完全一致；本次读取其报告与最终JSON，没有自行重复训练。
few-shot测试池固定为每风格v[20:40]，上一ArtEncoder基线取v[5:15]；0.0708与0.0912的差别来自划分，而非一套模型突然性能变化。

## 版本冲突与当前边界

| 版本 | local AUROC | 5shot | flat基线 | unseen | 用途 |
|---|---:|---:|---:|---:|---|
| 较早本地报告/diagnosis_R | 0.4667 | 0.0552 | 0.0747 | 0.1395 | 历史，不覆盖正式重跑 |
| 09:58冻结的1500步远端gate_R | 0.4654 | 0.0412 | 0.0912 | 0.1860 | 修复初始化前版本，保留历史 |
| 后来本地回传最终1500步gate_R | 0.4645 | 0.0353 | 0.0912 | 0.1395 | 本章主结论，hash c5224ff7… |
| 300步复现验证 | 最终报告称两次完全一致 | 不与1500步混用 | 单独诊断 | 单独诊断 | 修复随机初始化顺序的检查 |

正式gate仍保留计划阈值flat=0.0708；本次真实matched flat=0.0912，应同时报告。两种阈值下local/CNN都未通过检索门槛。
09:58远端快照的`diagnosis_R.json`仍是0.4667旧版，与同快照0.4654的gate不同步；后来本地回传的diagnosis与0.4645最终gate同步，本章上述诊断采用后者。
读取器这次给出了“固定统计并非唯一限制，但小型学习读出仍不足”的证据，尚未回答更大读出、不同负采样或直接image teacher是否有效。


远端原始 JSON 的位置、大小、SHA256 与压缩内容保存在 [采集快照](../appendix/live_snapshot.json)。这里的“压缩”只省略逐任务大数组，不改聚合数值。
