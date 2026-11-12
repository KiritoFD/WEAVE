# 命名、任务、指标与可比性

## 1. 易混淆名称

| 名称 | 本复盘含义 | 容易混淆项 |
|---|---|---|
| SD0 / sdvae / VAE-B0 | 原始sd-vae-ft-ema | 不是Art-A0 |
| Art-A0 / a0 | 艺术数据rec+KL后训练完整VAE | 不是原始SD-VAE |
| A1 / A2 | CE / MoCo-SupCon艺术后训练 | 不等于后面ABC |
| S10 / s10 | A0 Encoder + 10.658M剪枝恢复Decoder | 不是10M整个VAE |
| A_joint / B_style / C_edit | 联合重建/风格组织/编辑三个阶段 | 不等于Art-A0/A1/A2 |
| ArtEncoder Stage A | 通用表征500步，固定S10 | 不等于A_joint400步 |
| V6 A0 ablation | 某消融基准臂 | 不是VAE模型 |
| P0 A0 | Q2a Direct等Oracle对照中的局部臂名 | 不是Art-A0 |
| B3/B4 | WEAVE λ20/50算子蒸馏 | 不是Decoder蒸馏 |
| R1 | 约816K latent风格critic | 不等于CSD主判官 |
| Oracle1 | Q2a后加一次R1 latent梯度 | 不是最优解保证，也不是纯前向218K |
| R2 S1–S4 | 新教师/容量再蒸馏四学生 | 与V4 S1/S2不同 |
| LocalReader | 冻结latent上的DeepSets小读出 | 不训练VAE/Decoder |

## 2. 任务池与数据

D5五风格：Early_Renaissance、Impressionism、Minimalism、Rococo、Ukiyo_e。每风格30源图，共150源图。
每源图对应5目标：750=600cross+150same；cross有20种有向风格对，每对30任务，每目标120任务。
主结论用cross600；same150用于不该改变时的漂移/身份保持，不加权掺入迁移成功。

| 名称 | 实际含义 | 能回答什么 |
|---|---|---|
| Val200 | 约50源图×4cross的开发子集 | 参数选择/诊断，不是新的200独立源图 |
| Heldout400 | 另100源图×4cross | 对开发子集的扩展，但仍在反复查看任务池内 |
| Test750 | 同150源图×5目标 | 固定域配对比较；不是每项750独立样本 |
| V9 60任务 | 两个风格方向局部子集 | 桥接路线局部行为，不能当D5全600 |
| VAE2048真实图 | 艺术重建/风格几何验证 | 与WEAVE750不同 |
| S10最终256图 | 固定Encoder重建测试 | 压缩损失，不是迁移Top1 |
| 联合ABC128图 | 同系列重建对照 | 不与S10另256直接相减 |
| swap150图重建 | 同一150源图四VAE往返 | 可配对比较VAE兼容性 |
| ArtEncoder / Reader | 多风格、画家隔离、内容匹配难负、fewshot | 通用读出/几何，不是迁移主表 |

750任务SHA256：`8b578c2bc421f954fba2d76c64ed763d1a91c01522c29181d88f1231b83bbb12`。
测试图有1处历史Unicode路径字符串不同，图像字节hash一致；按hash寻址，不能以字符串不等推断缺图。


**证据来源：** [eval_protocol/tasks_750.json](../../eval_protocol/tasks_750.json)；[deliver/README.md](../../deliver/README.md)；[deliver/testset/MANIFEST.json](../../deliver/testset/MANIFEST.json)。

## 3. 评分链路家族

| 族 | 编码输入 | 主参照 | 解码 | 本复盘使用方式 |
|---|---|---|---|---|
| 早期论文/支路报告 | 各历史实现 | 原论文任务/750均值等 | 原管线 | 只在原表内比较 |
| 历史eval_tiers及旧P0/P1交付 | 部分[0,1]直接encode | 实际含VAE往返零点 | 历史TRT/缓存 | 不与新表绝对值混合 |
| 4090换VAE正式矩阵 | 正确[-1,1] | 原始RGB | PyTorch BF16 | 20格配对主证据 |
| 3060 R2 | 正确[-1,1] | 原始RGB，*_rt单列 | SD-VAE TRT fp16 | R2内部比较 |
| VAE独立重建 | 按各阶段实现审计 | 原始真实图 | 对应VAE | 同一集合内比较 |

输入归一化与评分参照是两件事；修正其中一个不等于整条链路一致。相同JSON键名paper_rgb_v1不构成实现等价证明。
本轮4090与R2分数相近但非完全相同，未做逐因素精度消融；禁止把差异直接归于模型更新。


**证据来源：** [tools/v6_24h/eval_tiers.py](../../tools/v6_24h/eval_tiers.py)；[tools/vae_swap_eval/run.py](../../tools/vae_swap_eval/run.py)；[tools/v6_next/r2_eval.py](../../tools/v6_next/r2_eval.py)；[deliver/vae_ranked_eval/REPORT.md](../../deliver/vae_ranked_eval/REPORT.md)。

## 4. 风格指标

设五类相似度为s_k，目标t、源类s，源图x、输出y。

| 指标 | 定义 | 解释边界 |
|---|---|---|
| CSD Top1 | argmax_k CSD(y)_k 是否等于t | 五原型判别；不是概率 |
| Target margin | s_t(y)−max_{k≠t}s_k(y) | 区分目标与竞争类；可因去源而改善 |
| CSD ΔT | s_t(y)−s_t(x) | 目标亲和度变化，不是成功率 |
| CSD ΔS | s_s(y)−s_s(x) | 源风格保留/抹除；same看漂移 |
| CSD ΔM | ΔT−ΔS | 相对源目标差，不是几何方向余弦 |
| DINO-S | 输出DINOv2-small CLS与目标30参考图CLS最大余弦 | 图像相似度，会受参考/内容影响 |
| CLIP-S | 输出CLIP ViT-B/32图像特征与目标30参考图归一化均值原型余弦 | 图像-图像，非文字prompt分数 |
| DINO/CLIP margin | 同判官目标分数减其他四类最大分数 | 补充目标选择性，不能混称CSD margin |
| DINO/CLIP Top1 | 同判官五类分数argmax | 补充判别，与CSD不是同一列 |
| 五类平均相似度 | 所有风格分数均值 | 排查整体相似度抬升 |
| StyleAffinity | .5(DINO-S+CLIP-S) | 人工组合，仅辅助；没有独立校准 |

均匀随机选五类的期望Top1为20%；强保源输出的cross正确率可低于20%，所以20%不是内容保留方法的物理下界。
分数为正或高于旧表不等于视觉更好。要联合检查内容、same保持、每目标均衡、真实图校准及目标相对其他类的选择性。


**证据来源：** [tools/vae_swap_eval/style_metrics.py](../../tools/vae_swap_eval/style_metrics.py)；[deliver/vae_ranked_eval/STYLE_ATTRIBUTION.md](../../deliver/vae_ranked_eval/STYLE_ATTRIBUTION.md)。

## 5. 内容、重建与效率

| 指标 | 方向 | 使用要求 |
|---|---|---|
| DINO-C | 输出与源CLS余弦越高越接近源 | raw源与往返源不可混用；不是纯几何指标 |
| LPIPS | 输出与参照距离越低越接近 | 迁移中接近源不必然更有画风；重建中才直接看保真 |
| L1/PSNR/SSIM | 重建误差/相似度 | 同分辨率、相同图集、相同归一化 |
| CSD Keep | 输出与输入CSD特征相似 | 与CSD分类Top1不同，不是85%类别保持率 |
| normalized_edge | 边缘量按空间方差等归一 | 平滑线索，不是完整频谱或风格真值 |
| 参数量 | 明确生成器/critic/Encoder/Decoder | 218K不含完整VAE与判官 |
| 训练耗时 | 实际wall clock、步数、有效batch | 不以Git时间/mtime代替 |
| 推理延迟 | 硬件、精度、bs、分辨率、是否含梯度/IO | Decoder段与端到端分开 |

## 6. 统计与证据强度

同一源图重复用于4个目标，按150源图聚类配对bootstrap，抽中重复簇要重复计入；不能用np.isin消掉重复抽样。
本轮attribution采用2000次、逐项95%CI；无多重比较校正。区间跨0写证据不足，不能当严格相等；不跨0也不自动代表视觉效应大或外域可复现。
旧报告的统计bug与新版修复各有范围，不能把旧CI无条件复制到新数值。独立风格参考与源图hash不重叠仅排除直接图像泄漏，不排除画家/题材等共享。
具体而言，4090 swap attribution的配对实现已正确保留重复簇；本地和3060的`v6_24h/stats.py`仍有np.isin问题，R2仍导入它，故R2原CI按未修正处理。

Test750被反复当研发集查看，尚缺新作者隔离盲集、域外测试与充分人工盲评。此限制对正结果和负结果同样适用。
