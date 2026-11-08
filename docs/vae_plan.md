# WEAVE Art-VAE：4090实验方案修订版

接受审查中的关键修正。保留A0/A1/A2三组，但改变训练协议和验收方式。现在有RTX 4090，可以进行完整的Encoder＋Decoder联合后训练，不再需要为了3060显存过度简化实验。

## 一、最终实验矩阵

所有实验均保持标准SD1.5 `AutoencoderKL`架构，输出仍为`4×64×64`。不添加永久风格分支，不引入DINO监督，不使用GAN。

| 实验     | Encoder | Decoder | 监督                  | 训练预算 |
| ------ | ------- | ------- | ------------------- | ---- |
| B0     | 原始      | 原始      | 不训练                 | 直接评测 |
| B1（可选） | 冻结      | 微调      | 重建＋KL               | 最多4h |
| A0     | 训练      | 训练      | 重建＋KL               | 4h   |
| A1     | 训练      | 训练      | A0＋137类Style CE     | 4h   |
| A2     | 训练      | 训练      | A0＋137类Style SupCon | 4h   |

优先完成A0、A1、A2，共12 GPU小时。 B1是用于区分Decoder瓶颈的低风险对照，可以补做，但不阻塞前三组。

三组必须从同一个`sd-vae-ft-ema`初始权重独立启动，不能把A0训练好的权重再交给A1，否则训练曝光量不公平。

## 二、4090服务器需要准备或上传什么？

既然数据集已经准备好，先检查4090服务器上有哪些资产，不要重复传输。

### 必需资产

| 资产                                | 用途                       | 获取方式                     |
| --------------------------------- | ------------------------ | ------------------------ |
| WikiArt训练图片                       | A0/A1/A2训练               | 已在4090则无需上传              |
| `all_data_info.csv`               | 137类Style、Artist、Genre标签 | 从3060复制，约13MB            |
| `sd-vae-ft-ema`完整权重和`config.json` | 三组共同初始化                  | 4090直接下载或从3060复制         |
| WEAVE仓库                           | 训练框架、评测协议、D5兼容代码         | `git clone` / `git pull` |
| LPIPS预训练权重                        | 重建训练监督与验收                | 4090缓存已有则不用传             |
| 冻结train/val清单                     | 公平比较和防泄漏                 | 首次准备时生成并固定               |

仓库现有VAE加载入口是`tools/v5/route_c.py`，使用`stabilityai/sd-vae-ft-ema`的本地ModelScope快照。因此要保证三组初始化的是同一份真实权重，而不仅仅是模型名称相同。

原始WikiArt盘点显示：

- train：79,433张，原始JPEG约36.43 GiB；
- test：23,817张，约11.20 GiB；
- 137种非空风格标签；
- 2,316位画家。

如果4090只有已经预处理的512图像，可以直接用于首轮实验，无需重新上传原始高清图。记录预处理版本即可。

### 验证额外需要的资产

需要我们原来的评测代码，但不需要旧D1或ScoreDiff的训练权重。

| 验证任务                     | 需要WEAVE代码？            | 需要旧模型CKPT？      |
| ------------------------ | --------------------- | --------------- |
| VAE重建LPIPS、PSNR、SSIM     | 仅需标准评测脚本              | 需要B0基础VAE       |
| Latent Style Probe（137类） | 需要新Probe代码和标签         | 不需要旧风格分类器       |
| KL、分布、噪声扰动稳定性            | 需要新VAE诊断代码            | 不需要             |
| CSD画风保持测试                | 需要现有CSD评测代码           | 需要CSD预训练权重及冻结原型 |
| D5五风格复验                  | 需要现有Evaluator V2和测试图片 | 不需要旧WEAVE权重     |
| 下游D1/ScoreDiff重新训练       | 需要现有训练代码与配置           | 不能直接复用旧模型权重     |

建议从仓库保留以下模块：

- `tools/evaluation_v2/`：统一评测协议；
- `tools/12h/csd_model.py`：官方CSD加载及特征提取；
- `eval_protocol/`：冻结750任务、参考集；
- `tools/v6/train_eval_v6.py`、`tools/v6/score_diff.py`：后续下游代理训练；
- `tools/vae_recon_probe.py`：已有高频重建诊断，可复用思路。

需要传到4090的额外二进制资产主要是CSD ViT-L权重、`csd_proto.pt`、LPIPS权重，以及D5测试图片。若全部能从原缓存或公开模型源获取，就不必单独上传。

不要上传旧的TensorRT VAE Engine作为新模型验证工具。 Decoder权重改变后必须重新构建Engine。首轮统一使用PyTorch VAE验证，避免新Encoder配旧Decoder的错误。

## 三、训练协议：针对审查意见的具体修改

### 1. 均值与采样混合重建

采用：

\\[ L\_{\rm rec} =0.75L(D(\mu),x) +0.25L(D(\mu+\sigma\epsilon),x) \\]

但工程实现不应每步执行两次Decoder，否则成本接近翻倍。

每张训练图片以75%概率使用posterior mean、25%概率使用posterior sample，混合成一个batch，只解码一次：

```
posterior = vae.encode(x).latent_distmu = posterior.meansample = posterior.sample()mask = (torch.rand(x.shape[0], device=x.device) < 0.25)z = torch.where(mask[:, None, None, None], sample, mu)recon = vae.decode(z).sample
```

这在随机采样期望上对应上述混合损失，同时减少计算。

下游WEAVE依然使用posterior mean，但VAE必须同时具备合理的随机邻域解码能力。

### 2. KL：先标定梯度，不预先固定β

所有损失项统一计算方式：

| 损失     | 定义              | 初始处理     |
| ------ | --------------- | -------- |
| L1     | RGB绝对误差均值       | 系数1.0    |
| LPIPS  | 图像感知距离均值        | 初始系数0.25 |
| KL     | 对潜变量所有维度取均值     | 根据梯度标定β  |
| CE     | 有标签样本平均         | 根据梯度标定λ  |
| SupCon | 有有效正样本的anchor平均 | 根据梯度标定λ  |

KL同时记录`KL_per_element`和`KL_per_image`，避免不同归约方式造成系数量级错误。

用共同初始权重在16–32个batch上测梯度，设置：

\\[ \frac{\\|\nabla\_{\theta_E}\beta L\_{\rm KL}\\|} {\\|\nabla\_{\theta_E}L\_{\rm rec}\\|} \approx 1\\%\sim5\\% \\]

这是校准起点，不是KL理论最优范围。随后A0/A1/A2使用相同β，不允许每组单独调到最优。

另外必须记录posterior方差和均值重建、采样重建的误差。如果KL逐步失控，即使LPIPS改善也不能通过验收。

### 3. A1：风格CE如何监督

保留原VAE，不增加永久结构。

从posterior mean生成一个固定的多尺度latent描述：

\\[ F(z)= [\operatorname{Pool}\_{8\times8}(z), \operatorname{Mean}(z), \operatorname{Std}(z), \operatorname{DWTStats}(z)] \\]

训练一个简单线性分类头：

\\[ L\_{\rm A1}=L\_{\rm base}+\lambda\_{\rm CE}CE(H(F(z)),y) \\]

这样比仅对4个latent通道进行GAP更合理，也能防止一个过强的临时分类器独自承担全部识别能力。

具体设置：

| 项目              | 设置                     |
| --------------- | ---------------------- |
| 类别              | WikiArt全部有效Style，约137类 |
| Head            | 固定特征描述＋线性分类器           |
| 采样              | 与A0相同，不额外重采样           |
| 类别平衡            | 仅CE温和加权，权重限制在合理范围      |
| 辅助梯度            | 初始约为重建梯度的15%           |
| Gradient Cosine | 记录与重建梯度是否长期冲突          |

分类头用梯度训练，梯度必须经过latent回到Encoder。最终只导出VAE本体。

### 4. A2：对比监督如何处理

A2独立从原始VAE初始化，只加SupCon，不加CE。

\\[ L\_{\rm A2} =L\_{\rm base}+\lambda\_{\rm con}L\_{\rm SupCon} \\]

采用MoCo式动量编码器和队列，以适应VAE训练batch较小的问题。动量编码器维护相对一致的历史特征，避免直接将持续变化的主Encoder输出长期缓存。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com



| 项目          | 设置                    |
| ----------- | --------------------- |
| 正样本         | 同Style、优先不同Artist     |
| 负样本         | 不同Style、优先相同Genre     |
| 队列          | 初始4096，显存允许可扩大        |
| Temperature | 初始0.1                 |
| 辅助梯度        | 初始约为重建梯度的10%–15%      |
| 类内变化        | 监测类内方差、effective rank |
| 无有效正样本      | 跳过该anchor的SupCon，保留重建 |

必须记录队列有效正样本率。如果大量稀有类别始终没有正样本，不能把这次实验解释成有效的全类别对比学习训练。

不使用任何随机数据增强。 A2同样直接使用不同真实画作构造正负样本。

## 四、4090训练预算如何安排？

不能再假设microbatch=4、有效batch=32一定合理。

建议使用4090的实际显存和吞吐决定配置。普通RTX 4090按24GB显存考虑；如设备实际配置不同，以检测结果为准。

| 参数                           | 建议                        |
| ---------------------------- | ------------------------- |
| 分辨率                          | 512×512                   |
| Microbatch                   | 从2开始，测试能否提升至4             |
| Effective batch              | 首选16                      |
| 精度                           | BF16 autocast，模型主权重保持FP32 |
| Optimizer                    | AdamW                     |
| Encoder LR                   | 1×10⁻⁵                    |
| Decoder LR                   | 5×10⁻⁶                    |
| quant_conv / post_quant_conv | 1×10⁻⁵                    |
| Warmup                       | 总优化步数的2%                  |
| Gradient clipping            | 1.0                       |
| 每组训练上限                       | 4小时                       |

BF16是否可用应先通过`torch.cuda.is_bf16_supported()`检查。KL、posterior统计和梯度诊断使用FP32，避免混合精度造成数值误判。PyTorch官方也明确指出AMP数值范围和梯度缩放需要谨慎处理。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 关键：固定曝光量，而不是强行训练满4小时

准备阶段先让A0/A1/A2分别运行100步，记录：

- 每次Optimizer更新耗时；
- 峰值显存；
- 数据加载等待；
- LPIPS前后向耗时；
- A2动量Encoder额外开销。

根据最慢的一组决定统一训练更新数\\(N\\)，确保三组都能在4小时内完成。

实际训练要求：

\\[ N\_{A0}=N\_{A1}=N\_{A2} \\]

\\[ B\_{\rm effective,A0} =B\_{\rm effective,A1} =B\_{\rm effective,A2} \\]

三组使用相同图片序列、随机种子和学习率调度。A0/A1提前完成的时间可用于验证，但不能偷偷多训练。

在固定512分辨率下可以使用梯度检查点降低显存；PyTorch当前建议明确使用`use_reentrant=False`。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



数据从4090本地SSD/NVMe读取，避免WSL反复访问Windows挂载盘。

## 五、验收体系：不能只看Probe

### P0：VAE可用性硬门槛

先在同一2048张独立作品上完成B0原始VAE评测，再比较A0/A1/A2。

| 指标        | 建议硬门槛                |
| --------- | -------------------- |
| 重建LPIPS   | 不超过B0的1.08倍          |
| PSNR      | 不低于B0减0.5 dB         |
| CSD原始画风保持 | 五风格平均Top-1下降不超过2个百分点 |
| 高频笔触误差    | 不得出现显著、系统性的恶化        |
| Posterior | 不允许NaN、Inf或明显方差塌缩    |
| 随机邻域解码    | 轻微扰动后不允许出现严重结构崩塌     |

这些是首轮筛选门槛。任何超限应保存失败诊断，而不是继续用更高Style Probe掩盖重建问题。

### P1：风格信息与可分性

这里必须把风格信息存在和风格容易读取区分开。

使用三种独立Probe：

| Probe        | 输入特征                | 作用                  |
| ------------ | ------------------- | ------------------- |
| Linear-Flat  | 8×8池化后的空间latent     | 判断线性读取能力            |
| Linear-Stats | Mean、Std、Gram、DWT统计 | 判断全局风格统计            |
| Shallow-CNN  | 原始4×64×64 latent    | 判断潜空间是否仍包含非线性可读取的信息 |

三种Probe均从零训练，且对B0/A0/A1/A2采用完全一致的配置。

输出137类Macro-F1、Macro Top-1、kNN检索、同异风格AUROC，并补充留画家测试。

如果A1只提高Linear-Stats，而Shallow-CNN几乎不变，说明它可能主要改善读取方式，而不一定增加潜空间包含的风格信息。

如果A2提高检索率却显著降低类内方差，说明存在过度聚集风险。

### P2：最终必须通过下游测试

这一层才决定使用哪个VAE。

选出最多两个候选，重新编码D5训练集，然后从头训练同一个轻量ScoreDiff或D1代理。

对照要求：

| 项目   | 要求               |
| ---- | ---------------- |
| 下游网络 | 完全相同             |
| 初始化  | 完全相同的随机初始化协议     |
| 训练步数 | 相同               |
| 数据集  | 相同D5划分           |
| 推理强度 | 使用预先冻结的参数网格      |
| 评测   | Evaluator V2统一协议 |

尤其要同时计算：

\\[ \Delta T\_{\rm RGB} =S_t(G(x))-S_t(x) \\]

\\[ \Delta T\_{\rm VAE} =S_t(G(x))-S_t(D(E(x))) \\]

前者衡量最终风格迁移，后者区分新VAE自身与下游编辑器的贡献。

下游挑战目标： 在内容仍可辨认的条件下，相比A0至少提高5个百分点Target Top-1，并且目标风格绝对增益不退化。特别关注Ukiyo-e与Early Renaissance。

达不到时不能宣称监督方案成功，即使137类Probe显著提高也一样。

## 六、最容易造成工程结果失真的四处问题

我检查了当前仓库代码，这些地方需要实际修改。

| 问题       | 当前情况                                    | 修正                                      |
| -------- | --------------------------------------- | --------------------------------------- |
| VAE加载    | `route_c.py`默认搜索原始`sd-vae-ft-ema`路径     | 增加显式`--vae-path`，禁止静默加载旧模型              |
| TRT      | `load_trt()`读取固定Engine目录                | 新VAE评测先走PyTorch，最终按权重重建Engine           |
| Eval缓存   | `features.py`使用固定`sources_latent_v1.pt` | 缓存key加入`vae_id`、权重SHA256及预处理版本          |
| Latent缩放 | 旧代码使用`vae.config.scaling_factor`        | 新VAE单独统计posterior mean和sample分布，校准后冻结配置 |

Diffusers对`scaling_factor`的定义是训练扩散模型前使用的潜变量缩放系数，并非VAE内部重建时应该随意乘除的值。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



训练VAE内部使用原始posterior latent：

\\[ \hat x=D(\mu) \\]

下游WEAVE才使用：

\\[ z\_{\rm downstream}=s\mu,\qquad \hat x=D(z\_{\rm downstream}/s) \\]

所以先测量新latent统计，后校准缩放；不能通过改scaling factor掩盖潜空间实际变化。

另外，所有训练checkpoint应当包含模型、Optimizer、Scheduler、EMA、随机数状态及完整配置，支持中断恢复；最终发布的则是可以由`AutoencoderKL.from_pretrained()`直接重新加载的标准模型目录。

## 七、最终执行顺序与产物

| 阶段    | 工作                  | 结果                               |
| ----- | ------------------- | -------------------------------- |
| 准备    | 资产检查、数据索引、去重、训练吞吐测试 | `manifest.json`、`benchmark.json` |
| B0    | 原始VAE完整验证           | `b0_report.json`                 |
| A0，4h | 重建＋KL               | `vae_a0/`、完整评测                   |
| A1，4h | 重建＋KL＋137类CE        | `vae_a1/`、完整评测                   |
| A2，4h | 重建＋KL＋137类SupCon    | `vae_a2/`、完整评测                   |
| 比较    | 重建、Probe、latent稳定性  | `vae_comparison.json`            |
| 下游    | 最多两个候选重新训练WEAVE代理   | 最终采用/淘汰结论                        |

最后需要上传或同步到4090的最小清单：

- 已准备好的WikiArt训练图片与必要验证图片；
- `all_data_info.csv`；
- `sd-vae-ft-ema`原始权重；
- WEAVE仓库代码；
- LPIPS权重；
- 用于最终风格验证的CSD权重、`csd_proto.pt`、D5测试图片及冻结协议。

不需要上传旧D1/ScoreDiff checkpoint，不需要旧latent缓存，不需要旧TRT Engine。 旧WEAVE代码需要保留，用来在新VAE定型后进行统一的下游训练与评测。

这次实验最关键的判断是：

A1/A2是否能够在保持正常重建和稳定潜空间的同时，让全部137类艺术风格更容易被读取，并最终让重新训练的WEAVE获得更强的目标画风。

如果A1/A2只是分类更好，而下游没有改善，就保留A0，停止往VAE里继续叠判别监督。