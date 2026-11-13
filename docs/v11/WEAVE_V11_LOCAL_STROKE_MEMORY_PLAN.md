> **状态（2026-11-14）：已延期。** 用户决定：先做本文 §六 P0–P2 的推理级局部匹配
> （作为 v10.1 执行，见 `docs/v10_1/`），本文的 P3–P5（32×32/64×64 局部风格
> Attention 分支训练）整体排后。本文档即 v11 方案底稿（源自 `_tmp/h2s_cmp/StyleGallery.md`）。
> 注意：其 P0–P2 已由 v10.1 吸收实现，P3–P5 待局部匹配验证有效后再启动。

# **StyleGallery 的核心机制，以及 WEAVE 应该如何吸收**

**读完 CVPR 2026 论文和官方推理代码后，我认为最值得 WEAVE 学习的不是 Attention 本身，而是：先找到语义对应的区域，再只向这些区域施加合适的局部风格约束。**

但还有一个关键区别：

**StyleGallery 依靠预训练 Stable Diffusion 已有的视觉特征来迁移笔触；WEAVE 必须从真实画作中学习可执行的局部笔触表示。**

不能简单把 StyleGallery 的 Attention 损失移植过来。

## **一、StyleGallery 究竟怎么做的？**

论文：[StyleGallery · CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/He_StyleGallery_Training-free_and_Semantic-aware_Personalized_Style_Transfer_from_Arbitrary_Image_CVPR_2026_paper.html?utm_source=chatgpt.com)

源码：[官方 GitHub 仓库](https://github.com/iiiiiiiword/StyleGallery?utm_source=chatgpt.com)

### **第一步：无监督识别语义区域**

它先使用 SD1.5 对内容图和风格参考图分别进行约 15 步 DDIM 反演，提取 UNet 中间特征：

\\[ F\_{\mathrm{mix}}=\sum_t w_tF_t \\]

再对特征执行 PCA、K-means 聚类和区域合并，得到最多约 10 个语义区域。

这一步区分的是天空、人物、建筑、植被等视觉区域，**不是直接识别笔触**。

代码对应 `pipeline.py` 中的 `forward_process()`、`feature_cluster()` 和 `generate_mask()`。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com



### **第二步：从风格画作中寻找对应区域**

StyleGallery 使用三种特征判断区域应该如何匹配：

| 匹配依据             | 权重      | 作用           |
| ---------------- | ------- | ------------ |
| UNet 区域统计        | 0.25    | 局部外观         |
| **DINOv2 语义相似度** | **1.0** | **寻找语义对应区域** |
| 区域几何信息           | 0.125   | 辅助匹配         |

形式上：

\\[ S(i,j)=0.25S\_{\mathrm{stat}}+ S\_{\mathrm{semantic}}+ 0.125S\_{\mathrm{geometry}} \\]

每个内容区域从一张或多张风格画作中选择最合适的参考区域。

**这是它相对于全局风格统计最重要的区别。**

它不会不加区分地让人物、天空、建筑匹配同一幅风格参考图的整体纹理。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com



### **第三步：区域级 Attention 风格约束**

这是产生明显局部风格化的关键。

从生成图中取 Self-Attention 的 \\(Q,K,V\\)，从对应的风格区域取参考 \\(K_s,V_s\\)。

StyleGallery 构造一个区域风格目标：

\\[ A\_{\mathrm{style}} = \operatorname{Attn}(Q_c,K_s,V_s) \\]

同时计算生成图自身的 Attention 输出：

\\[ A\_{\mathrm{gen}} = \operatorname{Attn}(Q,K,V) \\]

通过区域掩码，使二者只在对应区域比较：

\\[ \boxed{ L\_{\mathrm{RSL}} = \sum_r \left\\| M_r\odot A\_{\mathrm{gen}} - M_r\odot A\_{\mathrm{style}} \right\\|\_1 } \\]

再增加全局内容约束：

\\[ L\_{\mathrm{GCL}}=\\|Q-Q\_{\mathrm{content}}\\|\_1 \\]

最终：

\\[ L=L\_{\mathrm{RSL}}+\lambda_cL\_{\mathrm{GCL}} \\]

作者使用这个损失对生成 latent 执行 Adam 优化。标准配置为 150 步，LCM 加速版本为 28 步。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com+1



**注意：它并没有训练专门的笔触网络。** 它使用 SD1.5 已有的 Attention 表征，让输出在对应区域呈现参考画作的局部视觉特征。

### **消融证明：区域风格约束是有效的**

论文 Table 2：

| 设置                  | LPIPS ↓    | FID ↓     | Style ↑    |
| ------------------- | ---------- | --------- | ---------- |
| **完整 StyleGallery** | **0.3716** | **16.89** | **0.5337** |
| 去掉区域风格损失 RSL        | 0.5195     | 23.56     | 0.4387     |
| 去掉全局内容损失 GCL        | 0.6822     | 30.83     | 0.4150     |

完整模型优于两个删减版本。论文还通过视觉消融展示，不使用区域掩码会产生错误纹理泄漏。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com



但这还不能精确分辨全部收益究竟来自语义匹配、预训练 SD 特征，还是梯度优化本身；这些因素需要在 WEAVE 上独立验证。

## **二、源码里有两个细节值得特别注意**

我检查了官方 `pipeline.py` 和 `utils.py`，有两个实现细节会影响我们怎样借鉴。

**第一，StyleGallery 实际上使用的是一种停止梯度的 Attention 目标。**

它的核心代码相当于：

```
target_out = F.scaled_dot_product_attention(    q * 1.5, style_k, style_v)loss = F.l1_loss(    generated_attn_out,    target_out.detach())
```

参考 Attention 输出作为当前更新的目标，而不是通过它反向传播。

这说明我们不一定需要显式估计完整的风格概率密度。**一个能够根据当前内容动态产生局部风格目标的模块，也可能提供有效的优化方向。**

**第二，代码中的“稀疏 Attention”并不是严格的 Masked Softmax。**

`get_style_loss()` 主要通过乘掩码，将非对应区域的 \\(Q,K,V\\) 置零，然后调用 SDPA。零向量仍然参与 Softmax 归一化，不等价于完全排除无关 Key。

因此我们不应该照抄这部分实现。WEAVE 如果采用区域 Attention，应使用真正的有效位置掩码，并按区域有效像素数归一化损失。

还有一个值得注意的地方：区域统计匹配中使用的八层 Transformer 在代码里临时初始化，没有加载预训练权重。它不是一个已经训练好的笔触编码器。DINOv2 语义项权重最高，所以更应把该模块视为区域匹配的辅助统计，而不是 StyleGallery 成功的核心。

代码位置：

- [pipeline.py：聚类、区域匹配、优化](https://github.com/iiiiiiiword/StyleGallery/blob/main/pipeline.py?utm_source=chatgpt.com)
- [utils.py：Attention 特征与风格损失](https://github.com/iiiiiiiword/StyleGallery/blob/main/utils.py?utm_source=chatgpt.com)

## **三、WEAVE 应该学习什么？**

我建议保留 StyleGallery 的机制，但改变它的实现形式。

| StyleGallery          | WEAVE 对应改造               |
| --------------------- | ------------------------ |
| SD1.5 中间特征划分区域        | 使用 B4 的 16/32/64 多尺度内容特征 |
| DINOv2 匹配内容/风格区域      | 从真实训练画作建立语义条件化风格库        |
| 直接使用参考图 Attention K/V | 学习画派级局部笔触 Key/Value      |
| 区域 Attention 风格损失     | 区域笔触关系损失及风格梯度            |
| 150/28 步反向优化          | 小模型前馈生成 + 1–2 步可选精修      |
| 需要具体风格参考图             | 推理只需要 Style-ID           |

**最核心的区别：StyleGallery 的笔触知识存在于预训练 SD 模型和输入参考图中；WEAVE 要把这些知识压缩成自己的局部风格表示。**

### **1. 首先建立 Local Style Bank**

WEAVE 已有五个画派的大量真实训练画作，以及 V6 的 Haar 多尺度特征。因此不需要从零建立图像表示体系。

对于每幅真实画作，分别提取：

\\[ k_r=E\_{\mathrm{content}}(c_r) \\]

\\[ v_r=E\_{\mathrm{stroke}}(z_r) \\]

其中 \\(k_r\\) 描述局部内容结构，\\(v_r\\) 描述局部绘画特征。

建议将笔触分为三类尺度：

| 尺度        | 学习内容                 | 主要作用        |
| --------- | -------------------- | ----------- |
| 16×16     | 区域布局、粗略语义            | 决定风格作用的位置   |
| **32×32** | **笔触方向、色块边界、局部纹理关系** | **主要风格匹配层** |
| 64×64     | 线条细节、精细纹理            | 细化笔触        |

风格特征不能仅仅使用 Haar 均值和方差。此前 V6 已经证明，这种全局统计监督的增益有限。

应增加**局部特征之间的关系**，例如邻域方向相关性、多尺度纹理共现以及局部 Attention 关系。

这种表示的目标不是判断“这里有多少高频能量”，而是判断“这些线条和纹理如何组织”。

### **2. 用内容结构选择目标画派的笔触**

对五个画派分别建立风格记忆：

\\[ \mathcal B_t=\\{(k_j,v_j)\\}\_{j=1}^{N_t} \\]

生成时，当前区域的内容特征 \\(q_r\\) 查询目标画派风格库：

\\[ a\_{rj} = \operatorname{softmax}\_j \left( \frac{q_r^\top k_j}{\tau} \right) \\]

得到对应的局部风格目标：

\\[ \boxed{ v_r^\*=\sum_j a\_{rj}v_j } \\]

这一步对应 StyleGallery 的区域匹配，但不需要在推理时运行 SD1.5 或 DINOv2。

需要避免直接平均彼此不兼容的笔触模式。第一阶段应比较 Top-1、Top-K 和多模式采样；否则容易出现 WEAVE 目前的笔触模糊问题。

还应增加一个**无条件风格库**作为对照，严格验证内容条件是否真的有收益。

### **3. 真正让 WEAVE 学会生成笔触**

当前 B4 的 C3UNet 已经具有：

- 16×16 bottleneck；
- 32×32 decoder；
- 64×64 decoder；
- 32×32 DWT 输出头；
- Style-ID AdaLN。

这是很适合插入局部风格模块的架构。

尤其是 `tools/v6/models_v6.py` 中的：

```
u32 = self.dec32_1(u32, cond, i32)u32 = self.dec32_2(u32, cond, i32)u64 = self.dec64_1(u64, cond, i64)u64 = self.dec64_2(u64, cond, i64)
```

我建议优先在 **32×32 decoder** 位置增加局部风格记忆 Attention：

\\[ A_r=\operatorname{Attn}(W_qh_r,K_t,V_t) \\]

再通过零初始化残差注入：

\\[ \boxed{ h'\_r=h_r+\alpha_rW_oA_r } \\]

其中：

- \\(K_t,V_t\\) 来自目标画派的真实笔触风格库；
- \\(\alpha_r\\) 是局部风格强度门控；
- \\(W_o\\) 零初始化，保证初始模型严格等于 B4。

随后在 64×64 decoder 增加一个较小的 Depthwise Conv 残差分支，用于恢复细线条和高频笔触。

**为什么优先 32×32？**

因为 16×16 更偏向整体布局，而 64×64 主要负责细节。32×32 是目前最值得检验的中尺度笔触工作点。

不过，真正最优的插入层仍应通过 16-only、32-only、64-only 和联合消融确定。

## **四、训练不能只是复制风格 Patch**

这是我认为整个方案最容易犯错的地方。

StyleGallery 有现成的参考画作和大型预训练模型，因此可以直接构造区域 Attention 目标。

WEAVE 没有这些条件。我们必须自己学习画派笔触。

**建议采用真实画作的自监督笔触恢复，而不是用 StyleGallery 的输出直接当伪 GT。**

对于真实画作 \\(z\_{\mathrm{real}}\\)，自动构造削弱局部笔触后的版本：

\\[ z\_{\mathrm{weak}}=\mathcal D(z\_{\mathrm{real}}) \\]

训练局部风格模块恢复真实画作中的笔触：

\\[ \mathcal L\_{\mathrm{stroke}} = \left\\| \Psi\_{\mathrm{local}}(z\_{\mathrm{pred}}) - \Psi\_{\mathrm{local}}(z\_{\mathrm{real}}) \right\\| \\]

但这里要设置三个限制。

**首先，参考笔触必须来自另一幅真实画作。** 不能从当前目标图直接检索自己的 Patch，否则容易退化为局部复制。

**其次，监督不能只靠重建误差。** 应同时约束目标画派的局部笔触关系分布，让模型学到可跨画作迁移的规律，而不是简单的锐化或反模糊。

**第三，内容结构要有独立约束。** 对低频形状、主要边缘和区域边界进行保护，允许局部纹理变化，但不允许靠移动或删除物体完成匹配。

这一训练仍然不需要人工标注、RLHF、GAN 或 RGB 重建损失；可以在现有 VAE latent、Haar 子带及局部特征空间中完成。

不过，自监督笔触恢复不等价于真正的跨风格迁移。必须再做跨画派测试，检查模型是否只是学会了反退化。

## **五、推理时如何使用局部梯度？**

这部分保留 LD4 已证明有效的方向选择思路，但换掉其主要风格目标。

当前 LD4 优化的是分类器方向与已有风格统计方向。

新模型应优化：

\\[ L\_{\mathrm{style}} = \sum_r \omega_r \mathcal D\_{\mathrm{stroke}} \left( E\_{\mathrm{stroke}}(z_r), v_r^\* \right) \\]

计算：

\\[ g_r=-\nabla\_{z_r}L\_{\mathrm{style}} \\]

然后复用 LD4 的内容约束与回溯选步机制：

\\[ z\_{k+1}= \operatorname{Proj}\_{\mathcal C} (z_k+\eta g\_{\mathrm{safe}}) \\]

这里的关键区别是：

**梯度来自“真实目标画派的局部笔触应该是什么样”，而不是“怎样提高 R1 的目标画派分类分数”。**

训练完成后，风格模块先进行一次前馈修正，再保留 1–2 步局部梯度下降作为可选精修。

这既保留了 WEAVE 的低延迟优势，也给困难画派提供额外的编辑自由度。

## **六、建议的实验顺序**

考虑 4070 Laptop 约 6GB 显存，我不建议一开始就端到端训练整个新系统。

| 阶段     | 实验                      | 核心判据                 |
| ------ | ----------------------- | -------------------- |
| **P0** | 从 B4 多尺度特征自动聚类          | 区域是否具有稳定的语义一致性       |
| **P1** | 无条件 vs 内容条件局部风格库        | 内容匹配是否改善局部笔触目标       |
| **P2** | 冻结 B4，仅使用新局部风格梯度        | 是否优于 LD3，避免纹理模糊与内容损伤 |
| **P3** | 加入 32×32 局部风格 Attention | 能否产生新的、组织更好的笔触       |
| **P4** | 增加 64×64 细节分支           | 高频线条是否进一步改善          |
| **P5** | 前馈生成 + LD4 式梯度精修        | 是否获得独立的质量—速度收益       |

训练时先冻结 B4，并缓存中间特征，以 Microbatch 1–2 训练新增分支；等证明确有收益，再考虑联合微调部分 decoder。

评估时，五个画派都保留，但重点检查你刚才展示的三种典型问题：

**Early Renaissance → Impressionism：** 能否生成方向一致的局部笔触，而不是模糊人物。

**Early Renaissance → Ukiyo-e：** 能否形成清晰轮廓和平坦色块，而不只是增加锐化线条。

**Minimalism → 其他风格：** 能否从接近纯色的输入中生成必要的风格结构。这类任务可能超出局部笔触模块的能力，应单独归类，不应将失败隐藏在平均分里。

## **七、对 StyleGallery，我们真正能形成的超越是什么？**

StyleGallery 的优势来自：

\\[ \text{预训练扩散特征} + \text{参考图区域匹配} + \text{多步梯度优化} \\]

而 WEAVE 可以研究：

\\[ \boxed{ \text{真实画派局部笔触记忆} + \text{轻量前馈风格生成} + \text{少步受约束优化} } \\]

两者的核心研究问题相似，但解决方式不同。

我们不需要在 6GB 推理时加载 SD1.5、DINOv2，也不需要执行几十至上百步梯度优化；Style Bank 可以在训练后冻结并压缩。

另外，不建议直接使用 StyleGallery 生成结果蒸馏 B4。你已经看到它有明显的内容泄漏和场景重构问题，直接蒸馏容易把这些错误也学进去。

**优先学习它的局部匹配机制，而不是复制它的输出图像。**

## **最终建议**

下一阶段应当优先做 **P0–P2：验证 WEAVE 是否能从真实画作中获取正确的局部笔触目标**。

一旦确认局部方向确实优于 LD3，再投入 P3，训练新的多尺度局部生成分支。

最值得作为核心创新的不是“StyleGallery + 轻量化”，而是：

**让小型 Style-ID 生成器从真实画派的局部笔触分布中选择、学习并执行与源图结构相容的风格变化。**

这才有机会把 WEAVE 当前“产生绘画纹理”的能力，提升为“产生具有明确目标画派规律的笔触”，同时保留它相对于 StyleGallery 的效率优势。