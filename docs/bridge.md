# WEAVE V8：完全去Score化的Schrödinger Bridge Flow实施方案

## 一、结论：可以扔掉ScoreNet，但要区分两个目标

可以彻底取消V6/V7的ScoreNet、Real/Fake Score、DMD损失和Score Difference积分器。

使用概率流匹配（Probability Flow Matching），我们可以直接学习：

\\[ \boxed{ \frac{dz_t}{dt}=u\_\theta(z_t,t,s\_{\rm src},s\_{\rm tgt}) } \\]

推理只需积分这个速度场，不需要估计任何概率密度梯度。

但是必须明确：

- 实现无Score的分布传输：可以直接做到。
- 实现数学意义上的最优薛定谔桥：还需要正确的端点耦合，或原论文的在线α-DSBM优化。
- 实现1-NFE推理：不能保证，需要验证4–16 NFE的数值积分效果。

这三件事不能混为一谈。

我们可以先用固定耦合下的随机插值，直接回归概率流；这有严格的连续性方程依据，属于Stochastic Interpolants / Flow Matching体系。再通过内容感知OT和在线桥优化，逐步接近Schrödinger Bridge。

[image](https://www.google.com/s2/favicons?domain=https://doi.org\&sz=32)

doi.org+1



我的建议是：V8完全独立于V7实现，保留旧ScoreDiff作为离线评测基线，但新训练与推理代码均不调用ScoreNet。

## 二、数学理论：究竟在训练什么？

### 2.1 从分布传输定义问题

设：

\\[ z_0\sim p_s,\qquad z_1\sim p_t \\]

分别表示源艺术风格和目标艺术风格的真实VAE latent。

我们要学习一个映射\\(T\_{s\to t}\\)，使得：

\\[ \boxed{T\_{s\to t\\#}p_s=p_t} \\]

其中\\(\\#\\)表示分布推前。

但这个条件并不能唯一确定映射。例如，完全忽略输入内容、随机输出目标画风图像，也可以满足目标边缘分布。

因此WEAVE还需要：

\\[ \min_T\ \mathbb E\_{z_0\sim p_s} c\_{\rm content}(z_0,T(z_0)) \\]

也就是在满足目标风格分布的同时，尽可能保留内容。

无配对数据不能唯一确定语义内容对应关系。 薛定谔桥通过传输代价和参考随机过程提供一种选择原则，但不会自动理解“保留原图中的人物、建筑或物体”。

这决定了我们必须分别验证分布匹配和内容保持。

### 2.2 普通Flow Matching为什么能直接学习传输？

先从最简单的直线插值开始：

\\[ z\_\tau=(1-\tau)z_0+\tau z_1 \\]

对应的路径速度：

\\[ \dot z\_\tau=z_1-z_0 \\]

训练网络：

\\[ \boxed{ L\_{\rm CFM}= \mathbb E\_{\tau,z_0,z_1} \left\\| u\_\theta(z\_\tau,\tau,s,t)-(z_1-z_0) \right\\|^2 } \\]

这里\\(s,t\\)是风格条件，\\(\tau\\)是连续时间。

MSE回归的最优解为：

\\[ u^\*(z,\tau,s,t) = \mathbb E[z_1-z_0\mid z\_\tau=z,s,t] \\]

它满足连续性方程：

\\[ \frac{\partial p\_\tau}{\partial\tau} +\nabla\cdot(p\_\tau u^\*)=0 \\]

因此，在理想训练和精确积分条件下，从\\(p_s\\)采样后沿ODE演化，可以得到\\(p_t\\)。

这已经完全不需要Score。

### 2.3 布朗桥究竟增加了什么？

参考过程采用扩散系数为\\(\varepsilon\\)的布朗运动，给定端点后，布朗桥中间分布满足：

\\[ \boxed{ z\_\tau=(1-\tau)z_0+\tau z_1 +\sqrt{\varepsilon\tau(1-\tau)}\\,\xi } \\]

其中：

\\[ \xi\sim\mathcal N(0,I) \\]

对应的逐路径时间导数为：

\\[ \boxed{ \dot z\_\tau= z_1-z_0+ \frac{\sqrt{\varepsilon}(1-2\tau)} {2\sqrt{\tau(1-\tau)}}\xi } \\]

因此不必分别训练两个漂移场，也可以直接监督概率流：

\\[ \boxed{ L\_{\rm BFM}= \mathbb E \left[ \left\\| u\_\theta(z\_\tau,\tau,s,t) - \dot z\_\tau \right\\|^2 \right] } \\]

关键性质是：

\\[ \varepsilon\rightarrow0 \quad\Longrightarrow\quad L\_{\rm BFM}\rightarrow L\_{\rm CFM} \\]

所以我们可以在同一个模型、同一个训练框架中，仅改变桥噪声，严格测量布朗桥是否有用。

这比直接实现一套复杂的双向网络更适合首轮实验。

需要特别强调：这里的速度是随机插值路径的平均概率流速度，不是声称每条随机布朗运动轨迹都服从同一个确定性ODE。

### 2.4 它什么时候才是真正的Schrödinger Bridge？

设端点联合分布为：

\\[ \pi(z_0,z_1) \\]

随机配对时：

\\[ \pi=p_s\otimes p_t \\]

用这个联合分布构造布朗桥，即使速度回归达到最优，所得过程也一般不是最优薛定谔桥。

真正的SB求解：

\\[ \boxed{ P^\*= \arg\min\_{P:\\,P_0=p_s,\\;P_1=p_t} D\_{\rm KL}(P\Vert Q\_\varepsilon) } \\]

其中\\(Q\_\varepsilon\\)是参考布朗过程。

对于标准二次传输代价，端点耦合对应熵正则最优传输：

\\[ \boxed{ \pi^\*= \arg\min\_{\pi\in\Pi(p_s,p_t)} \left[ \frac12\mathbb E\_\pi\\|z_1-z_0\\|^2 +\varepsilon D\_{\rm KL}(\pi\Vert p_s\otimes p_t) \right] } \\]

在固定参考过程及适当假设下，这是静态与动态SB的对应关系。

[image](https://www.google.com/s2/favicons?domain=https://www.proceedings.com\&sz=32)

proceedings.com



因此我们有三种层次：

| 层次   | 方法           | 数学保证            |
| ---- | ------------ | --------------- |
| V8-A | 随机配对＋CFM/BFM | 理想情况下匹配两个边缘分布   |
| V8-B | OT配对＋CFM/BFM | 在配对近似准确时改善传输代价  |
| V8-C | 在线α-DSBM     | 以原论文算法进一步逼近真正SB |

其中V8-B如果使用自定义内容代价，就不是标准欧氏布朗参考过程下的严格SB。我们是有意使用任务先验来争取更好的内容保持。

## 三、网络设计：不再使用V7的直接残差算子

### 3.1 推荐的网络结构

复用V7已经验证可以运行的轻量U-Net主干，但改变输出语义：

| 部分         | V7                  | V8                       |
| ---------- | ------------------- | ------------------------ |
| 输入         | 原始latent            | 路径中间latent \\(z\_\tau\\) |
| 条件         | Source-ID、Target-ID | Source-ID、Target-ID、连续时间 |
| 网络输出       | latent残差            | 概率流速度 \\(u\_\theta\\)    |
| 训练监督       | DMD＋风格统计            | 解析速度回归MSE                |
| 训练ScoreNet | 2个                  | 0个                       |
| 推理         | 直接残差                | ODE积分                    |
| 输出形状       | 4×64×64             | 4×64×64                  |

网络定义：

\\[ u\_\theta: \mathbb R^{4\times64\times64}\times[0,1]\times\mathcal S^2 \longrightarrow \mathbb R^{4\times64\times64} \\]

建议从`SingleStepTransportNet`的卷积主干修改：

```
z_tau [B,4,64,64]
       │
       ▼
   Conv 4→48
       │
   ResBlocks + AdaGN   ◀── Time Embedding
       │                    Source Embedding
   Downsample ×2            Target Embedding
       │
   Middle Blocks
       │
   Upsample ×2 + Skip
       │
   Conv 48→4
       │
       ▼
velocity [B,4,64,64]
```

模型宽度48，预计仍为约2M参数量级。

不需要增加Style Encoder、Content Encoder、C1条件分支或者独立的全局风格latent。

C1只在OT配对阶段使用，不进入概率流模型。

### 3.2 时间条件

连续时间使用Sinusoidal Embedding：

\\[ e\_\tau=\operatorname{MLP}(\operatorname{SinEmb}(\tau)) \\]

风格条件：

\\[ e\_{st}= \operatorname{MLP}([e_s,e_t,e_t-e_s]) \\]

最终：

\\[ e=e\_\tau+e\_{st} \\]

经AdaGN或现有StyleAdaLN注入各尺度。

必须使用连续时间，而不能将4步、8步、16步推理分别训练成不同网络。

### 3.3 不允许残差截断

V7曾使用：

\\[ T(z)=z+\operatorname{clip}(R\_\theta(z)) \\]

这使得Stable版本很容易被限制在接近恒等映射的区域。

V8直接预测速度：

\\[ \frac{dz}{d\tau}=u\_\theta(z,\tau,s,t) \\]

首轮不加输出Clamp、不使用Straight-Through Estimator。

稳定性通过梯度裁剪、学习率、规范化以及ODE步长处理。

同风格请求在推理接口中直接返回输入latent，不需要额外训练Identity Loss。注意这种硬恒等只证明接口正确，不代表跨风格内容得到了保持。

## 四、数据与配对设计

### 4.1 第一轮数据范围

继续使用现有D5真实latent，五类：

Early Renaissance、Impressionism、Minimalism、Rococo、Ukiyo-e。

共有：

\\[ 5\times4=20 \\]

个有向跨风格任务。

不要只训练Impressionism与Ukiyo-e两类后就决定整个V8的成败。

两类可以用于调试，但正式A/B实验必须覆盖全部20个方向，沿用Test600统一评测。

数据直接读取已有：

```
pseudo_gt_24h/
  orig_mean_fp16/
    00_Early_Renaissance.pt
    01_Impressionism.pt
    02_Minimalism.pt
    03_Rococo.pt
    04_Ukiyo_e.pt
```

实际路径通过现有`paths.asset_root()`解析，不硬编码磁盘盘符。

训练前校验每个缓存文件的`latents`数量、数据划分及其对应VAE的缩放系数。

### 4.2 Latent尺度必须统一

当前`tools/v7/common.py`将VAE posterior mean乘以`vae.config.scaling_factor`，V8保持相同协议。

如果：

\\[ z=s\mu_E(x) \\]

则训练、端点配对、Bridge噪声、ODE推理必须全部使用同一尺度。

不能A0用scaled latent、B1用raw latent，否则\\(\varepsilon\\)的物理含义完全不同。

也不能为了让不同风格容易配对而分别标准化各风格的均值和方差——这可能直接删除需要学习的风格差异。

### 4.3 OT配对的完整过程

对于某个有向风格对\\(s\to t\\)，随机获取：

\\[ X=\\{x_1,\ldots,x_B\\}\sim p_s \\]

\\[ Y=\\{y_1,\ldots,y_B\\}\sim p_t \\]

计算内容特征：

\\[ f_c(z)= [ \operatorname{Pool}(C_1(z)), \operatorname{SpatialSelfSim}(C_1(z)) ] \\]

特征标准化使用全部训练集共同统计量，而不是逐风格单独统计。

代价矩阵：

\\[ C\_{ij}= \left\\| f_c(x_i)-f_c(y_j) \right\\|\_2^2 \\]

使用Log-domain Sinkhorn求：

\\[ P^\*= \arg\min\_{P\mathbf1=a,\\;P^\top\mathbf1=b} \langle P,C\rangle +\eta\_{\rm OT}D\_{\rm KL}(P\Vert ab^\top) \\]

其中\\(a,b\\)均为均匀边缘权重。

实现时：

- OT batch初始为64或128；
- 只计算已经缓存的latent特征，不需要反向传播；
- 使用FP32计算代价及Sinkhorn；
- 通过有边缘覆盖约束的匹配采样形成真实端点对；
- 不使用Barycentric Average生成伪目标latent。

最后一点尤其重要：输出端点必须来自真实艺术作品，而不是多幅画作latent的平均值。

有限batch的Sinkhorn只提供近似配对，采样时也要记录实际的目标样本覆盖率，避免只使用少量容易匹配的目标作品。

### 4.4 配对质量必须先于训练验收

OT不一定有效。C1目前并不能完全去掉画风。

因此，先在一批固定跨风格图片对上检查：

| 指标        | 判断目标           |
| --------- | -------------- |
| 匹配后内容特征距离 | 比随机配对下降        |
| 目标作品覆盖率   | 不能明显降低         |
| Genre匹配率  | 尽量提高，而不是无意义聚类  |
| 真实结构视觉检查  | 相似物体、布局是否更容易匹配 |
| 风格信息泄漏    | 不能主要根据色彩区分匹配关系 |

如果OT只是把蓝色画配给蓝色画、红色画配给红色画，实际上没有解决内容对应问题，应直接拒绝这种配对代价。

另外，OT配对不保证最终ODE会实现同一组逐图配对关系。概率流ODE保证的是理想条件下的时间边缘分布，而不是随机桥的完整端点联合分布。即使端点耦合具有较低内容代价，ODE诱导的确定性映射仍可能破坏内容。

所以OT有效性最终必须通过生成图的内容保持来确认。

## 五、训练实现：核心代码与稳定性设计

### 5.1 通用Bridge路径采样器

这是V8最核心、也最值得写单元测试的部分。

```
import torchdef sample_bridge(z0, z1, tau, sigma_mid=0.0):    """    z0, z1: [B, 4, 64, 64]    tau:    [B]    sigma_mid: Brownian bridge在tau=0.5时的噪声标准差    """    tau = tau.float().clamp(1e-4, 1.0 - 1e-4)    t = tau[:, None, None, None]    # gamma(t) = sqrt(epsilon*t*(1-t))    sqrt_eps = 2.0 * sigma_mid    noise = torch.randn_like(z0)    root = torch.sqrt(t * (1.0 - t))    gamma = sqrt_eps * root    z_tau = (1.0 - t) * z0 + t * z1 + gamma * noise    # d/dt[(1-t)z0 + tz1 + gamma(t)*noise]    velocity = (        z1 - z0        + sqrt_eps * (1.0 - 2.0 * t)        / (2.0 * root) * noise    )    return z_tau, velocity
```

当`sigma_mid=0`，自动退化为CFM。

这个实现不依赖ScoreNet，也不需要目标图像的伪GT。

### 5.2 单步训练伪代码

```
z0, src_ids = source_sampler.sample(batch_size)z1, tgt_ids = target_sampler.sample(src_ids)# B2/B3在这里执行content-OT重配对if use_ot:    z1 = ot_match(z0, z1)tau = torch.rand(z0.shape[0], device=z0.device)tau = 0.01 + 0.98 * tauz_tau, v_target = sample_bridge(    z0, z1, tau, sigma_mid=sigma_mid)with torch.autocast("cuda", dtype=torch.float16):    v_pred = model(z_tau, tau, src_ids, tgt_ids)loss = (v_pred.float() - v_target.float()).square().mean()optimizer.zero_grad()loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()ema.update(model)
```

代码是核心训练逻辑示意。正式实现还需要梯度累积、学习率调度、恢复训练和数值检查。

### 5.3 时间端点问题

当\\(\varepsilon>0\\)时：

\\[ \dot z\_\tau = z_1-z_0+ \frac{\sqrt\varepsilon(1-2\tau)} {2\sqrt{\tau(1-\tau)}}\xi \\]

采样目标在\\(\tau\to0,1\\)附近方差增大。

因此首轮建议：

\\[ \tau\sim U(0.01,0.99) \\]

并记录分时间桶的回归MSE、梯度范数和ODE误差。

注意，推理最终仍需积分完整的0→1区间。仅在内部时间训练会产生端点外推误差，所以必须单独比较时间裁剪与端点训练策略。

第二阶段再研究EDM式预条件，不在B0–B3中同时引入。

博客指出原论文的预条件推导可能存在错误，但文章没有给出可直接验证的完整修正推导。工程上不能直接采用这一结论。原论文的具体公式在附录J，重新实现前应先进行解析协方差检验。

[image](https://www.google.com/s2/favicons?domain=https://www.proceedings.com\&sz=32)

proceedings.com



### 5.4 训练配置

首轮采用单卡RTX 3060 12GB，4090仍可继续其他训练任务。

| 参数              | 建议                 |
| --------------- | ------------------ |
| Latent分辨率       | 4×64×64            |
| 主干              | V7 U-Net改造         |
| Microbatch      | 从16开始测试            |
| Effective batch | 128                |
| Optimizer       | AdamW              |
| Learning rate   | 2×10⁻⁴             |
| Weight decay    | 1×10⁻⁴             |
| Gradient clip   | 1.0                |
| EMA             | 0.999，保留non-EMA对照  |
| 时间采样            | Uniform(0.01,0.99) |
| 输入增强            | 无                  |
| 每组训练预算          | 4 GPU小时            |

先实际Benchmark 100次Optimizer更新，确定相同的训练更新数\\(N\\)。

B0–B3保持：

\\[ N\_{B0}=N\_{B1}=N\_{B2}=N\_{B3} \\]

并统一源/目标采样序列、有效batch和学习率调度。

由于BFM的噪声项可能使监督方差明显增大，必须单独记录各时间段的Target RMS和预测误差；不能只比较一个整体Loss。

## 六、完整α-DSBM：什么时候需要，怎么实现？

前面的B0–B3已经可以完全去Score。但如果效果不足，才需要升级到原论文的在线Schrödinger Bridge Flow。

### 6.1 双向漂移监督

前向漂移：

\\[ b^+(z\_\tau,\tau) = \mathbb E\left[ \frac{z_1-z\_\tau}{1-\tau} \\;\middle|\\;z\_\tau \right] \\]

反向漂移，使用反向时间\\(r=1-\tau\\)：

\\[ b^-(z\_\tau,r) = \mathbb E\left[ \frac{z_0-z\_\tau}{\tau} \\;\middle|\\;z\_\tau \right] \\]

使用同一个网络：

\\[ b\_\theta(z,\tau,s,t,d) \\]

其中\\(d\in\\{+,-\\}\\)。

训练损失：

\\[ \boxed{ L\_{\rm BM} = \frac12 \mathbb E\left[ \left\\|b\_\theta^+-\frac{z_1-z\_\tau}{1-\tau}\right\\|^2+ \left\\|b\_\theta^--\frac{z_0-z\_\tau}{\tau}\right\\|^2 \right] } \\]

最终概率流：

\\[ \boxed{ u\_\theta(z,\tau) = \frac12 \left[ b\_\theta^+(z,\tau) - b\_\theta^-(z,1-\tau) \right] } \\]

这正是原论文使用的双向表示方式。

[image](https://www.google.com/s2/favicons?domain=https://www.proceedings.com\&sz=32)

proceedings.com



虽然不训练显式ScoreNet，但双向漂移实际上隐含了分布Score的信息。 对于扩散系数为\\(\varepsilon\\)的相应过程，理想情况下正、反漂移之和与\\(\varepsilon\nabla\log p\_\tau\\)有关。

因此：

- 如果目标是彻底不构建Score估计器，直接概率流已经满足。
- 如果目标是严格实现α-DSBM，则保留双向漂移，但无需单独的Score网络。

### 6.2 在线α-DSBM具体训练流程

原论文Algorithm 1包含两个阶段。

[image](https://www.google.com/s2/favicons?domain=https://www.proceedings.com\&sz=32)

proceedings.com



Stage A：预训练

使用真实的独立端点：

\\[ z_0\sim p_s,\qquad z_1\sim p_t \\]

构造Brownian Bridge，训练正、反漂移。

Stage B：在线改进端点耦合

每次迭代：

1. 从真实源分布采样\\(z_0\\)。
2. 用当前前向漂移SDE，从\\(z_0\\)生成\\(\hat z_1\\)。
3. 从真实目标分布采样\\(z_1\\)。
4. 用当前反向漂移SDE，从\\(z_1\\)生成\\(\hat z_0\\)。
5. 形成两组新端点对：\
   \\[\
   (z_0,\hat z_1),\qquad(\hat z_0,z_1)\
   \\]
6. 在这些端点之间重新采样Brownian Bridge。
7. 使用双向漂移回归损失更新网络。

在线生成的端点全部`detach()`，不需要反向传播通过SDE求解器。

每次仅做增量梯度更新，不需要重新训练一套完整网络。这是α-DSBM相对于传统DSBM的主要计算优势。

### 6.3 为什么仍然需要实验验证？

理论收敛结果针对满足相应假设的非参数α-IMF过程，不能直接推广为有限容量神经网络、有限训练步数以及近似SDE采样都一定收敛。

工程上尤其需要监测：

- 生成目标端点\\(\hat z_1\\)与真实目标分布的差距；
- 生成源端点\\(\hat z_0\\)与真实源分布的差距；
- 正反漂移一致性；
- 在线更新是否产生目标分布漂移；
- SDE积分步数对训练结果的敏感性。

建议在线SDE先使用16步和32步试验，再根据端点分布误差增加步数。原论文的部分图像实验使用100步Euler–Maruyama，16步不能预先视为足够准确。

[image](https://www.google.com/s2/favicons?domain=https://www.proceedings.com\&sz=32)

proceedings.com



## 七、推理实现：NFE如何控制？

### 7.1 直接概率流

对于B0–B3：

\\[ z\_{k+1}=z_k+h\\,u\_\theta(z_k,\tau_k,s,t) \\]

其中：

\\[ h=\frac1N,\qquad \tau_k=kh \\]

这就是Euler积分，每步仅需1次网络前向。

需要注意首末时间处的数值稳定性。若训练排除了端点，需要在评测中明确记录端点时间裁剪策略造成的误差，不能悄悄修改ODE时间范围。

### 7.2 测试不同求解器

| 方法    | 积分步数 | NFE | 目的     |
| ----- | ---- | --- | ------ |
| Euler | 4    | 4   | 极速推理   |
| Euler | 8    | 8   | 主候选    |
| Euler | 16   | 16  | 较高精度   |
| Heun  | 2    | 4   | 比较数值精度 |
| Heun  | 4    | 8   | 比较数值精度 |
| Heun  | 8    | 16  | 比较数值精度 |

Heun每步需要两次前向，因此与Euler按相同NFE比较。

双向漂移版本每次计算\\(u\\)通常需要正、反两个网络调用，因此网络前向数翻倍。

首轮验收优先关注：

\\[ \boxed{\text{8 NFE和16 NFE的内容—风格Pareto曲线}} \\]

如果16 NFE仍然远弱于V6.1，就先检查速度场学习和目标耦合，不能继续增加到96 NFE试图掩盖模型问题。

## 八、完整实验矩阵与训练预算

### 第一阶段：确定概率流是否有效

| 实验 | 配对         | 路径       | 训练 | 核心问题         |
| -- | ---------- | -------- | -- | ------------ |
| B0 | Random     | CFM      | 4h | 最基础Flow能做到什么 |
| B1 | Random     | Brownian | 4h | 布朗桥噪声是否有益    |
| B2 | Content-OT | CFM      | 4h | 配对是否改善内容保持   |
| B3 | Content-OT | Brownian | 4h | 两项改动是否互补     |

第一阶段总计16 GPU小时。

这里的4小时是每个模型的训练上限。先实测最慢组吞吐，按统一更新数和曝光量执行，避免B0实际上训练量远大于B3。

布朗桥噪声的候选强度使用：

\\[ \sigma\_{\rm mid}\in\\{0.10,0.25\\} \\]

先在训练集内部做短程筛选，再用固定设置进行正式4小时训练；不能在冻结Test600上调噪声。

### 第二阶段：判断双向桥及在线训练是否值得

只有第一阶段发现有效方向才执行。

| 实验          | 方法                | 训练预算 |
| ----------- | ----------------- | ---- |
| B4          | 双向Bridge Matching | 4h   |
| B4-Continue | 普通BM继续训练          | 4h   |
| B5          | B4基础上在线α-DSBM     | 4h   |

B4-Continue与B5必须从同一个B4 checkpoint开始，并控制额外训练预算。

如此才能回答：

> B5提升是因为在线SB优化，还是仅仅因为训练时间更长？

在线α-DSBM还有额外SDE采样开销，应同时报告真实GPU时间与网络更新次数。

## 九、验收标准：不能再出现“擦除式风格迁移”

统一使用现有WEAVE Evaluator V2。

### 9.1 评测数据

| 数据                 | 用途          |
| ------------------ | ----------- |
| Train-split独立200任务 | 配置选择、早期调试   |
| 冻结Test600          | 跨风格正式验收     |
| 冻结Same150          | 同风格恒等和推理兼容性 |
| 真实目标风格参考池          | 分布匹配评测      |

禁止在Test600上扫描噪声、OT权重、积分步数后选择最优配置。

### 9.2 主指标

| 指标           | 第一轮目标           |
| ------------ | --------------- |
| CSD ΔT       | 相比B0明显提高        |
| Target Top-1 | 提升，但不得靠图像破坏     |
| DINO-C       | 优先保持≥0.70       |
| Ukiyo-e ΔT   | 相对VAE No-op必须为正 |
| CSD-MMD      | 生成分布更接近真实目标分布   |
| 推理NFE        | ≤16             |
| 高频笔触与局部纹理    | 不出现明显退化         |

特别要分别报告：

\\[ \Delta T\_{\rm RGB} = S_t(G(x))-S_t(x) \\]

\\[ \Delta T\_{\rm edit} = S_t(G(x))-S_t(D(E(x))) \\]

第二个指标才反映编辑器相对VAE No-op真正贡献了多少目标画风。

目标风格Top-1很高，但\\(\Delta T\_{\rm edit}\le0\\)，不能视为有效迁移。

### 9.3 第一阶段Go/No-Go

建议至少满足：

- 在独立200任务上，存在B1/B2/B3优于B0的内容—风格Pareto点；
- 对目标风格绝对增益的改善具有正向配对Bootstrap置信区间；
- 没有出现明显目标类别集中、严重语义破坏或内容塌缩；
- 8–16 NFE相比4 NFE表现出合理的数值收敛趋势。

若随机桥B1和内容OT B2都没有带来可靠改善，暂时不值得进一步实现α-DSBM。

### 9.4 最终替代V6.1的目标

| 指标            | V6.1    | V8目标     |
| ------------- | ------- | -------- |
| NFE           | 96      | ≤16      |
| Target Top-1  | 27.7%   | ≥27.7%   |
| CSD ΔT        | +0.0773 | ≥+0.0773 |
| DINO-C        | 0.6803  | ≥0.68    |
| Ukiyo-e Top-1 | 6.7%    | ≥15%     |
| Ukiyo-e编辑增益   | 当前不足    | 显著大于0    |

这些是正式验收目标，不是我对首轮4小时训练结果的预测。

## 十、工程改造清单

我核对了当前WEAVE代码，建议新增V8模块，而不是在V7训练脚本上继续叠改。

### 10.1 目录

```
tools/
  v8/
    __init__.py

    data.py              # D5 latent加载、索引和划分
    normalizer.py        # 统一latent尺度
    pair_sampler.py      # Random/OT配对
    bridge.py            # Brownian路径和解析速度
    models.py            # 概率流U-Net、双向漂移U-Net
    loss.py              # CFM/BFM/Bidirectional BM
    train.py             # B0-B4统一训练
    online_dsbm.py       # B5在线更新
    solvers.py           # Euler/Heun/SDE积分
    evaluate.py          # 统一指标评测
    benchmark.py         # GPU吞吐、显存、NFE
    export.py            # 导出纯推理权重

    tests/
      test_bridge.py
      test_ot.py
      test_time_reverse.py
      test_gaussian.py
      test_ode.py
      test_data_split.py

docs/
  v8/
    V8_EXPERIMENT_PLAN.md
    V8_RESULTS.md

runs/
  v8/
    B0/
    B1/
    B2/
    B3/
    B4/
    B5/
```

### 10.2 应复用的模块

| 现有模块                         | 处理方式                          |
| ---------------------------- | ----------------------------- |
| `tools/v7/models_v7.py`      | 提取U-Net主体，改成速度输出              |
| `tools/v6/models_v6.py`      | 复用StyleAdaLNBlock和时间Embedding |
| `tools/v7/common.py`         | 抽离数据与评测工具，删除ScoreNet依赖        |
| `tools/v6/train_eval_v6.py`  | 参考C1及现有空间结构计算                 |
| `tools/v7/p3_eval_v7.py`     | 复用冻结750任务评测协议                 |
| `tools/v7/p2_train_rdmd*.py` | 不复用训练目标                       |
| `tools/v6/score_diff.py`     | 仅保留历史基线，V8不导入                 |

注意：当前`tools/v7/common.py`顶层导入`ConditionalScoreNet`。要把共享评测函数单独抽离，避免V8的依赖链继续指向旧Score模块。

### 10.3 必须通过的数学单测

这部分不要省略，V7已经出现过梯度方向与训练/推理不一致的问题。

测试A：布朗桥公式

验证：

\\[ z_0=z\_{\tau=0},\qquad z_1=z\_{\tau=1} \\]

并用有限差分验证：

\\[ \frac{z\_{\tau+h}-z\_{\tau-h}}{2h} \approx\dot z\_\tau \\]

有限差分时必须保持相同端点与相同随机噪声。

测试B：零噪声退化

当\\(\varepsilon=0\\)：

\\[ \dot z\_\tau=z_1-z_0 \\]

数值误差应接近浮点精度。

测试C：解析高斯传输

使用：

\\[ p_0=\mathcal N(m_0,\sigma_0^2), \qquad p_1=\mathcal N(m_1,\sigma_1^2) \\]

验证生成样本的终点均值、方差、Wasserstein距离以及时间边缘分布。

对于一维二次代价OT，还可以与解析映射比较：

\\[ T^\*(x) = m_1+\frac{\sigma_1}{\sigma_0}(x-m_0) \\]

但要注意：这是标准无熵OT的解析映射，不应要求有限熵随机桥的确定性概率流逐样本严格相同。

测试D：正反时间

双向模型必须满足：

- 前向时间从源域0走向目标域1；
- 反向模型的输入时间为\\(1-\tau\\)；
- PF-ODE计算使用\\(b^+-b^-\\)，不能把两个方向相加。

在解析高斯上测试漂移与概率流符号。

测试E：训练与推理一致

相同checkpoint、相同输入及条件，训练器和独立推理入口必须给出一致的速度预测。

不允许推理中偷偷加入训练未使用的残差缩放、裁剪或C1规范化。

测试F：ODE数值收敛

比较4、8、16、32 NFE的数值结果，确认增加NFE时ODE离散误差总体下降。

若出现NFE越高内容越差，需要区分数值误差与学到的速度场本身存在问题。

### 10.4 训练checkpoint和实验追踪

每次训练完整保存：

- 模型参数与EMA；
- Optimizer、Scheduler；
- 随机数状态；
- 数据索引与划分Hash；
- VAE权重Hash及scaling factor；
- 配对策略、OT正则系数、桥噪声强度；
- 实际曝光量、训练更新数；
- 每时间桶速度MSE、梯度范数；
- 推理NFE与完整验证结果。

这样可以准确重现任何一组实验，而不是只有生成PNG和最终指标。

## 十一、需要特别防范的两个研究陷阱

### 陷阱1：目标分布正确，不代表内容正确

即使概率流已经满足：

\\[ T\_\\#p_s\approx p_t \\]

仍可能把一幅建筑画转换成另一幅建筑画，甚至改变建筑主体。

这不是实现错误，而是无配对迁移本身的不可辨识性问题。

如果发生这种情况，应首先改善端点耦合和内容传输代价，而不是立即叠加新的Classifier Loss或DMD。

### 陷阱2：优化出来的ODE不等于原随机SB的端点耦合

这是很容易被忽略的数学区别。

原SB是随机路径测度，其端点联合分布通常具有随机性。

PF-ODE是确定性流，虽然可以拥有与随机过程相同的各时刻边缘分布，但不必拥有相同的端点联合分布。

因此，用OT耦合训练ODE，也不能直接保证生成输出逐图满足OT对应关系。

如果B3的目标风格分布匹配很好，内容却仍然不足，应对照真正的SDE采样，而不是认定OT配对完全无效。

## 十二、最终工程裁决

我的建议是：

正式废弃Score作为V8训练和推理的必要组件，但不立刻删除V6/V7代码。

执行路径：

| 阶段 | 工作                      | 是否需要Score |
| -- | ----------------------- | --------- |
| P0 | 数学单测＋数据与OT配对审计          | 否         |
| P1 | B0/B1：CFM与Brownian Flow | 否         |
| P2 | B2/B3：内容OT与桥路径交互实验      | 否         |
| P3 | B4：共享双向Bridge Matching  | 否         |
| P4 | B5：在线α-DSBM             | 否         |
| 部署 | 4–16 NFE概率流ODE          | 否         |

最初不需要上传额外教师模型，也不需要旧ScoreNet checkpoint。只需要现有D5 latent数据、风格标签、VAE Decoder、统一评测模型及冻结任务协议。

只有B0–B3都失败，才应该重新考虑是否需要额外的分布监督；不能因为V7此前依赖Score，就默认V8也需要它。

### 我认为最有研究价值的最终结构

\\[ \boxed{ \text{真实源/目标latent} \xrightarrow{\text{内容感知端点耦合}} \text{随机桥路径} \xrightarrow{\text{速度回归}} u\_\theta \xrightarrow[\text{4--16 NFE}]{\text{ODE}} \text{目标风格latent} } \\]

它把此前V7不稳定的Real/Fake Score分布匹配问题，转化为：

1. 端点耦合是否合理；
2. 解析速度监督是否容易学习；
3. 学到的概率流是否保持内容；
4. 少步积分是否足够准确。

这四个问题都可以独立测量、独立消融。

如果B3已经能通过验收，就应停止，不必为了“严格薛定谔桥”继续增加α-DSBM复杂度。只有B3达到瓶颈且在线改进有明确收益时，才值得实现完整版。