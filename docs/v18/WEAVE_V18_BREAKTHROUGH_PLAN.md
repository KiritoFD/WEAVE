# WEAVE V18 · 架构级突破预注册方案：去风格结构流（CCF）与校准摊销（LD4+）

**日期**：2026-11-17 · **执行机**：RTX 3060 12GB（单卡严格分进程串行，tmux 持久化）  
**评测基准**：`eval_protocol/tasks_750.json`（Cross600 主读数 + Same150 恒等控制，5 画派各 120 Cross 任务）· **S10 解码器**（唯一口径）· `src/eval/measure.py` 冻结裁判。

---

## 1. 为什么必须从架构层面破局？（V15–V17 的终局定论）

V15–V17 共 40+ 个 Test750 实验臂已经把“自配对 Flow + $\tau_0=0.7$ SDEdit”路线探索至物理极限：
- **源画风直通死锁（Source-Style Lock）**：在 $\tau_0=0.7$（$\text{SNR}=5.44$）下，$z_{0.7} = 0.7 z_{\text{src}} + 0.3\varepsilon$ 将 70% 源图色彩与笔触统计直接喂给仅在自配对数据上训练的去噪网络，导致 **58%–60% 的输出被 CSD 直接判回源画派**；扩容至 34.2M（`S6`）反而使 Top-1 从 `10.67%` 跌至 `6.67%`。
- **多数类先验吞噬（Prior Collapse）**：训练池 57.8% Impressionism 不平衡使 `V15-96` 即便在 $\tau_0=0.0$ 纯噪声下生成 `Early Renaissance` 与 `Ukiyo-e` 的比例也为 **0.0%**（>93% 塌缩为 Impressionism）。
- **单向度指标幻觉**：缺乏显式去风格结构通道时，降 $\tau_0 \to 0.4$ 虽解锁风格，却使 `DINO-C` 从 `0.75` 暴跌至 `0.44`；而直接传未彻底去风格的 latent（如 `PT2_P3`）则立即触发恒等复制捷径（`Top-1 = 1.8%`）。

与此同时，全仓已有三块被严格证实有效的核心资产：
1. **谱系 A（`B4 / N2C / V9-LD4`）的去风格起步 + 局部约束编辑**：`V9 LD4` 已达 **35.17% CSD Top-1 / 0.7718 DINO-C / 0.4872 LPIPS**，距外部强基线 Seedream 4.5（`37.3%`）与 StyTr2（`39.5%`）仅差 **2–4pp**，且在 DINO-S/DINO-C/LPIPS 三项上已全面领先！
2. **V17 `P2_S2` 的 DINO Patch 上下文单步摊销**：12k 步 + 冻结 DINOv2 16×16 上下文使 0.3M 单步 Adapter 100% 吸收并反超局部教师。
3. **V15 纯噪声 Flow（$\tau_0=0.0$）的高画质生成能力**：`S6` 在 $\tau_0=0.0$ 达 **49.3% Top-1 / 0.591 DINO-S**，证明轻量 Conv Flow 具备极强的绘画纹理由无到有生成能力——唯一缺的是**零色彩/零笔触泄漏的显式空间结构注入**。

---

## 2. V18 双轨突破设计（Track A 刷新可交付帕累托 + Track B 纯噪声结构流破局）

### Gate 0：去风格结构表征 $S(x)$ 泄漏与充分性硬门禁（`exp/v18/structure.py`）

构造完全确定性、无可学习捷径的双尺度去风格结构条件 $S(x) = (C_{64}, \tilde{P}_{16})$：
1. **色彩数学归零 + 多尺度局部对比度结构图 $C_{64} \in \mathbb{R}^{4 \times 64 \times 64}$**：
   - 源图 RGB 转纯亮度 $Y = 0.299R + 0.587G + 0.114B$（**100% 消除色相与饱和度通道相关性**）；
   - 提取四通道几何特征：① 细尺度局部对比度归一化 $\text{LCN}_{k=5}(Y_{64})$；② 粗尺度明暗体块 $\text{LCN}_{k=13}(Y_{64})$；③ 经 $\sigma=1.0$ 平滑滤除微笔触后的各向同性 Sobel 边缘幅值（按每图 95% 分位数归一化，消除硬边/软边画派幅度差）；④ 归一化拉普拉斯轮廓 $\Delta(Y_{64})$。
2. **空间实例白化灰度 DINO 语义布局 $\tilde{P}_{16} \in \mathbb{R}^{384 \times 16 \times 16}$**：
   - 将灰度三通道图 $[Y, Y, Y]$ 送入冻结 DINOv2-small 提取 $16\times 16$ patch tokens $P$；
   - 对每幅图逐通道减去空间均值并除以空间标准差（`InstanceNorm2d`），再逐 token L2 归一化：
     $$\tilde{P}_{b,c,h,w} = \text{Normalize}_{\text{ch}}\!\left(\frac{P_{b,c,h,w} - \mu_{b,c}}{\sigma_{b,c} + \epsilon}\right)$$
   - **数学性质**：每幅图在任意通道上的全局均值恒为 $0$、空间方差恒为 $1$——全局 Gram/CLS 风格统计被精确归零，仅保留“人脸、天空、建筑、树木、衣纹在哪里”的纯语义拓扑布局！
3. **预检断言**：在留出集上对比 `raw z_src`、`c1_gentle(z)` 与 `S(x)` 的 5-way 风格探针准确率，验证全局统计泄漏降至随机水平（~20%），空间探针泄漏大幅下降且布局自相似相关性保持高位。

---

### Track A：校准型 `LD4+` 算子升级与 `N2C/B4` 单步极速摊销（`exp/v18/ld4_plus.py`）

直接冲击 **>38%–42% CSD Top-1** 并全面支配 Seedream 4.5 与 StyTr2：
1. **发现的 V9/V17 算子瓶颈**：
   - V17 校准诊断揭示冻结 `R1` critic 存在显著类别偏置（留出集平均 margin：ER `-3.31`、Imp `-3.65` vs Uki `-4.48`），导致 `LD4` 的候选接受函数对 Ukiyo-e / Minimalism 欠激励；
   - 原 `LD4` 的 `r1` 方向仅对目标 margin 求导，未显式压制源画派特征（V17 `T2` 虽提出 `Target − Source`，但因误用 L2 归一化缩小了 128 倍而未生效）。
2. **实验臂设置**：
   - **`V18_A0_LD4_repro`**：复现 `N2C` 起点上的标准 `LD4`（同轮 3060 锚点校验）。
   - **`V18_A1_LD4plus`**：在 `N2C` 起点上引入 **RMS 归一化的 `Target − 0.35·Source` R1 梯度** + **独立留出集校准的去偏 proxy** + 适度内容预算扩展（$\varepsilon=0.16, \rho=0.20, K=3$，Top-30% 风险门控掩码）。
   - **`V18_A2_FastStudent`（纯前馈单步蒸馏）**：在 500×5=2,500 条严格去重、五派均衡的 clean 训练对上，用 V17 验证成功的 **灰度白化 DINO-Context + `u_tgt - u_src` 零恒等残差 Adapter**（0.35M 参数，8,000 updates）直接蒸馏 `B4 → LD4plus` 映射，实现 **零测试期 R1 反传、单次前馈 + S10 解码** 的高速度高特异模型（配对评测 `V18_A2_FastStudent` 与 `V18_A2_FastStudent_wrong`）。

---

### Track B：去风格结构条件生成流 `CCF-UNet`（从 $\tau_0 = 0.0$ 纯噪声起步，`exp/v18/ccf_model.py`）

彻底抛弃 SDEdit $\tau_0=0.7$，让生成流从 **100% 纯高斯噪声 $\tau_0 = 0.0$** 起步：
1. **模型结构（`CCFUNet`，`base_dim=128`，~5.6M 参数）**：
   - **空间结构注入（SPADE + 多尺度拼接）**：输入拼接 `[z_tau (4ch), C_64 (4ch)]`，编码器/解码器在 $64\times 64, 32\times 32, 16\times 16$ 三级均通过 **SPADE（Spatially-Adaptive Modulation）** 注入 $C$ 与投影后的白化 DINO 语义图 $\tilde{P}_{16}$，并在 $16\times 16$ 瓶颈层配 4 头全局自注意力保证全图构图协调。
   - **五派严格 1:1:1:1:1 均衡 + 类内 $K=8$ 子模态原型条件化（`SubMode-AdaLN`）**：
     - 在干净训练池内（严格排除 Test750 源与参考）按每派 48 维小波统计提取 $K=8$ 个聚类子模态原型；
     - 训练时：50% 概率喂 `(style_id, submode_proto)`（学习鲜明子模态，消除类内平均灰暗化）、35% 喂 `(style_id, class_mean_proto)`（保证纯类别 ID 推理同样锐利）、15% 置为无风格条件 `(uncond)`（支持推理期 **Style-CFG**）；同时以 10% 概率独立随机遮挡结构条件支持 Dual-CFG，并对 $C_{64}$ 施加随机模糊与对比度抖动防过拟合。
   - **速度场 + 预测终点结构一致性损失**：
     $$\mathcal{L} = \|v_\theta - (z_{\text{real}} - \varepsilon)\|_2^2 + \lambda_{\text{struct}} \mathbb{I}(\tau \ge 0.25)\,\|\text{LCN}(Y_{\text{proj}}(\hat{z}_1)) - C_{64,0}\|_1, \quad \hat{z}_1 = z_\tau + (1-\tau)v_\theta$$
     其中 $Y_{\text{proj}}$ 为训练集线性拟合的 4→1 潜空间亮度投影，确保高 CFG 下几何边缘丝毫不漂移！
2. **实验臂设置（全部从 $\tau_0 = 0.0$ 纯噪声积分 20 Euler 步）**：
   - **`V18_B1_CCF_w2` / `V18_B2_CCF_w3.5`**：纯类别条件 + Style-CFG $w_s \in \{2.0, 3.5\}$；
   - **`V18_B3_CCF_submode_w3`**：训练集代表性子模态原型条件 + $w_s = 3.0$；
   - **`V18_B3_CCF_wrong`**：固定同噪声、同结构条件下的错标签控制（验证纯正的目标画派控制力与切换响应 $I$）；
   - **`V18_C1_CCF_LD4plus`**：在 `CCF` 生成的去源风格潜变量上叠加 2 步轻量掩码 `LD4+`，测试生成流与局部判别精修的协同上限。

---

## 3. 成功判据与硬性工程护栏

1. **突破判据（Cross600，S10 冻结裁判）**：
   - **Track A**：`V18_A1_LD4plus` 超越 `V9 LD4`（`Top-1 > 37.5%`，超越外部 Seedream 4.5 的 `37.3%`）；纯前馈单步 `V18_A2_FastStudent` 超越 `P1_B4_distill50`（`22.2%`）至少 **+8pp**（达 `30%+`）且 `correct > wrong` 显著。
   - **Track B**：`CCF` 从 $\tau_0=0.0$ 起步，**彻底打破 `ER = 0%` 与 `Ukiyo-e = 0%` 死锁（五画派每派 Cross Top-1 均显著破零，总 Top-1 大幅超越 V15–V17 的 `12.33%` 天花板）**，且 `pred == src` 源锁定率从 `59%` 降至 `< 25%`，`correct - wrong` 差值 > `+15pp`。
2. **工程铁律**：
   - 训练池严格按 NFC canonical stem 排除全部 `Test750` 源图（150）与 `references_v1` 参考图（150）；
   - 生成与 `src.eval.measure` 严格拆分为前后两个独立操作系统进程，杜绝 12GB 显存叠加 OOM；
   - 启动长训练前必须通过单元测试与全链路 2-step Canary。
