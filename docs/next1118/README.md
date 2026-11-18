# NEXT-1118 总纲：从 `PRISM v1` 迈向顶会级无死角单步法则（`PRISM v2 / V19`）

**撰写日期**：2026-11-18 · **当前基线**：[`docs/icme2027/paper.pdf`](../icme2027/paper.pdf)（`PRISM-Pro` Cross600 Top-1 `38.00%`；`PRISM-1` 纯单步 `28.5 ms/img` Top-1 `33.83%`，反事实因果特异性 `+32.50pp`）· **执行算力**：`RTX 4090 (24GB)` 全权实验线。

---

## 一、提纲挈领：我们站在哪里，为什么要发起这轮决定性推进？

在上一阶段（V18 + S10 + `PRISM v1`），我们已经确立了**“重建—传输拮抗（Reconstruction–Transport Antagonism）”**与**“源域风格锁定（Source-Style Lock）”**这一极具穿透力的理论框架，并在 750 任务统一裁判下首次正面超越了商业基座 `Seedream 4.5`（`38.00%` vs `37.30%`），同时实现了 `28.5 ms` 的纯单步高特异推理（`33.83%`，错标签跌至 `1.33%`）。

然而，站在 **CVPR / ICCV / ACM MM Oral 乃至 ICME Best Paper** 的严苛视角审视，当前体系仍存在 **4 处结构性破绽**，而对历史代码（`LatentStyleNet`、`solver.py`、`direction.py`、`canonicalize.py`）与负结果档案（`docs/negative_results/`、`V12`、`V14`）的交叉深挖表明：**这 4 处破绽并非偶然，而是底层潜空间判别器与各向同性信任域遗留的 4 个确定性数学缺陷**——一旦精准拆除，性能与理论优雅度将迎来第二次跃迁。

---

## 二、四大现存短板与源码级病灶一一对应表

| 现存短板（审稿人攻击面） | 源码级确定性病灶（本次复查定位） | `V19 / PRISM v2` 的根治算子 |
|---|---|---|
| **1. 少数派画风偏低、源锁仍超半数**<br>（`Ukiyo-e` 仅 `5.0%–9.2%`，`ER` 仅 `9.2%–10.8%`，`Src-Lock` `52.7%`） | ① `LatentStyleNet (R1)` 每层均含 **`GroupNorm(8, w)`** + 一阶 `avg_pool`，**在架构上直接抹除了全局色盘均值/方差与二阶 Gram 纹理协方差**；<br>② `direction.py` 中 `frac=0.35` 硬掩码将 **65% 画布像素置零不编辑**，整幅木刻平涂无法发生。 | **穿过可微 `S10`（`10.66M`）的无归一化多尺度像素域均衡判别教师**（严格隔离四裁判，仅在 4,400 训练画作上提取无 GroupNorm 的一阶色度/亮度矩 + 低秩二阶 Gram 特征训练均衡分类头）。 |
| **2. 信任域把“换色盘”误判为“毁内容”**<br>（稍放开 `eps` 内容即受损，不放开则色盘锁死在源画暗调） | `solver.py` 的 `cost_tot` 与 `project_rms` 使用 **4 通道各向同性欧氏球** $\|z - z_0\|_2 \le \rho$，而 4 维潜空间中 **3 维是色度（Chrominance），仅 1 维是亮度（$\hat{w}_Y$）**，且 $z_0$ 携带 `71.3%` 源色度泄漏！ | **潜空间亮度—色度正交分裂各向异性信任域（Anisotropic Luma-LCN Trust Region）**：在 $\hat{w}_Y$ 局部对比度纤维上严锁几何轮廓，在三维正交色度子空间 $\hat{w}_Y^\perp$ 上彻底放开色盘迁移！ |
| **3. 推理期暗含“已知源标签 $y_s$”假设**<br>（$e = u_{y_t} - u_{y_s}$ 无法直接处理无标签自然照片 `P2A` 或零样本新画派 `R5`） | `FastDINOStudent` 直接查离散表 `style_emb(tid) - style_emb(sid)`，未利用被 $\Pi_\perp$ 剥离掉的互补源风格信息。 | **正交商—纤维对偶 $(\Pi_\perp, \Pi_\parallel)$**：收集被 $\Pi_\perp$ 剥离的 RGB/Lab 色度矩与原始 DINO 空间均值/标准差构成 $\Pi_\parallel(x)$，学习连续风格坐标 $\hat{u}(x)=\phi(\Pi_\parallel(x))$，构造**免源标签通用弦** $e(x_s\to\tau) = u_\tau - \phi(\Pi_\parallel(x_s))$！ |
| **4. 单步学生与多步教师仍有 `4.2pp` 落差，且论文缺定性原图对比矩阵** | `FastDINOStudent`（`0.44M`）对整张图施加**空间均一标量 `AdaGN`**，未能按前景人脸/天空/背景语义角色差异化施墨；论文尚无多画风原图拼版。 | **潜色度预中和起点 $z_s^\perp$ + 语义路由空间弦调制网络（`Spatially-Routed Chord Adapter`，`~1.2M`，仍 `<30 ms/img` 单步）** + 补齐 `P2A/R5` 泛化与高清定性对比大图。 |

---

## 三、核心新方法浓缩公式（Unified `PRISM v2` Formulation）

### 1. 正交商—纤维直和分解对偶 $(\Pi_\perp, \Pi_\parallel)$
任意输入图像 $x \in [0,1]^{3\times H\times W}$（无论已知画派画作、未知流派画作、还是自然摄影图像）被一对正交互补算子无损拆解为**结构商坐标**与**风格纤维坐标**：
$$\Pi_\perp(x) = \big(C_{64}(Y(x)),\; \tilde{P}_{16}(Y(x))\big), \qquad \Pi_\parallel(x) = \big(\mathcal{M}_{\mathrm{Lab}}(x),\; \mu_{\mathrm{sp}}(P_{\mathrm{RGB}}(x)),\; \sigma_{\mathrm{sp}}(P_{\mathrm{RGB}}(x))\big).$$
- $\Pi_\perp(x)$ 严格满足色度核不变性 $\nabla_{\ker(w_Y^\top)}\Pi_\perp \equiv 0$ 与零空间一阶矩 $\mu_{\mathrm{sp}}(\tilde{P}_{16}) \equiv \mathbf{0}$（5 分类风格探针 `18.3%`，低于随机 `20%`）；
- $\Pi_\parallel(x)$ 恰由被 $\Pi_\perp$ 投射抹除的色度分位数矩与视觉特征一阶/二阶空间统计组成，经轻量 MLP $\phi$ 映射至连续风格流形 $\hat{u}(x) = \phi(\Pi_\parallel(x)) \in \R^d$。
- **通用无源标签弦向量（Universal Chord）**：
  $$e(x_s \to \tau) = u_\tau - \phi(\Pi_\parallel(x_s)), \quad \text{其中 } u_\tau = \begin{cases} \bar{u}_{y_t}, & \text{注册画派模式（无需源标签 } y_s\text{）} \\ \phi(\Pi_\parallel(x_{\mathrm{ref}})), & \text{任意单参考图零样本模式（} x_{\mathrm{ref}}=x_s \Rightarrow e\equiv\mathbf{0}\text{）} \end{cases}$$

### 2. 潜空间亮度—色度正交分裂与各向异性 `S10` 可微教师（`V19-Teacher`）
设 $\hat{w}_Y \in \R^4$（$\|\hat{w}_Y\|_2=1$）为训练集最小二乘标定的潜空间单位亮度轴，定义一维亮度投影 $P_Y = \hat{w}_Y \hat{w}_Y^\top$ 与三维正交色度投影 $P_C = I_4 - \hat{w}_Y \hat{w}_Y^\top$：
1. **潜色度预中和起点**：将源潜变量低频色度直接对齐至目标画派先验均值/标准差，彻底拔除 $z_0$ 中 `71.3%` 的源色盘残留；
2. **穿过冻结 `S10`（`10.66M`）的像素域多尺度双向判别目标**（仅用 ImageNet 预训练 `VGG-16` / `ConvNeXt-Tiny` 在 4,400 训练画作上训练均衡无 GroupNorm 多尺度矩+低秩 Gram 判别头 $F_\psi$，**100% 物理隔离四大评测裁判**）：
   $$\max_{z} \; \tilde{m}_{F_\psi}(\mathcal{D}_\star(z), y_t) - \gamma\,\tilde{m}_{F_\psi}(\mathcal{D}_\star(z), \hat{y}_s) - \lambda_{\mathrm{mom}}\,\mathcal{D}_{\mathrm{style}}\big(\mathcal{D}_\star(z), \mathcal{B}_{y_t}^{\mathrm{train}}\big)$$
   受限于**各向异性结构信任域**：
   $$\big\|\mathrm{LCN}_5(\hat{w}_Y^\top z) - \mathrm{LCN}_5(\hat{w}_Y^\top z_s)\big\|_2 \le \rho_Y \quad (\text{严锁几何边缘}), \qquad \|P_C(z - z_{\mathrm{init}})\|_{\mathrm{RMS}} \le \rho_C \;\; (\rho_C \approx 3.5\rho_Y,\; \text{放开色盘}).$$
   *（注：使 `S10` 参与离线可微轨迹生成，直接把论文中的 **Operator III (`PRISM-Dec`)** 从“末端加速插件”升格为“解锁像素级快速轨迹蒸馏的核心赋能算子”，三算子逻辑彻底闭环！）*

### 3. 语义路由空间弦单步学生（`PRISM-1 v2`，`~1.2M` 参数，`<30 ms/img` 纯单步）
利用 $\tilde{P}_{16}(u)$ 的局部语义角色（人物面部 / 天空 / 建筑 / 背景）与通用弦向量 $e(x_s\to\tau)$ 的双线性门控生成**空间异质调制场** $G(u), B(u) \in \R^{C\times 64\times 64}$，实现“前景主体精保结构、背景与色域大胆施墨”的纯单步前馈推理，且对同风格/自身参考严格保持恒等。

---

## 四、文档导航与预注册验收红线

| 编号 | 文档路径 | 内容概要 |
|---|---|---|
| **01** | [`01_CRITICAL_DIAGNOSIS_AND_HISTORICAL_AUDIT.md`](01_CRITICAL_DIAGNOSIS_AND_HISTORICAL_AUDIT.md) | 当前稿件 4 大弱点拆解、4 处源码病灶行级取证、与 `Negative_Results`（L1–L4）/ `V12` / `V14` / `04_bugs` 的逐条防踩坑核对表 |
| **02** | [`02_MATHEMATICAL_THEORY_AND_ARCHITECTURE.md`](02_MATHEMATICAL_THEORY_AND_ARCHITECTURE.md) | $(\Pi_\perp, \Pi_\parallel)$ 正交对偶定理、$\R^4 = \operatorname{span}(\hat{w}_Y)\oplus\hat{w}_Y^\perp$ 各向异性投影公式、`S10` 可微多尺度判别器与语义路由单步网络精确张量规格 |
| **03** | [`03_4090_EXECUTION_AND_PAPER_UPGRADE_PLAN.md`](03_4090_EXECUTION_AND_PAPER_UPGRADE_PLAN.md) | `RTX 4090` 分阶段预注册实验表（Gate 审计 $\to$ Teacher 消融 $\to$ 免 `sid` 单步学生 $\to$ `P2A/R5` 零样本 $\to$ 定性大图拼版与论文回填） |
| **04** | [`04_V20_SUBMODE_SLF_THEORY.md`](04_V20_SUBMODE_SLF_THEORY.md) | `V20 / PRISM-Flow` 突破超球面均值坍缩与 $64\times 64$ 笔触边缘枷锁（场景匹配子流派 `Patch-SWD` + Coarse-16 `SLFUNet` 超越 `StyleGallery`） |
| **V19** | [`V19_RESULTS.md`](V19_RESULTS.md) | `V19 / PRISM v2` 14 臂全量 750 任务实测结果报告 |
| **V20** | [**`V20_RESULTS.md`**](V20_RESULTS.md) | **`V20 / PRISM-Flow` 25 臂全量 750 任务实测收官报告（`DINO-S = 0.5789`, `Affinity = 0.6683`, `Top-1 = 64.00%–64.50%`，全面超越 `StyleGallery`）** |

### 预注册硬性突破目标（Cross600 / S10 四裁判协议）
1. **教师端（`PRISM-Pro v2`）**：Cross600 CSD Top-1 突破 **`42.0%–48.0%+`**（无条件正面压倒 `StyTr2` `39.50%` 与 `Seedream 4.5` `37.30%`），`DINO-C >= 0.72`，`LPIPS <= 0.50`，`Src-Lock < 40%`，且 `ER` 与 `Ukiyo-e` 双双突破 **`20%–30%+`**。
2. **免源标签纯单步端（`PRISM-1 v2`，`<32 ms/img`）**：**不输入测试集真实 `sid`**，仅凭 $(x_s, y_t)$ 单步前馈达到 **`38.0%–42.0%+` CSD Top-1**（单步直接超越旧多步 `38.0%` 与 `Seedream 4.5` `37.3%`！），反事实错标签控制 `< 3.0%`（因果特异性 **`> +35pp`**）。
3. **跨域零样本与视觉闭环**：在 `P2A`（自然照片转艺术）与 `R5`（5 个未见 WikiArt 画派）上验证连续弦向量 $e(x_s\to x_{\mathrm{ref}})$ 超越 `IDT` 锚点并保持高内容保真；生成全套高清原图对比大图并更新入 [`docs/icme2027/paper.pdf`](../icme2027/paper.pdf)。
