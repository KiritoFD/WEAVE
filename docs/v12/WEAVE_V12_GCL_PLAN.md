# WEAVE V12-GCL：Content-Query Geometry / Global Content Loss 专项

**Status:** 实验预注册方案（未实施，未测时）；**Date:** 2026-11-14；**Hardware:** 4070 Laptop，可用显存约 6 GiB；**协议:** 全部正式臂统一 `eval_protocol/tasks_750.json`：Cross600 + Same150，禁用 Dev200 / Val200 / Pilot80 作为效果筛选。

## 0. 核心决策

- **首要目标**：在相同内容保持约束下扩大真实风格 Pareto 前沿。GCL 仅让 DINO-C 上升、风格下降，不算成功。
- **冻结** B4/N2C、LD4 风格方向、S10 解码器和现有评测器；优先只换内容几何与 step acceptance。
- **不与 3060 V11-A 混线**：V12 不训练风格编码器，不依赖局部 KDE/Score，也不使用人工偏好或 GAN。
- **两条成本轨道**：G1/G2/G3 小模型或零训练参数；G4/G5/G6 允许额外前向、反向、JVP/VJP 或预训练模型。

## 1. 来源与当前代码边界

- StyleGallery (CVPR 2026): `L_GCL = ||Q_gen - Q_content||_1`，`L_RSL` 为区域 Attention 对齐；官方 GCL/RSL 删除消融均导致结果恶化。它的 Query 来自**预训练 SD1.5 UNet**，不是任意卷积层。
- `KiritoFD/WEAVE/tools/v6/models_v6.py`：`C3UNet` 原始 B4 多尺度 CNN（64/32/16），Style-ID AdaLN/FiLM，32×32 Haar 及 64×64 空间输出；**没有可复用的 self-attention Q**。
- `src/model/canon.py` / `tools/v6/canonicalize.py`：冻结 `c1_gentle` content canonicalizer；注意 canonicalized latent 不是理论上的 style-invariant content，必须实测。
- `b1_style_operator.py` / `output_supervision.py`：已有 latent self-sim、edge、DINO patch self-sim，不能把 G2 当成全新原理；本轮要测试它们是否真正给 LD4 增加价值。
- `src/infer/run.py`：750 统一推理输出 `z_src`/`z0`/`z_final`，允许缓存 source Query 和中间 latent。
- `docs/v6/S10_READY_TO_USE.md`：旧 S10 必须保持原 checkpoint 内 **恰好一次 PQC** 和正确 latent 缩放契约；在本轮不修改 S10。
- 新近上传的 V10.2 结果：同协议 LD4 的 Cross600：DINO-S .4401、DINO-C .7718、LPIPS .4872、Top-1 .3517、Affinity .5615；E4 为 .4415 DINO-S / .5616 Affinity（口径须保持一致）。

> 注：连接的 GitHub 默认分支不一定包括 2026-11-14 本地 LD4/V10.2 最新代码。集成时必须在实际工作机检查 `git rev-parse HEAD`、LD4/S10 调用接口与任务缓存；这里不虚构不存在的函数和命令。

## 2. 统一数学定义与陷阱

设 `z_s` 为**源图真实 latent**，`z_0` 为 LD4 当前优化的初始结果，`z` 为候选编辑结果，`g_s` 为原有风格编辑方向，`Q_φ(z)` 为冻结的、对结构敏感而尽量对风格不敏感的内容特征。

`E_Q(z) = D(Q_φ(z), stopgrad(Q_φ(z_s)))`。

约束优于一味加权：

\[
\max_\delta g_s^\top\delta\quad \text{s.t.}\quad
E_Q(z_0+\delta)\leq B_Q,\quad
C_{\rm existing}(z_0+\delta)\leq B_C,\quad
\|\delta\|_{\rm RMS}\leq\rho.
\]

- Query 必须从**实际编辑输出** `z` 提取。如果两支都从未变化的 `canon(z_s)` 提取，损失恒零。
- Query 对**源图**锚定是内容保留的原始定义；`Q(z_0)` 锚定可以另做保守对照，但可能锁定 B4 自己的几何错误。
- 同步缓存 `Q(z_s)` 并停止其梯度，`Q(z)` 保留对 `z` 的梯度。
- 风格不变性不能假设成立：先检查同内容外观扰动 vs 已知几何破坏的判别能力。
- **零梯度陷阱**：若 `Q(z_0)=Q(z_s)`，则 `∇_z 1/2||Q(z)-Q(z_s)||²|_{z_0}=0`，内容梯度投影对第一步无约束。必须使用**更新后的闸门**或 `J_Q^T J_Q` 的前视约束。
- 对 L1 而言零点取子梯度通常为零，问题类似。
- 现有 LD4 预算不能直接由 GCL 替代；先叠加/对照，再判断 GCL 是否能允许**增加**风格编辑强度。

## 3. 候选路线与成本分层

| Arm | 方法 | 新增可训练参数 | 每个候选所需主要额外工作 | 验证的机制 |
|---|---|---:|---|---|
| G0 | 原 LD4 + 原内容闸门（重用缓存） | 0 | 无 | 固定基线 |
| G1 | Haar/edge Query-GCL | 0 | 少量 DWT/梯度算子 | 局部几何 Query 是否比单一内容预算更灵敏 |
| G2 | 16² 全局关系 + 32² 局部关系 GCL | 0 | self-sim/局部关联矩阵 | 保护关系而非特征值，是否释放笔触变化 |
| G2-old | 直接复用 V5 self-sim 的独立对照 | 0 | 现成函数 | 新损失是否真正优于旧内容关系约束 |
| G2N | **冻结 C3UNet 原生 Encoder Query** | 0 | 每次候选多次 B4 Encoder 前向/反向 | 已训练内容特征是否比 Haar 更稳健 |
| G3 | Tiny Style-Invariant Query Encoder（CQE） | 0.2–0.6M | 双分支轻量 Q 前向及梯度 | 学习外观不变、几何敏感的 Query |
| G4 | S10 解码 + 冻结视觉教师的语义关系 GCL | **0 个新训练参数**，但加载教师权重 | S10 与视觉教师全链路反向 | 外部强视觉表征的上限 |
| G5 | Jacobian metric / CG 约束更新 | 0 | 每步 4–8 次 JVP/VJP + 线搜索 | 主动寻找 Query 的近似零敏感方向 |
| G6 | SD1.5 多时间步 Self-Attention Query 教师 | **0 个新训练参数**，大量冻结权重 | SD 编码与 UNet 前向/反向、可选 CPU offload | StyleGallery 原始 GCL 的跨骨干参考上限 |

**G4/G6 不属于轻量部署方式**；若有效，仅用于后续训练期间的蒸馏教师或离线 Oracle。不得把教师运行内存漏报为零。

### G1：Haar Query，零参数

- 输入现有 `4×64×64` latent，使用冻结的 content canonicalizer（与 raw-latent 版本作消融）。
- 以 `LL_{32}`、16×16 池化、边缘梯度幅值/方向为 Query；必要时采用相对/局部规范化，避免把颜色和光照直接当作结构。
- 候选 `Q` 比较采用 SmoothL1，保留 coarse/fine 分项及梯度范数；不将高频笔触能量全部锁死。
- 区分 `z_s` 锚点与 `z_0` 锚点；预期前者较能保障实际原内容，后者可能抑制改动更少。
- 核心失败模式：Query 与 latent 低频 L2 等价，新增约束无法产生额外 Pareto 增益。

### G2：关系 GCL，零参数，首选

设 `q_i` 是从 G1 的风格弱敏感特征经通道归一化取得的 token。

\[
A_{ij}(z)=\frac{\exp(\langle q_i,q_j\rangle/\tau)}{\sum_k\exp(\langle q_i,q_k\rangle/\tau)},
\quad
L_{\rm rel}=\frac1N\sum_i \mathrm{KL}(A_i(z_s)\parallel A_i(z)).
\]

- 全局尺度用 16×16：256²、每样本单 float32 相似矩阵约 **0.25 MiB**；32×32 完整矩阵为 1024²，约 **4 MiB**，建议只做 5×5/7×7 窗口或源图 top-k 图。
- Query 特征只用低频/结构量；直接 raw latent Gram 可能过度约束色彩与笔触。
- 输出的 row softmax `τ` 用训练图源内容的几何扰动分布校准，不依赖 Test750。
- `G2-old` 复用 `b1_style_operator.py` 的旧 `patch_self_similarity`，保证新特征/局部图真正带来贡献。
- 风险：Minimalism 等极低纹理内容无法提供有辨识力的关系图，应分别记录图熵与有效边缘数，不强迫所有画派得到同一收益。

### G2N：C3UNet 原生内容 Query，零新增权重，中等计算

- 对冻结的 B4/C3UNet 增加**非侵入式特征 Hook**，从 `enc32_2`、`mid16_2` 等 **Encoder** 中间层提取 `H_l(z)`，把实际 `z_out` / `z_src` 分别作为用于该审计分支的内容输入；保持两支完全相同的 `tau` 和 `style_id`。不把固定 `c(z_src)` 在输出分支复用，否则 GCL 可能与 `z_out` 无关。
- 由于 B4 的 AdaLN 含有 Style-ID，`H_l` 不天然风格不变：先以固定 `style_id` 的 5 种取值和训练集外观扰动做特异性审计，必要时使用 Style-ID 结果均值或去风格规范化作为 `Q_native`。这会增加前向次数，**不是零计算成本**。
- 对 `H_l` 直接 L1 与局部关系损失分别做消融；只有通过“对画派变化不敏感、对结构破坏敏感”的 Query Gate 才进入完整 Test750。
- 该臂的价值是检验**现有 WEAVE 网络到底是否已经具有可利用的内容 Query**，在训练 G3 之前不要直接假定必须训练另一个 Encoder。

### G3：学习轻量 Content Query，0.2–0.6M 参数

冻结真实训练画作的 latent，构建**自动监督**：

- 正样本：同一幅图的颜色/Haar 幅值变化、亮度、弱风格扰动，要求结构保持；如使用 WEAVE 同源多风格输出，先用现成内容筛选去掉结构严重损坏的伪正样本。
- 已知几何变换：合成 affine/局部 warp，要求 Query **等变**而不是不敏感；利用已知变换对应位置对齐。
- 内容破坏负样本：局部位移、擦除、随机交换区域，要求 Query 关系明显变化。
- 加入 VICReg/方差/协方差约束，防止所有 Query 塌缩成常量；必要时引入 5-way 风格探针/对抗去风格，但不能因此删除必要的内容信息。
- 空间输出 `Q_θ(z)`：16×16、32×32，固定 `θ` 后用于 LD4 GCL；**不引入新的风格目标**。
- 可选择用离线 DINOv2 patch 关系蒸馏（教师只在训练出现），运行时只保留 tiny CQE。
- Gate：在留出画作的**自动内容扰动审计**中，对几何破坏的敏感度 / 对外观变化的敏感度明显优于 G1；且 Query 有非塌缩空间方差。

### G4：Decoded Feature-GCL，高计算、零新训练参数

`x(z)=S10(z)`，用冻结的 DINOv2/VGG 的中间 Patch 特征 `F(x)` 构造**局部关系**，不直接优化全局 CLS 或与评测使用的同一 DINO-C 目标。

\[
L_{\rm decoded}=\sum_l w_l\,D\bigl(\operatorname{Rel}(F_l(S10(z))),\operatorname{Rel}(F_l(S10(z_s)))\bigr).
\]

- **必须可微：** `z → S10 → Teacher` 的输出支完整保留梯度；只冻结权重。源图分支缓存。
- 各阶段单独装卸模型，microbatch=1，bf16/fp16、activation checkpoint；**6 GiB 能否跑通必须 P0 实测**。
- 论文评价 DINO-C 不得与该教师的完全相同特征构成唯一内容验收，防止优化指标作弊；增加 LPIPS、边缘/几何、independent content features。
- 不应以 RGB 逐像素保持为目标，否则会阻止风格变化。
- 这条是**质量上限/蒸馏教师**，不是 0 开销的部署选项。

### G5：Implicit-Jacobian GCL，零参数，高算力

使用 G2 或 G3 的 Query，把新增风格方向 `g` 限制到 Query 近似不变化的切空间。

记 `e = Q(z_0)-Q(z_s)` 和 `J = ∂Q/∂z|_{z_0}`，线性化 `Q(z_0+δ)≈Q(z_0)+Jδ`。求解：

\[
\delta^* = \arg\max_{\delta}\left[g^T\delta-\frac\mu2\|\delta\|^2-\frac\lambda2\|e+J\delta\|^2\right],
\quad
(\mu I+\lambda J^TJ)\delta=g-\lambda J^Te.
\]

- 不显式构造 `J`，只用 JVP/VJP 求矩阵—向量积；PCG 4/8 次，停止阈值和误差日志齐全。
- 其优势在于：即使 `e=0` 导致普通平方 GCL 的一阶梯度为零，`J^TJ` 仍可限制第一步会破坏结构的方向。
- **单独测试**：G2 的普通 loss penalty vs G2+post-step gate vs G2+Jacobian solver；否则无法说明更高计算的必要性。
- 高阶局部近似可能失效，最终必须使用原有 LD4 实际回溯与多指标闸门；遇到 CG 发散、NaN、奇异性时保守回退。

### G6：真正的 SD1.5 Query 教师，极高计算

- 仅作为与 StyleGallery 最接近的上限实验。`x=S10(z)`，再由兼容 SD1.5 encoder 得到扩散 latent；在**相同固定噪声和相同时间步**上提取 `Q_gen` / `Q_src`，比较选定 2–6 个 self-attention 层。
- 不能不经过图像/编码对齐就假定 WEAVE 4×64×64 latent 与 UNet 的噪声时间分布兼容。
- 标准 SD1.5 UNet 需要大量权重和反向激活；在 6 GiB 上必须先单样本测峰值，必要时采用 CPU/offload 或迁移至 4090。如果无法运行，不应伪造替代数值。
- 可缩至单个/少数时间步，仍与原作者 150 步采样不同，不应声称精确复现 StyleGallery。
- 若教师有清晰 Pareto 增益，再将其 Query 的空间关系蒸馏给 G3，而**不是长期保留 SD1.5 推理**。

## 4. 调度与约束：三种必要对照

对每种 `Q`（G1/G2/G3/G4）至少比较以下两种内容用法；G2 对后续 G5 提供 Query：

1. **Penalty**：`L=L_style + λ L_Q`；记录风格梯度、内容梯度的夹角/范数。
2. **Gate**：根据真实编辑 `z_cand` 的 `L_Q(z_cand)` 是否超出内容预算接受或回溯；不依赖起点一阶梯度。
3. **Metric（G5）**：在 Jacobian 内容几何中求带权风格步长，再做 Gate。

`B_Q` 只通过训练集源图上的外观不变性/几何扰动统计预注册。风格编辑预算用 `LD4-base` 与统一 `LD4-strong` 两档，避免只因 GCL 更严格就获得虚假的“内容提升”；同预算与加预算都要配套同一无 GCL 对照。

**强制记录：** 每张样本的基准 Q 距离、候选 Q 距离、接受率、平均回溯次数、`cos(g_style, grad L_Q)`、`||J δ||`、最终内容指标，以及前向/反向/解码/teacher 的真实墙钟与显存峰值。

## 5. 正式实验臂与 750 协议

预注册主表推荐 **8 条独立全量 750**（每条 600 cross + 150 same）：

- G0：LD4 base（可从可信缓存复用）。
- G0S：LD4 strong（同 G4/G5 风格强度预算对照）。
- G1：Haar Query + post-step Gate，base。
- G2O：旧 V5 selfsim + Gate，base。
- G2：新 MultiScale Relation-GCL + Gate，base。
- G2S：G2 + strong 风格预算。
- G3：Tiny-CQE Query + Gate，base（仅当表示 P0 Gate 通过）；G2N 是零新权重候补，若 P0 表征 Gate 通过则以完整 750 替换 G3 或安排独立补测。
- G5：G2 Query + implicit Jacobian solve + Gate，strong（仅当时间/内存 P0 Gate 通过）。

**高算力扩展（独立队列，仍然每臂完整 750）**：G4 decoded teacher + Gate / G6 SD UNet teacher。其 GPU 成本与主队列可相差数量级，**不预先承诺 12 小时内能够跑完**。

用户约束：**不使用 Dev200、Val200、Pilot80 或任何子集效果分数来筛选模型/超参数。** 允许少量样本只做 smoke、显存、梯度连通/有限差分、latency probe，不做效果排名。完整效果一律 750，记录所有预注册臂与所有失败。

### Cross600 指标与判据

- 主要纵轴：同一 `paper_rgb_v1` 下 Style Affinity（沿用当前 Pareto 图）；同时报 DINO-S、CSD Target Top1/margin。
- 主要横轴：DINO-C；同时报 LPIPS、边缘/内容结构、Same150 输入保持。
- 成功 = **非支配前沿有配对统计支撑的扩张**：举例，在 `|ΔDINO-C|≤.003` 时 Style Affinity 提升 ≥.003，或者在 `|ΔAffinity|≤.002` 时 DINO-C 提升 ≥.008；这些是实验 Gate，而不是对结果的预测。
- 不只选择全量均值上最好看的点：按照源图 ID 作 cluster-bootstrap（150 源图 × 5 目标），同时报告 5 个目标画派的增益/退化，特别检查 Minimalism。
- Same150 要检查平均编辑强度、DINO-C/LPIPS 退化和输出变化；允许 same-style 中按需轻微修复，但不能系统性过度风格化。
- 不允许单纯优化评测所用 DINO 特征得到表面指标优势；G4 使用 DINO Teacher 时必须同时通过非同源结构约束与图像审计。
- 坚持单 GPU、相同任务和同配置真实时间：分别报告 `cached edit time` 与 `full pipeline time`，包括教师初始化/权重加载/CPU offload；不能把不同 GPU/缓存口径拼接为单一 FPS。

## 6. Gate、资源与 12h 阶段安排

首先运行 **P0 纯正确性/资源审计**（不是效果筛选）：

- 固定 SD0/S10 latent 输入/缩放/PQC，验证 source 等价与梯度可达。
- `Q(z)`, `Q(z_s)` identical 时损失应为零、原型判断不得错。
- 风格扰动 vs 已知几何破坏的 Query 响应关系、feature collapse、数值/NaN 检查。
- 检验 `JVP` 与有限差分一致；CG 残差收敛；LineSearch 只接受预算内样本。
- 实测单样本峰值显存、速度；确保主流程在约 6 GiB 可用显存下跑通。

**主线 12 GPU-hours 的建议预算（动态，以真实基线速度复核）**：

| 阶段 | 预留 | 内容 |
|---|---:|---|
| P0 | 1h | contract、单测、显存、时间测量（不筛效果） |
| P1 | 1.5h | G1/G2 计算与局部扰动 Query 识别审计 |
| P2 | 4.5h | G0/G0S/G1/G2O/G2/G2S 的全 750 推理+评测，缓存现成结果 |
| P3 | 2h | G3 CQE 自动训练与表征 Gate；通过后全 750 |
| P4 | 2h | G5 CG 4/8 迭代选择预注册配置及全 750 |
| P5 | 1h | 配对统计、Pareto、延迟、Same150、结果报告 |

该表是**资源分配而非预测耗时**。若 P0 测得所有完整 750 臂无法放入 12h，则在预注册优先级下减少**完整臂数量**或延长预算，不能将一个臂改为 200/80 子集并报告为效果。G4/G6 作为独立高计算扩展，可能需要额外 12–24h 或更大 GPU，必须根据实测确认。

## 7. 失败机理与决策树

- **G1/G2 没有明显收益**：当前 LD4 内容保护可能已经足够，新的 Query 只是重复旧的低频/自相似度损失。若风格仍受限，回到风格表示而不是继续堆内容正则。
- **G2 相对旧 selfsim 无收益**：Relation-GCL 不具有独立方法贡献，应直接复用旧实现。
- **G3 只提升内容**：Learned Query 可能学成过度保守/风格敏感，核查扰动不变性与风格探针。
- **G4/G6 有效、G1–G3 无效**：预训练教师的 Query 信息确实重要，优先小模型 distillation，证明不是“注意力结构本身”奏效。
- **G5 比 G2 Gate 明显更好**：表明内容敏感编辑几何比标量内容损失更有效，值得后续用学习式近似降低推理成本。
- **所有 GCL 只能增加 DINO-C，无法在固定内容下增加 Style Affinity**：本阶段证实内容约束不是眼前主要瓶颈，保持 LD4/E4，继续 3060 风格表征与 4070 笔触生成支线。

## 8. 工程结构（待创建，勿当成现有可执行脚本）

```
exp/v12_gcl/
  audit_contract.py
  query_haar.py
  query_relation.py
  train_content_query.py
  query_teacher.py
  jacobian_solver.py
  constraints.py
  run_750.py
  aggregate.py
  tests/test_q_identity.py
  tests/test_jvp_finite_difference.py
  tests/test_s10_grad.py
  tests/test_budget_gate.py
runs/v12_gcl/{profiles,train,query_cache,all750,metrics,logs}/
docs/v12/V12_GCL_RESULTS.md
```

**工程接口建议**：`Q_phi(z) -> {q16,q32}`；`constraint(z_candidate,z_source,cache) -> per_sample_cost`；`project_or_gate(z0,g_style,...)->z_final,trace`。独立于风格梯度模块和模型主干，便于与 LD4、E4、后续 V11 交叉复用。

## 最终选择

优先跑 **G0/G0S、G1、G2O、G2/G2S**，再审计已有 B4 Encoder Query（G2N），根据表征/资源 Gate 决定 G3 与 G5。G4 和 G6 是质量上限实验而非轻量化部署方案。**真正可能形成方法贡献的是 G5：利用内容 Query 的 Jacobian 几何主动寻找风格可编辑方向，而非单纯在 Loss 上添加新的权重。**
