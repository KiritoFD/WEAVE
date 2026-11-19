# V25-P：WEAVE 多 Agent 并行科研、诊断与工程收敛计划

**定位：** 在现有 V25 12 小时 GPU 计划之外，增加独立的 CPU 科学诊断、测量、工程重构、统计审计和论文工作流。

**执行基础：**

- 当前 V25 Round 1 已完成，Round 2/3 正在执行。
- 4090：Linux，`tmux`。
- 3060：Windows，PowerShell＋`schtasks`。
- 全部正式 WEAVE 评测保持 test750（Cross600＋Same150）。
- 最终交付：简洁、可复现、可测量、具有明确科学结论的 `PRISMCore` 与 `PRISMRef`。

## 1. Agent 组织结构

采用**多 Agent 分工＋确定性实验队列**，而不是让所有 Agent 自由运行命令。

```text
V25 ORCHESTRATOR（总控，唯一集成人）
│
├── GPU-4090（唯一 4090 执行负责人）
│   └── DEV-CORE（主线代码开发，无 GPU 权限）
│
├── GPU-3060（唯一 3060 执行负责人）
│   └── DEV-REF（One-Shot 代码开发，无 GPU 权限）
│
├── STAT（5 个 CPU 统计 Agent）
├── DATA（5 个 CPU 数据与特征诊断 Agent）
├── MEASURE（3 个 CPU 测量 Agent）
├── ENGINEERING（5 个代码工程 Agent）
├── DOCS（2 个文档 Agent）
│
├── REVIEW-SCIENCE（独立科学审查）
└── REVIEW-ENGINEERING（独立工程审查）
```

共 1 个总控、2 个 GPU 执行、2 个 GPU 开发、20 个 CPU 专项任务、2 个审核角色。

**不要求 27 个 CPU 进程同时运行。**

子 Agent 的代码阅读、数学分析和文档工作可以大规模并行；真正进行 CPU 计算的任务限制并发，避免影响 GPU 训练的数据加载与 I/O。

### 1.1 权限分层

| 角色 | GPU | 修改实验脚本 | 修改公共代码 | 提交 main |
|---|---|---|---|---|
| 总控 | 禁止 | 仅集成时 | 是 | 唯一负责人 |
| GPU-4090 | 独占 4090 | 自有脚本 | 禁止 | 禁止 |
| GPU-3060 | 独占 3060 | 自有脚本 | 禁止 | 禁止 |
| DEV-CORE / REF | 禁止 | 自有新模块 | 独立分支 | 禁止 |
| CPU 专项 Agent | 禁止 | 限定文件 | 限定文件 | 禁止 |
| 审核 Agent | 禁止 | 只读 | 只读 | 禁止 |

需要修改共享接口时，先提出变更契约，由总控合并，不能自行覆盖 GPU 运行中的代码。

### 1.2 三条工程铁律

**第一，GPU 不竞争。**

每张卡始终只有一个执行 Agent。其他 Agent 不得启动 CUDA、加载模型到显卡、修改设备环境或重启运行任务。

**第二，文件不竞争。**

每个 Agent 使用独立 Git worktree 和独立分支。共享文件只允许总控修改或合并。

**第三，结果不竞争。**

原始实验结果、checkpoint、配置与指标一经登记不得静默覆盖。新结果必须采用独立 run ID。

---

# 2. GPU 专属 Agent：继续原定科研实验

## G1：GPU-4090 —— PRISM-Core 负责人

只管理 4090 GPU 队列。

### G1-0：完成正在执行的 V25 Round 2

- 接管并检查 `Tr_T0–T3`。
- 完成双分支统一评测。
- 汇报不同训练配方的内容、风格、失真和时间。
- 将原始 JSON 交给 CPU 统计 Agent。
- 不直接根据点估计删除组件。

### G1-1：冻结 Minimal PRISM

以 Round 1/2 结果确定：

- 最终 SLFUNet checkpoint；
- `cs_j` 或 `full`；
- `fixed` 或原始 Gate；
- T4 或 PRISM-1 锚点；
- 连续强度参数的有效范围。

特别注意：

**不能凭 D6 在 DINO-C=0.73 处的外推结论删除 Hybrid。**

目前 D6 的实测高内容点只到约 0.70。Hybrid 在 0.73–0.75 区间仍是合法的优秀候选。

### G1-2：机制 A——空间自适应混合

按原计划执行 A1/A2/A3，但增加两个正确性控制。

**控制一：相同教师。**

训练和评测必须使用相同定义的：

`PRISM-1 + D6 checkpoint + anchor + CFG + strength`

禁止训练使用 PRISM-1 锚点的 D6，评测却与 T4 锚点的 D6 直接比较后归因于空间门控。

**控制二：相同总体编辑强度。**

空间门控必须保证全局平均编辑强度近似不变，否则可能只是等效提高了 γ。

除原始 γ 外，增加一个固定编辑能量对照：

\[
E=\|\gamma M\odot(z_{D6}-z_{P1})\|_2.
\]

比较相近的 E，而不是只比较相同的名义 γ。

### G1-3：机制 B——单模型多强度蒸馏

保留 B0/B1，并明确增加：

- `γ=0` 精确恢复冻结 PRISM-1；
- 中间 γ 插值误差；
- 未见 γ（如 0.33、0.63）泛化；
- 不同风格的蒸馏误差；
- 失败后的错误类型归因。

学生建议使用：

\[
z_{\mathrm{student}}
=
z_{\mathrm{P1}}
+
\gamma R_\theta(z_c,s,\gamma).
\]

这样在 γ=0 时天然恢复 PRISM-1。

如果需要训练 PRISM-1 主体，应分成独立实验，不能同时宣称具有精确零强度回退。

### G1-4：最终候选验证

- 最佳模型至少两个独立训练种子；
- 完整 test750；
- 五类及 20 条定向迁移分解；
- 完整 RGB→RGB 性能测量；
- 输出视觉拼图；
- checkpoint 与配置 hash 入库。

**G1 输出：** `docs/v25/GPU4090_RESULTS.md`。

---

## G2：GPU-3060 —— PRISM-Ref 负责人

只管理 3060 GPU 队列。

### G2-0：完成正在执行的 Round 3

- `O3_self_only` 训练与评测；
- `O3_label_free` SG-25 OOD 与参考更换测试；
- τ 曲线归档。

### G2-1：机制 C1——教师监督强度校准

保留 `w_c=2.2/1.2/0.7` 三档。

但不能预先规定 `mid_cos=0.80–0.86` 就一定最好。

同时观察：

- 内容保持；
- 实际风格增量；
- LPIPS；
- 参考图影响程度；
- 局部伪影。

教师筛选从“仅按内容分数取 Top-K”改为**内容与风格双指标筛选**，避免所有保留下来的伪目标都近似原图。

### G2-2：机制 C2——参考替换一致性

修正损失：

\[
\mathcal L_{\mathrm{swap}}
=
\mathcal L_{\mathrm{style}}(G_a,r_a)
+
\mathcal L_{\mathrm{style}}(G_b,r_b)
+
\lambda_c[
\mathcal L_c(G_a,x_c)
+
\mathcal L_c(G_b,x_c)
].
\]

如果使用余弦相似度，风格项应采用 `1-cos`，不能直接最小化正余弦相似度。

增加必要对照：

- 正确参考；
- 错误参考；
- 同风格不同参考；
- 完全不同风格参考。

证明模型真正读取参考图，而不是只学会改变颜色或增加输出扰动。

### G2-3：SG-25 OOD 评测

继续使用固定 125-pair 清单作阶段诊断。

另外增加参考覆盖：

- 每风格至少多张不同参考；
- 同一内容对应多个参考；
- 相同参考对应多个内容；
- 单独统计水墨、抽象、线稿等困难类别。

不要将目前的 125-pair 结果称为 StyleGallery 论文原始完整协议。

### G2-4：独立种子与端到端测量

至少复验最佳候选，并明确区分：

- 预编码 latent→RGB；
- RGB→RGB；
- 单张参考首次编码；
- 同参考复用特征缓存。

**G2 输出：** `docs/v25/GPU3060_RESULTS.md`。

---

# 3. 两个 GPU 开发 Agent：只写代码，不碰显卡

## DEV-CORE：主线科研开发

**文件所有权：**

- `exp/v25/mechanism_a.py`
- `exp/v25/mechanism_b.py`
- `tests/v25/test_mechanism_ab.py`

任务：

1. 实现空间门控，并证明初始与全局混合完全等价。
2. 处理门控 clamp 后均值不再等于 1 的问题。
3. 实现连续强度学生。
4. 测试 γ=0 精确回退。
5. 建立教师配置的一致性检查。
6. 提供 CPU mock 测试。

将经过测试的提交交给 G1。

## DEV-REF：One-Shot 科研开发

**文件所有权：**

- `exp/v25/mechanism_c.py`
- `exp/v25/reference_teacher_v2.py`
- `tests/v25/test_reference_training.py`

任务：

1. 修复教师强度参数接口。
2. 修正参考替换损失。
3. 统一 RGB 外观特征预处理。
4. 加入参考条件消融开关。
5. 支持参考图缓存。
6. 提供无类别 ID 的接口测试。

将提交交给 G2。

---

# 4. CPU 统计与科学诊断：STAT-01 至 STAT-05

这五项优先级很高，不应只写报告。

| ID | 任务 | 具体工作 | 输出 |
|---|---|---|---|
| STAT-01 | **配对非劣性统计** | 完整实现源图聚类 Bootstrap、单侧非劣性、共同随机重采样；验证 150 来源×4 cross | `paired_noninferiority.py`＋单测 |
| STAT-02 | **真正的 Pareto 计算** | 计算 DINO-C/DINO-S/LPIPS/Latency 四维支配关系；输出每个模型族的有效前沿 | `pareto_multimetric.py`＋SVG |
| STAT-03 | **五类与 20 路线诊断** | 统计源锁定、目标混淆、各迁移方向失效；比较 PRISM-1/D6/Hybrid/O3 | `route_diagnostics.json`＋热力图 |
| STAT-04 | **强度旋钮测量** | 从已有结果检查 λ、γ、τ 的单调性、灵敏度、饱和区和逐来源异常 | `strength_calibration.md`＋图 |
| STAT-05 | **One-Shot 参考响应审计** | 从缓存结果构造参考替换矩阵，区分真正风格响应与无目标扰动 | `reference_response_audit.md` |

## 4.1 STAT-01 的关键要求

当前仓库已有：

`src/eval/metrics.py::paired_cluster_ci`

不要重新发明一套不兼容的统计公式。

应增加：

- 非劣性边界判断；
- 双方 task_id 完全一致的断言；
- 同一来源的四个 cross 样本成组抽样；
- 缺失任务立即报错；
- 对 NaN、重复任务、指标方向的检测；
- 单独输出 Effect Size 与 CI。

对于所谓“同内容水平插值”，还要区分：

**可以：** 在真实测得的结果间画描述性插值曲线。

**不能：** 用两条均值曲线插值后，虚构与其配对的 600 个样本，再宣称获得严格配对 CI。

若要严格判断 DINO-C=0.73 的模型优劣，最好生成真实工作点；没有真实输出就只能报告插值的不确定性。

## 4.2 STAT-02 的重要性

现在的可视化将不同 checkpoint 和参数配置串成一条人工选择的曲线。

建议形成三种不同的图：

1. 全仓真实非支配前沿；
2. 每个模型家族自身的可达曲线；
3. 速度/复杂度参与计算的三维或四维前沿。

这能避免“图上的红线看起来连续，但实际上需要切换多个模型”的表述问题。

---

# 5. CPU 数据与特征诊断：DATA-01 至 DATA-05

| ID | 任务 | 具体工作 | 输出 |
|---|---|---|---|
| DATA-01 | **训练—推理锚点分布审计** | 检查真实 `z0_tr`、PRISM-1、T4、D6 训练与推理路径；比较 latent 统计与分位数 | `anchor_distribution_audit.md` |
| DATA-02 | **伪教师质量分布审计** | 分析 `mid_cos`、编辑幅度、各来源分布；检查当前 Top-K 筛选是否偏向近恒等输出 | `teacher_quality_report.md` |
| DATA-03 | **风格编码有效维度** | 对缓存的 128D fiber、795D 外观统计、局部特征做 PCA、相关性和同类/异类距离分析 | `feature_geometry.md` |
| DATA-04 | **SG-25 / WikiArt 数据清单** | 统计风格家族覆盖、尺寸、长宽比、参考数量、异常图片、重复文件及配对覆盖 | `dataset_manifest_audit.md` |
| DATA-05 | **空间编辑分布诊断** | 使用已有 latent 和 PNG，计算不同方法的空间变化图、频谱、局部失真和结构变化统计 | `edit_spatial_audit.md` |

### DATA-01 特别需要回答

V25 机制 B 的计划曾将训练侧 D6 设置为 PRISM-1 锚点，而既有高内容 D6 使用 T4 锚点。

应由 DATA-01 输出清晰的三元组：

`(checkpoint, anchor, solver)`

并比较两种教师的输出分布。

最终规定：**教师三元组不一致，就不得用一个模型的结果解释另一个模型的蒸馏性能。**

### DATA-02 特别需要回答

O2 Teacher 当前对生成结果使用内容相似度筛选。

需要计算：

- 编辑幅度与内容相似度是否高度负相关；
- 被筛掉的样本是否反而含有更明显的风格；
- 各风格家族的保留率；
- 是否出现只学习“少改图”的系统性偏差。

这是可以完全在已有缓存上完成的科研诊断，不需要再训练模型。

### DATA-03 的意义

这是最可能帮助下一阶段设计新风格编码器的 CPU 实验。

如果 795D 外观统计高度冗余，或 RGB ConvNeXt 局部特征与 Oklab 全局统计覆盖的信息几乎相同，可以进一步简化 One-Shot 条件模块。

但本轮只做分析，不直接删除特征分支。

---

# 6. CPU 测量与系统性能：MEASURE-01 至 MEASURE-03

## MEASURE-01：统一端到端性能测量规范

当前仓库存在：

- `tools/benchmark_timing.py`
- `tools/profile_staged.py`
- `tools/profile_staged_3060.py`
- V22–V25 各版本自带的测时逻辑。

问题是它们面向不同历史模型，测量边界并不统一。

任务是编写 `tools/v25_cpu/benchmark_protocol.py` 和测量规范。

统一拆分：

| 阶段 | 必须记录 |
|---|---|
| Encoder | 首次载入、稳态 encode |
| 内容特征 | DINO、亮度结构 |
| 风格特征 | fiber / RGB ConvNeXt / Oklab |
| Anchor | B4、PRISM-1、T4 |
| 主模型 | NFE、有效前向次数、稳态 GPU 时间 |
| Decoder | S10 |
| 数据传输 | CPU↔GPU |
| 完整调用 | RGB→RGB P50/P95、吞吐、峰值显存 |

CPU Agent **只编写、分析和验证测量脚本**；真正需要 CUDA 的测试提交 GPU Agent 排队执行。

必须区分首次调用、热缓存调用和同参考复用缓存调用。

## MEASURE-02：CPU / I/O 性能诊断

对应仓库：

- `src/utils/raw_io.py`
- `src/utils/runtime.py`
- `src/eval/runner.py`

进行纯 CPU 基准：

- 750 个任务的批量序列化；
- PNG 编码与写盘；
- JSON 读写；
- 缓存命中与失败；
- 哈希校验；
- 大量小文件与批量容器格式的 I/O 差异；
- 缓存元数据加载开销；
- Windows/ Linux 路径行为差异。

重点找出评测流水线里哪些耗时根本不是 GPU 推理。

**不得在 4090/3060 正在训练时进行大规模磁盘压力测试。** CPU/I/O 测量优先在独立工作目录与空闲时段执行。

## MEASURE-03：成本模型与性能归因

建立可从静态信息计算的性能模型：

\[
T_{\mathrm{total}}
=
T_{\mathrm{encode}}
+
T_{\mathrm{feature}}
+
T_{\mathrm{anchor}}
+
T_{\mathrm{flow}}
+
T_{\mathrm{decode}}
+
T_{\mathrm{IO}}.
\]

统计：

- 参数量；
- NFE；
- 实际模型前向次数；
- batched CFG 带来的额外算力；
- 额外参考编码器成本；
- 性能随 batch 的变化；
- 是否有机会通过缓存减少重复计算。

输出：

`docs/v25/performance_cost_model.md`

特别应澄清“模型有 20 NFE”与“真实端到端速度”之间的关系，避免继续用缓存后推理时间描述完整产品性能。

---

# 7. 代码整理与工程重构：ENG-01 至 ENG-05

## ENG-01：实验注册表与配置规约

**目标：将跨 V19–V25 的繁杂参数收束为结构化配置。**

新增：

`src/v25/config.py`

建议公开：

```python
TransferConfig(
    engine="core",
    strength=0.5,
    quality="balanced",
)
```

内部固定：

- checkpoint；
- anchor；
- CFG；
- Gate；
- solver；
- NFE；
- feature cache policy。

保留高级配置供研究人员使用，但不作为最终推理 API 的必要输入。

**验收：** 相同配置可生成确定的配置 hash，且能恢复完整历史参数。

## ENG-02：Checkpoint 与 Cache 血缘管理

现有各代实验存在大量：

`if cache_path.exists(): return cached`

需要统一检查缓存是否真的属于当前：

- checkpoint SHA；
- 输入数据 manifest；
- 参数；
- 解码器版本；
- 特征编码器版本；
- Git SHA。

建议新增：

`src/v25/artifact_registry.py`

每个缓存记录明确的依赖哈希。

任何依赖改变都应使缓存失效或生成新的键。

**不得重写、删除旧缓存。** 先建立兼容层。

## ENG-03：统一原始结果与实验元数据

以现有 `RawWriter` 为基础，规范：

- `run_id`；
- task_id 集合；
- config hash；
- checkpoint hash；
- git_sha；
- device；
- seed；
- decoder；
- NFE；
- completion state；
- runtime。

增加结果写入的原子性、断点恢复和重复任务检测。

不要产生第二套不兼容的原始评测格式。

## ENG-04：CPU 单元测试与回归测试

优先新增：

- CFG 双分支与四分支代数一致性；
- `drop_content=False` 时内容条件始终有效；
- γ=0 严格退回 PRISM-1；
- 空间门控初始与全局混合完全一致；
- DWT 正反变换；
- 同一配置下的确定性；
- Bootstrap 重采样正确性；
- 空缓存/坏缓存/错误 hash；
- 原始结果缺 1 个任务时立即失败；
- 批量推理与逐张推理接口一致。

使用小张量、mock 模型与 CPU。

**禁止把真实大模型 checkpoint 当成基础单测依赖。**

## ENG-05：跨平台运行器

统一：

- Linux `tmux`；
- Windows `schtasks`；
- UTF-8 日志；
- 路径；
- 退出码；
- 状态记录；
- 重启后的恢复；
- 超时和失败重试。

目前 3060 已遇到 `schtasks /TR` 引号转义错误，因此建议始终使用 `.bat` 包装脚本，而不是在任务注册命令中嵌套复杂 PowerShell。

提供：

`tools/v25_cpu/run_host_job.py`

仅提交任务描述，不直接决定使用哪张 GPU。GPU 启动权限属于 G1/G2。

---

# 8. 文档与论文：DOC-01、DOC-02

## DOC-01：仓库知识整理

重点整理 V19–V25，而不是重写全部历史。

交付：

1. **模型谱系图**：PRISM-1→V20→V22→V23→V24→V25。
2. **实验结论账本**：有效、失败、尚未证明。
3. **模块清单**：哪些进入 Minimal PRISM，哪些只是研究原型。
4. **checkpoint 清单**：模型权重、对应源码、参数、评测结果。
5. **最小复现步骤**：训练、评测、导出、推理。
6. **错误与修复记录**：CFG 内容清零、DWT 逆变换、缓存分布偏差等。

建议交付：

- `docs/v25/MODEL_LINEAGE.md`
- `docs/v25/ABLATION_LEDGER.md`
- `docs/v25/REPRODUCIBILITY.md`

历史代码不删除，只在文档中标记废弃状态。

## DOC-02：论文材料整理

结合 `docs/icme2027/paper.tex` 和当前实验：

- 统一 PRISM 方法定义；
- 将单步编辑、Flow 编辑、Hybrid 和 Reference Adapter 区分清楚；
- 绘制模型架构示意图；
- 重画 Pareto 图；
- 整理有效消融表；
- 补充真实 RGB→RGB 速度比较；
- 整理失败模式图；
- 生成参考风格迁移视觉拼图；
- 划清五画派任务与 SG-25 OOD 的评测边界。

**本阶段只准备证据与叙事草稿，不提前宣称新机制成功。**

---

# 9. 两个独立审核 Agent

## REVIEW-SCIENCE

只读，不参加训练，也不修改主模型代码。

负责检查：

- 是否有混杂变量；
- 训练与评测教师是否一致；
- Bootstrap 是否正确；
- 外推是否被当成实测；
- 不同实验是否具有相同内容水位；
- One-Shot 是否真的无类别输入；
- 视觉结论是否得到指标支持。

要求对每个新结论给出：

`PASS / FAIL / INCONCLUSIVE`

并附对应证据路径。

## REVIEW-ENGINEERING

只读，负责：

- 共享文件冲突；
- 任务重复；
- 测试覆盖；
- 缓存和 checkpoint 血缘；
- Linux/Windows 可移植性；
- GPU 独占是否被破坏；
- 代码是否能从干净 checkout 运行；
- 文档链接是否有效。

审核通过后，才由总控集成。

---

# 10. 并行策略：大量派发，但分波执行

## Wave 0：立即启动

GPU：

- G1：继续 4090 Round 2。
- G2：继续 3060 Round 3。

CPU：

- STAT-01：配对 CI。
- STAT-02：Pareto。
- STAT-03：路线诊断。
- DATA-01：锚点分布。
- DATA-02：教师质量。
- ENG-02：缓存血缘。
- MEASURE-01：性能测量规范。

DEV-CORE、DEV-REF 同时开始编写机制代码。

**这些任务的结论会直接影响后续 GPU 实验，所以必须优先。**

## Wave 1：0–2 小时

在 GPU 继续原队列时，再派发：

- STAT-04/05；
- DATA-03/04/05；
- MEASURE-02/03；
- ENG-01；
- ENG-04。

总控优先接收 Wave 0 的结果，为机制 A/B/C 提供修订意见。

## Wave 2：2–6 小时

GPU：

- G1：机制 A→机制 B。
- G2：机制 C1。

CPU：

- ENG-03：结果格式；
- ENG-05：跨平台运行器；
- DOC-01：知识整理；
- DOC-02：论文材料。

审核 Agent 开始检查第一批已完成任务。

## Wave 3：6–10 小时

GPU：

- G1：最佳机制复验；
- G2：C2/C3 与 OOD。

CPU：

- 统计 Agent 对新结果进行自动再分析；
- 数据 Agent 对新增伪教师包做质量审计；
- 测量 Agent 解析新 benchmark；
- 工程 Agent 完善最小推理接口；
- 文档 Agent 更新实验结论。

## Wave 4：10–12 小时

GPU：

- 最终候选的完整 test750、性能测量与必要复验。

CPU：

- 最终 Pareto 与 CI；
- 模型与 checkpoint 注册；
- 文档链接；
- CPU 回归；
- 代码审查；
- 方法图和结果表；
- 双接口验收。

总控合并并提交最终结果。

---

# 11. 防止 CPU Agent 反过来干扰 GPU

CPU-only 的含义不仅是不调用 CUDA，还包括不造成明显的 CPU、内存和磁盘竞争。

建议限制：

- 普通 CPU 工作任务：最多使用 2 个计算线程；
- 同一 GPU 主机上的 CPU 计算任务：原则上最多 2 个同时执行；
- 多个只读分析 Agent 可以并行，但重型数据读取应排队；
- 禁止在训练期间进行全盘 hash、全量图片重编码或大规模磁盘压力测试；
- 限制 CPU 内存峰值，避免触发 swap；
- CPU-only 任务不加载需要 CUDA 的模型，不调用 `torch.cuda`；
- 所有共享缓存只读访问；
- 不得修改正在运行的 Python 环境或安装/升级公共依赖。

Linux CPU 任务可以设置：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m tools.v25_cpu.TASK_NAME
```

Windows 则使用单独 PowerShell 环境变量，并设置低于 GPU 任务的进程优先级。

如果有其他空闲机器，统计、文档和测试任务应优先转移过去。

---

# 12. 独立分支与文件所有权

每个子 Agent 必须在开始前领取任务编号。

建议使用：

`agents/v25-stat01`

`agents/v25-data02`

`agents/v25-eng04`

等独立分支。

CPU 任务的新增文件统一放入：

- `tools/v25_cpu/`
- `tests/v25/`
- `docs/v25/audits/`
- `src/v25/`

每个任务拥有唯一文件名前缀。

例如：

- STAT-01 只拥有 `tools/v25_cpu/stat01_noninferiority.py`；
- DATA-02 只拥有 `tools/v25_cpu/data02_teacher_audit.py`；
- ENG-02 只拥有 `src/v25/artifact_registry.py`；
- DOC-01 只拥有其明确分配的文档文件。

**以下文件仅总控允许修改：**

- `docs/performance/metrics.csv`
- `docs/performance/metrics.json`
- `docs/19_pull_together/PROGRESS.md`
- `docs/performance/MASTER_LEDGER.md`
- 最终 `src/model/prism_v25.py`
- 主分支的公共配置与项目依赖。

CPU Agent 提交自己分支的 commit；总控进行 cherry-pick 或有序合并。

运行中的 GPU Agent 必须锁定代码 commit 和配置 hash，不能因主分支更新自动改变运行代码。

---

# 13. 每个子 Agent 的标准交付契约

每个任务必须明确：

| 字段 | 内容 |
|---|---|
| `task_id` | 唯一编号 |
| `base_sha` | 开始时的代码版本 |
| `owner` | Agent 名称 |
| `gpu_allowed` | 默认 false |
| `files_owned` | 允许修改的文件 |
| `inputs` | 明确输入数据和缓存 |
| `outputs` | 代码、报告、图表、测试 |
| `acceptance` | 可验证完成条件 |
| `dependencies` | 先决任务 |
| `status` | pending/running/done/failed/blocked |
| `commit` | 最终提交 SHA |

任务完成不能只回复“已完成”。

必须提交：

1. 简短结果摘要；
2. 修改文件列表；
3. 运行或检查命令；
4. 测试结果；
5. 数据或代码依据；
6. Git commit；
7. 已知限制与后续建议。

如果某任务发现主线中的数学错误，应向总控提出 blocker，而不是直接修改共享代码。

---

# 14. 总控如何运行整场任务

总控只做六件事：

1. **任务拆解与分配。** 保证每个 Agent 有独立所有权。
2. **GPU 排队。** 4090 和 3060 各一个执行队列，禁止其他 Agent 直接触碰。
3. **接口冻结。** 在 Round 1/2 收敛完成后公布最终模型配置。
4. **交叉审查。** 科学结论交 REVIEW-SCIENCE，工程改动交 REVIEW-ENGINEERING。
5. **冲突合并。** 按依赖关系集成，不同时修改同一文件。
6. **最终交付。** 生成统一报告、测试、模型接口与 Git 提交。

建议每完成一个里程碑生成一次汇总：

```text
V25 STATUS

GPU-4090:
  running:
  completed:
  next:
  blocker:

GPU-3060:
  running:
  completed:
  next:
  blocker:

CPU AGENTS:
  done:
  running:
  blocked:

NEW FINDINGS:
  confirmed:
  rejected:
  inconclusive:

INTEGRATION:
  merged commits:
  pending review:
  failing tests:

NEXT PRIORITIES:
```

GPU 队列不因 CPU 文档未完成而阻塞；CPU 任务也不因某个机制判负而整体停止。

---

# 15. 今晚最终应当获得什么

## 科学成果

- Minimal PRISM 最简配方；
- 双分支与四分支的最终取舍；
- T4/PRISM-1 锚点取舍；
- Hybrid 是否必要的实测边界；
- 机制 A/B/C 的受控结论；
- One-Shot 的参考响应与教师强度归因；
- 可复算的配对置信区间；
- 真正的多目标 Pareto 前沿。

## 工程成果

- 统一 checkpoint 与缓存血缘；
- 统一推理配置；
- 统一测量工具；
- 统一结果格式；
- 独立 CPU 回归测试；
- 跨 Linux/Windows 执行入口；
- 失败恢复能力；
- 可复现的 `PRISMCore`、`PRISMRef` 接口。

## 文档成果

- 模型谱系；
- 全阶段消融结论；
- 科学失败模式；
- 结果与配置注册表；
- 重绘后的 Pareto 图；
- 端到端性能报告；
- One-Shot OOD 评测说明；
- 论文方法与结果初稿。

**验收重点不是提交数量，而是：是否减少了最终模型的复杂度，是否提高了结果可信度，是否让下一阶段能够在干净基线上继续推进。**

---

# 16. 可以直接交给 Coding Agent 的总控指令

你现在担任 WEAVE V25 多 Agent 执行总控。

请完整阅读：

- `docs/19_pull_together/README.md`
- `01_convergence_plan.md`
- `02_mechanism_plan.md`
- `03_schedule_and_eta.md`
- `04_round1_results.md`
- `PROGRESS.md`

以及本多 Agent 计划。

执行要求：

1. 首先核查本地 Git HEAD、现有两台机器的实际运行任务和工作区状态，不得重复启动已在运行的实验。
2. 立即将 GPU-4090 和 GPU-3060 的任务分别交给独立子 Agent，各自独占其 GPU。
3. 将机制 A/B 的代码开发交给独立 CPU-only 开发 Agent；机制 C 同理。
4. 并行派发 STAT、DATA、MEASURE、ENG、DOC 专项任务，优先启动 Wave 0；具备足够子 Agent 并发能力时，可以扩大只读分析并发。
5. 所有 CPU Agent 禁止运行 CUDA，且不得修改 GPU 正在执行的工作区与公共文件。
6. 每个 Agent 必须有独立文件所有权、明确验收条件和 Git 分支。
7. 运行中的实验以已经冻结的代码和配置为准；新发现的错误必须经过审查后决定是否中断。
8. 每完成一项实验立即保存原始结果、配置 hash 和 checkpoint hash。
9. 前一个机制失败不能阻断其他独立机制。
10. 统计、测量和图表必须基于真实数据，不得把曲线外推当成实测。
11. 每个子 Agent 完成后交由对应审核 Agent 审阅，最后由总控合并。
12. 每个里程碑更新进度报告，但不得为了报告中断训练。
13. 所有正式 WEAVE 质量结论必须使用完整 test750，不允许恢复 dev200 筛选。
14. 最终进行 CPU 回归测试、必要 GPU 复验、端到端性能测量和双接口验收。
15. 将最终成果统一整理到 V25 文档目录，形成可复现交付。

**立即按照任务依赖启动并行工作，不需要等待全部准备工作完成才开始分派。**
