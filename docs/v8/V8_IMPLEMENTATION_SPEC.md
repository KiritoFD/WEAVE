# V8 模块实现清单与接口

导航：[总方案](<V8_24H_RESEARCH_PLAN.md>) · [测试验收](<V8_TEST_ACCEPTANCE.md>) · [固定矩阵](<V8_MATRIX_AND_RUNBOOK.md>)。

**所有条目均为待实现任务，不是完成声明。** 代码建议新增 `tools/v8_24h/`，不覆盖已有 Bridge 代码。

## 1. core：配置、数据契约和产物管理

### 1.1 common.py / protocol.py

实现 `FrozenAssets`、`LatentSpec`、`TaskManifest`、`SourceBatch` 与路径解析；启动时实查 B3/B4/Q2a/R1/SD-VAE/参考缓存，不猜路径。

- tasks750 必须唯一750 ID、150源、每源5目标、Cross600/Same150、20 routes每条30。
- Encoder 输出未缩放 posterior mean；模型输入可能带 scaling factor。显式适配标准VAE/B4/R1的真实约定；测试 encode/model/decode 连通，不从历史文件名推断。
- `post_quant_conv` 一次且仅一次；SD0-S10 与旧 A0-S10分别记录训练接口，不要求所有 S10 都配 A0。
- 固定模型版本、preprocess、参考集、RGB零点、PNG量化、同源参考 self-mask 规则。所有五风格 S 分数应用一致 self-mask。
- 参数计算方式、种子、task/file/raw-index映射写入schema。未解析 painter ID 标UNKNOWN，不拿样本数当画家数。

### 1.2 registry.py

实现不可变 `ArmSpec` 和 `expand_registry()`：阶段、族、起点、目标能量、几何条件、内容保护、剂量、seed、依赖、训练更新数、checkpoint列表和必要产物。

运行状态与科学结果分开：`run_state=DONE` 可对应 `science=NO`。不得让 NO 触发后续取消。详细状态机见 [Runbook](<V8_MATRIX_AND_RUNBOOK.md>)。

### 1.3 artifacts.py

实现原子写入、schema/hash校验、task级续写、完整性验证、history只读检查、运行目录锁。临时文件校验后 rename；不以存在 eval.json 判完整。缓存键包括代码revision/模型hash/数据hash/配置hash。

保存750最终latent和PNG；存储空间预检，压缩包有逐文件hash。训练缓存用分片fp16，trace不能只存batch均值。

## 2. objectives：真实多模态目标

### 2.1 features.py

实现 `GeometryDescriptor` 与 `StyleDescriptor`，统一64个空间patch（64×64 latent的8×8区域，对应32×32 Haar HF的4×4区域）。不强制保留绝对边缘幅值。

- 几何 q：方向用 cos(2θ)/sin(2θ) 避免角度跨界；结构张量coherence、相对边缘rank、局部自相似、连续平坦/轮廓/纹理特征。多尺度几何保持空间对应。
- 风格 h：原始Haar HF相位展开，用作历史相位路线。
- 风格 u：每patch通道幅值/能量、局部相关/协方差等相位稳健统计，明确定义维度与数值稳定项。
- LL：独立低频均值/方差/协方差与必要空间统计；不把HF当全部风格。
- **统一训练池尺度白化**：跨五画派共用拟合的均值/尺度，含D维归一化；不按目标画派分别白化而掩盖画派差异。特征/白化hash进配置。

### 2.2 bank.py

实现 `build_clean_source_index()`、`parse_painter()`、`build_bank()`、`calibrate_on_train()`。

1. 扫描15545真实latent与候选15045池，解析样本ID→文件stem/raw-index/原图hash/painter。
2. 对全部评测源和独立参考图精确排重，必要时补图像hash与latent重复检查；去重前后数量和每风格有效画家数公开。
3. 样本或画家来源不可验证时，不构建声称“无污染/画家隔离”的bank；阻断受影响bank，其他已验证历史基线继续。
4. 正式原型记录真实q/h/u/LL、style、painter、raw_index、source_hash、patch坐标。每风格8组×96原型为首版登记规格；半容量对照固定4组×48原型，组映射/抽样规则预先锁定。
5. 所有bank共享同样的数据与特征尺度；条件消融共用prototype集合，不能额外增加库容量。G12使用painter-balanced同数量原型，参考默认库也说明抽样策略。
6. 不生成全部原始patch巨大两两矩阵。分块特征提取、原型采样、统计校准，记录峰值CPU RAM/GPU VRAM和覆盖率。
7. 温度τ两档根据训练池同画派距离分位数锁定；OT ε与训练绝对能量校准在训练/画家诊断划分上确定，不用750选温度。画家完全隔离的校准样本不混入拟合原型。

### 2.3 mixture.py

实现以下逐图 `Energy`：

- M0 `LegacyE3Energy`：调用冻结源码快照，保留原损失/子采样和算法；明确与新bank规模区别。
- M1 `CentroidEnergy`：固定π的平方距离期望，明确是加权均值回归基线。
- M2 `MixtureSupportEnergy`：`−τ mean_i logsumexp(log w_ij − d_ij/τ)`，权重和=1，`d=whitened_dist²/D`；支持任意几何prior、两档τ、h/u/LL路线。
- 返回责任r_ij、有效模式数/entropy、原型使用率、每图绝对目标支持、HF/LL分量。

权重应非负、可归一；无条件均匀、几何条件和错误目标共用相同能量计算。logsumexp稳定，温度不混用全量/维均值距离。不能以降低loss直接判科学成功。

### 2.4 transport.py

实现真正逐图Patch集合OT与SWD，不沿用B×N展平的组中心算法。

- 输入每图64个生成patch和目标原型（首版最多768个）；`cost[B,N,K]=style_dist+λ_g geom_dist`，源q固定，style_dist随当前生成h变化。
- 每图独立r/c，log-domain Sinkhorn；记录行/列边缘误差、有效使用原型数、停止状态。
- 目标为 `<π,cost>+ε KL(π||r⊗c)`；π detach可用近似envelope gradient，**仅在求解收敛和有限差分验证成立时**；未收敛不可假称梯度为真OT梯度。训练/编辑明确solver和反传方式。
- 计算G-shuffle/G-wrong时仍使用同solver/ε/原型数/边缘；只改条件。所有batch内图完全独立。
- SWD：训练seed固定单位投影方向，比较两个集合投影分位数（N≠K时统一分位点），不是均值/单patch投影距离。采样方向hash固定。

### 2.5 specificity.py

实现按画家校准的绝对能量及有界目标区别：`softplus(m+E_t−softmin(E_other))` + 独立绝对支持E_t。

- 负类项有预登记的饱和/权重上限；保留目标支持，不能通过远离全部画派获益。
- 同一标量能量用于方向与接受；如设置负类梯度裁剪，须对应可解释的有界势函数，或明确使用广义方向且保持同目标下降检查，不能偷偷变更目标。
- 输出E_t、四个E_other、校准NLL-like score、支持差；仅称模型能量/校准支持，不把无归一化分数称为精确概率密度。

## 3. editing：方向、约束与干预

### 3.1 legacy.py / audit.py

实现历史适配、代码事实与trace审计。保留正确Oracle `+∇margin`；损失下降探针使用`−∇loss`。H0完整复测R1/S1/S2的正/反/错目标，补B4零更新；随机五seed正确时复用、否则按task级共享噪声修复重测。

- legacy4使用真实历史调用链，不假设zr是初始z0。存`z_k` hash、调用输入、loss和α。
- true4实现新契约，与legacy的差异逐条说明；没有累积问题就写“未复现不累积”，不创造一个人为nonaccum版本伪装历史。
- noC/disp、E0/E1比对解析配置、分支ID、source/z0/final latent、逐图trace和缓存键。结构等价、数值等价、文件复用是三种不同结论。

### 3.2 editor.py

`Editor.run()`逐图执行：

1. 当前z_k计算各项逐图loss和梯度，step0预登记标定s_l；梯度零的项不得用1e-12放大噪声。
2. 固定`E_used=Σ(w_l/s_l) L_l`，方向和线搜索始终同一标量；记录raw、scaled、weighted norm及cosine。
3. direction做逐图RMS/TopK，4步每步名义0.08；8步每步0.04，总名义预算0.32。
4. 生成候选→逐样本投影RMS球→重新算同一目标和有限步内容代理→逐图接受。
5. 全拒绝保持当前z；零方向保持；有非有限值标数值失败，不硬前进。
6. 输出实际接受步数、累计路径长度和相对z0终点位移（两者不同）及每次候选理由。

候选α列表、下降容差、内容δ、ρ在preflight锁定。ρ来自重新审计后的R3c/R4c及R1逐图终点位移分布，保存分位数与两个档位；不能依据错误trace临时猜。

### 3.3 content.py

- C0：只RMS约束。
- C1：有限步代理接受`L_C(z_cand,z_s)≤L_C(z0,z_s)+δ`；代理可用冻结canon/结构descriptor，独立DINO-C/LPIPS仍是评测。
- C2：内容敏感度预条件，JVP/VJP实现`(I+ηλJᵀJ)d=g_style`，2–3次CG，不形成dense Jacobian；符号和小矩阵精确解对照测试。
- 代理质量诊断：训练池受控扰动与结构破坏测试，报告它与真实内容损伤的相关；代理不相关也照跑登记C2，结果标为负机制证据，不在运行中替换。
- T9等最终位移R1对照使用配对task的预先保存预算，或统一ρ加实际终点匹配策略；若无法精确匹配，报告误差，不以相同名义步长称为等位移。

### 3.4 interventions.py

保持同源z0固定，只改edit_target；同时记录request_target/edit_target。几何打乱使用每源seed、其他源几何按固定错配表；不得跨batch现场roll引入排列依赖。

Same150默认按该臂登记策略，不能所有方法偷偷不编辑；另外“same不编辑”为明确对照。随机噪声由(source_hash,seed)生成，四个跨风格目标共享，同seed跨objective共享，跨batch/倒序不变。

## 4. learning：教师与学生

### 4.1 teacher.py

固定A=R4c/E3历史教师版本（登记唯一选择，不运行时择优）、B=M2+内容约束、C=M3+内容约束。

每教师3000个干净训练源×四个其他目标=12000任务；共36000。分片缓存fp16 source/canonical/z0/终点与接受轨迹，包含目标、painter、hash、每步梯度/结构状态/accept；所有四目标都生成，不把源style误当target、不把batch第一个目标用于全部样本。

- B4实际forward签名从代码核验，使用真实canon输入，不使用全零content近似。
- 分片保存/恢复精确到task，不靠目录存在；未见画家诊断源另建，训练/诊断互斥。
- 在线教师生成仅在缓存阶段；学生训练用缓存状态，不每update在线重跑完整OT教师。
- 36000任务超预算则保留剩余分片队列，D0与已就绪独立训练可继续，不能缩成几百张。

### 4.2 models.py

- `Delta142K`：参考142004参数模型，但显式输出delta再ClipRMS；不先z0+delta再相减造成数值问题。零初始化与每样本硬rho。
- `Policy142K`：4步共享网络输入当前z、target embedding、停止梯度的style gradient、结构状态；输出有界方向/步长/接受分数。状态投影及heads纳入参数数目，要求142K±20%。明确一阶/截断训练，不叫DEQ/完整二阶元优化。
- `Delta500K`：同Teacher-B监督、相同4000updates与有效batch，约500K实际参数记录；不更改教师数据以制造容量比较。
- source条件不自动加入；本矩阵主差异是终点/策略/容量，而非重复N1/N2。

### 4.3 losses.py

- LL/HF分带delta监督、direction cosine、RMS ratio、相对误差；每项实际加权梯度norm审计，不只打印分带L1。
- policy用轨迹的当前状态→教师方向/步长/accept监督；拒绝样本与零更新有明确定义，不能只学缩残差。
- D0是固定R4c风格/内容目标的直接损失训练，不读取教师终点作为GT；D1–D3是终点蒸馏；D4/D6是policy；D5是容量对照。
- 风格梯度在训练中是否detach、哪些路径保留autograd写入配置。CLIP/DINO/CSD/LPIPS不进入任何训练loss。

### 4.4 train.py

D0–D6各4000 optimizer更新，有效batch240；microbatch×accum明确，batch5风格/画家平衡。bf16 warmup100，优化器/lr/scheduler/seed42或43锁定，同监督臂不偷改。

保存0/1000/2500/4000，非零全部750；0检查共享初始化与B4精确一致，若不相同单列评测，不静默复用。OOM仅减microbatch增加accum，保持update/effective batch/数据；非有限训练暂停该臂、两次有记录修复重试。

保存checkpoint时先创建parent，不将.pt路径当目录；checkpoint重载数值、键前缀、optimizer/scaler/RNG恢复都要验收。原始/EMA若使用，必须预登记，不临时择优。

## 5. reporting：完整评价与真实速度

### 5.1 evaluate.py / statistics.py

- 生成750标准Decoder RGB、PNG、source/z0/final latent及features；CLIP ViT-B/32等版本按协议核实，五目标prototype与DINO reference保持冻结。
- 每row：task/source/request/edit IDs、五风格DINO/CLIP/CSD分数、G、原RGB相对变化、Top1/Margin/ΔT、DINO-C、LPIPS；正确/错目标输出同时对请求和实际目标评分。
- I：每源四跨风格输出组成完整组，固定评分类别t比较y_t与其他三输出；先按源计算I，再源cluster bootstrap ΔI。标签无响应合成数据I=0。
- 聚类bootstrap4000、源cluster重复抽样保留重复索引；ΔG/ΔI/质量与正确-错误/几何差的CI，seed变异独立报告。
- 统一Same150 schema，不再混淆same_lpips和lpips；缺task或缺CLIP不是NaN填表放行。

### 5.2 diagnostics.py

逐五风格、20 routes、ER/Ukiyo-e、绝对支持、prototype覆盖/entropy、跨painter库干预、共享噪声、same漂移、seen/unseen painter统计。固定案例按源和场景预选，保存所有正式arm；另选最差10目标G与最大10内容损伤（公开规则）。生成visual_review表，人工尚未看时PENDING。

### 5.3 speed.py

质量候选对应的实际latent/config测试batch1/batch8：Encoder、canon+B4/Q2a、R1/检索/OT/更新每步、student各step、标准Decoder/S10、总生成。

- warmup5、测量20次为首版；CUDA event计GPU段，synchronize墙钟计端到端；repeat数/计时边界预登记。
- 纯生成不含DINO/CLIP/CSD/LPIPS/PNG存储，模型加载和bank CPU→GPU预载另列；若部署每请求搬运bank，另列真实冷/热路径，不隐藏搬运。
- 同次实际E2E计时，不能stage相加当实测。报告batch1 ms/request和batch8 batch_ms、摊销ms/img、img/s、峰值VRAM、实际接受步数。
- SD-VAE与S10分别验证Encoder/PQC/latent scale；标准质量+标准速度主表，S10单独cross配置，不能配错接口后合并结果。
- policy使用在线style gradient就计入，不把142K forward时间当整个policy推理。

### 5.4 report.py

从注册表而非已有eval glob合并所有arm。输出完整arms.csv、manifest、运行/科学状态、三个联合标准、双2D前沿、可用真实速度的3D前沿、失败清单、未完成队列和24→48h剩余工时。

外部方法以论文主表质量/速度原文为准，不猜速度。未知速度不画大气泡；不按不同方法或是否ours随意缩点。同shape圆形、统一图例；正式默认半径正比img/s，`scatter.s ∝ speed²`，极慢点可另用明确locator。前沿是离散支配，连线只是展示；相邻点插值不是新模型。

主报告只描述实测；阶段预算/局部smoke/固定batch过拟合不当成绩。科研门槛、实现门禁、视觉人工确认分别列出。
