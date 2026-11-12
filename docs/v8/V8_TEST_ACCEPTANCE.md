# V8 测试与启动验收

导航：[总方案](<V8_24H_RESEARCH_PLAN.md>) · [实现清单](<V8_IMPLEMENTATION_SPEC.md>) · [串行队列](<V8_MATRIX_AND_RUNBOOK.md>)。

本清单是**待实现的测试契约**。只测关键性质，不写镜像实现的冗余断言；局部单测、GPU数值冒烟和正式750分别记录。禁止用try/except吞异常或恒真断言制造通过。

## 1. CPU：配置、数据、指标、状态

| ID | 输入/操作 | 必须检验的结果 |
|---|---|---|
| P01 | 合法/缺失/重复task清单 | 750唯一ID、150源×5目标、Cross600/Same150、20×30；异常拒绝，不自动丢行 |
| P02 | 有已知painter/stem/raw-index的样本 + UNKNOWN | 正确解析；UNKNOWN不当独立画家；样本数不等于画家数 |
| P03 | 训练/源/参考中同stem、不同stem同hash、同raw-index冲突 | 全部登记排除/冲突，不靠[100:]声称无重叠；最终源manifest可追溯 |
| P04 | G的5风格分数合成向量 | 精确符合公式、target索引/顺序正确；target/other共升时ΔG=0 |
| P05 | 同源四输出完全相同 | I=0；只压低源风格可提高G，但I仍0；目标响应矩阵有预定对角增益时I为预期值 |
| P06 | 150源cluster、重复抽样的固定bootstrap draw | 重复簇不能被set/isin去重；Δ=0的CI为0；输入交换后delta/CI符号翻转 |
| P07 | 错目标输出 | 对request_target和edit_target分别计分；相同z0 hash，其他config相同 |
| P08 | per-task metrics随机重排 | 按task_id配对结果一致；缺ID直接失败，不能取交集冒充全量 |
| P09 | 旧same_lpips、新统一字段和缺值 | 明确schema转换；缺CLIP/关键字段报INCOMPLETE，不用NaN填过 |
| P10 | 一个NO、一个FAILED、一个依赖缺失的队列 | NO不取消独立arm；最多2次修复重试；下游标BLOCKED而非NO |
| P11 | 运行中断/重启/文件写到一半 | 持久计时不重置；只有完整hash/schema产物才恢复DONE；半文件不复用 |
| P12 | 样例Pareto含ours、随机、重复、未知速度 | 外部/ours双前沿正确；诊断排除；未知速度不能继承45.78；同吞吐点同大小 |

**随机契约**：用(source_hash,seed)的独立生成器，same-source四跨目标共享噪声；不同seed不同；同seed跨objective可以相同。测试batch=1/8/倒序/分片/续跑后同task噪声逐元素一致。

## 2. CPU/小规模FP64：数学与优化性质

### 2.1 Features / bank / mixture

| ID | 测试 | 通过证据 |
|---|---|---|
| E01 | geom8×8与HF4×4网格 | 同N=64且局部扰动影响对应格子，不能只检查shape |
| E02 | 单轴边缘/旋转/幅值缩放/平坦patch | orientation连续、数值有限；相对几何不锁死绝对幅值；std零有保护 |
| E03 | q_source requires_grad、z_generated requires_grad | 对z梯度非零且有限，q梯度None；不能loss无grad_fn后吞异常就通过 |
| E04 | 原型均衡采样和半容量库 | 风格/画家/索引一致；负对照共用同prototype集合；样本hash排除正确 |
| E05 | 固定π平方距离 vs weighted centroid | 梯度一致，明确验证M1是均值回归；不是“多模态通过” |
| E06 | 一维双峰原型，低温mixture | 责任随当前h改变；近两峰支持高、空白中点支持低，区别于M1均值；高温极限单列 |
| E07 | 极大距离、低温、空组 | logsumexp有限；空组fallback有记录且不混风格，不出现NaN |
| E08 | 全画派whitening | 训练拟合一套尺度供五风格用；评测数据不能改统计 |

### 2.2 OT / SWD / specificity

- **E09**：逐图log-Sinkhorn输出非负、质量总和正确、行列边缘最大残差首版≤1e-3；FP64合成解目标与可知最优相符。实际阈值在preflight锁定，不在看750成绩后放宽。
- **E10**：同一个样本单独、与不同图拼batch、倒序、分片运行结果一致；默认GPU FP32误差 `atol=1e-5, rtol=1e-4`，超差先查真实耦合/求解收敛，不能盲目放宽。
- **E11**：几何/目标负对照配置diff中只允许指定字段；wrong_target仍OT，shuffle仍同ε/边缘/求解器。逐图哈希证明确实换库/换序。
- **E12**：OT style cost变化会改变plan；detached plan的envelope gradient通过小FP64有限差分。未收敛时测试必须拒绝“精确OT”声明并记录残差。
- **E13**：SWD对集合排列不变、同集合为0、平移增大，N≠K分位匹配正确；不能只比均值。
- **E14**：有界区别项有限、满足margin后梯度减弱；构造所有目标绝对支持变差而margin变好的样例，验收器不能判成功。负类梯度上界与实际总目标一致。

### 2.3 Editor / content

- **O01 符号**：对每个新loss选非零梯度样本，在FP64/FP32小ε范围验证`L(z−εg)<L(z)`与中心差分方向导数；对R1单独验证`margin(z+ε∇margin)>margin(z)`。DINO/CSD变化不作为自身损失符号测试。
- **O02 累积**：已知二次loss，记录调用链每步输入hash；第k+1步输入等于第k步输出。核查真实legacy代码，不预设旧版不累积。
- **O03 全拒绝**：构造所有候选都增loss或超内容δ；终点必须逐元素等于当前z，accept_count=0。
- **O04 投影后接受**：构造未投影候选下降、投影后不下降的例子；必须拒绝；记录projected loss，不比较原候选。
- **O05 一致标量**：自动求导`E_used=Σ(w/s)L`所得梯度与提议方向符号一致；线搜索使用同E_used且step0 scales固定，不跨图均值耦合。
- **O06 硬约束**：每图最终RMS≤rho（FP32容差1e-6）；混合batch中一图超限只投影该图；记录路径长度不冒充终点位移。
- **O07 零/非有限梯度**：零梯度保持；NaN/Inf标失败；不放大1e-12噪声、不偷偷前进。
- **O08 batch独立**：TopK逐图、归一化逐图、接受逐图；B=1/8/倒序的输出和accept mask一致到数值容差。不能只测无编辑分支。
- **O09 JVP/CG**：小线性proxy的JᵀJ与显式矩阵解相符；正定系统2–3次CG残差公开；零proxy退化为原风格方向；不构造真实大Jacobian。
- **O10 反事实**：同source/z0/预算，仅edit_target变化；源q和request_target不变，输出用两种target评分。

## 3. Learning：先做可学习性单测

| ID | 测试 | 验收要求 |
|---|---|---|
| L01 | 142K delta/policy、500K实例化 | 输出shape/parameter实际计数正确；policy包含全部投影/heads，142K±20% |
| L02 | 零初始化 | step0输出精确=B4（FP32最大差≤1e-6）；零delta不能产生sqrt(0)梯度NaN |
| L03 | 强制大残差 | ClipRMS在真实forward每图生效；clip分支前后梯度有限；不是仅loss正则 |
| L04 | 固定batch与已知非零教师delta | 有限步内loss明显下降、方向cosine转正并趋近教师；不能靠输出全零通过 |
| L05 | 固定batch 4步教师轨迹，含接受/拒绝/零更新 | policy再现方向/步长/accept；输出界与累计总界成立；拒绝不是缩步前进 |
| L06 | 多目标混合batch | 每样本使用自己的tid、bank、q；四目标不同，不能全用tid[0] |
| L07 | 教师manifest | source不在750/参考；每源4个非源目标；12000任务/教师，重复/缺失检测；UNKNOWN画家不混作unseen |
| L08 | checkpoint保存/重载/恢复 | .pt是文件；前缀严格；step/optimizer/RNG/seed恢复，下一更新与未中断路径一致到约定容差 |
| L09 | accumulation vs单大batch | 有效240样本的loss/梯度归一正确；microbatch OOM缩小不改update数和数据 |
| L10 | 分带loss | LL/HF实际加权梯度可测；每项对模型有梯度；不是只打印误差代替梯度审计 |
| L11 | D0与D1–D6监督路径 | D0不读取teacher endpoint；delta读终点、policy读轨迹；D4/D6无在线完整OT教师反复生成 |
| L12 | frozen资产 | B4/VAE/R1参数不变且grad为空；R1可有对latent的梯度，DINO/CLIP/CSD/Decoder无训练反传 |

固定batch过拟合只诊断代码与优化，不用于筛选科学模型或超参数。若未通过，先修实现，不用4000更新掩盖失败。合成可学任务必须先过，真实教师batch若难拟合要作为可学习性发现保留，不擅自删D臂。

## 4. 3060 GPU 冒烟与全量评测验收

所有GPU冒烟仍严格单卡串行；与旧训练/评测不抢卡。运行前检查真实进程和显存，不靠某次利用率低就启动另一个GPU任务。

### 4.1 小批次数值闭环（启动前）

1. B4、正确Oracle、旧E3各1个真实batch，确认scale/PQC/输出范围。
2. mixture、OT、SWD各跑1步与4步，含混合target batch；记录有限能量下降和约束。
3. OT/TopK/editor B1/B8/反序测试；q/whitening/bank固定，不能换库后比较。
4. delta/policy/500K各做少量optimizer更新、存盘、重载并推理；记录step/s/VRAM。
5. Teacher-A/B/C各一小分片生成、失败恢复与task去重；测teacher_task_sec，不缩减正式36000任务。
6. 标准Decoder与每种S10接口独立验证，post_quant_conv一次，官方与S10不能混用同一未校验wrapper。
7. 故意制造一个异常arm，确认进程树/日志/状态/下游阻断正确且不取消独立arm。

小批次任务不生成正式榜单成绩；不使用Val200/Test200筛候选。

### 4.2 一次完整750接口验收（启动前）

用标准B4做完整750，验收：

- 保存750个PNG、750个final latent、完整task_id，不仅一张grid。
- 每row五风格DINO/CLIP/CSD分数，G齐全；I由每源四跨目标输出组重建。
- Cross600/Same150和20×30精确，same字段正确，自参考mask一致。
- CLIP-S/Style Affinity按冻结参考协议计算，不能占位。
- 输出图重新读PNG可补算CLIP，与同RGB量化路径一致。
- 对既有B4数值采用预登记回归容差（首版DINO-S/C≤5e-4，LPIPS≤1e-3，Top1最多1个task差；超差定位版本、参考、解码，不自动改阈值）。该容差用于协议核对，不用于科学显著性。
- 中断部分任务后恢复，不重复评分、不丢ID；已完成750重启应零重新生成。
- 从产物生成partial report，在缺一个正式arm时必须INCOMPLETE。

若与H0.B4相同code/config/hash，完整验收输出可登记复用，避免重复GPU评测；复用必须显式写provenance，不把小冒烟当750。

## 5. 真实测速验收

- warmup5/重复20，batch1延迟与batch8吞吐各计，方法加载/预载/冷启动分开。
- CUDA event段与同步E2E墙钟核验，报告encoder/predictor/每步检索或梯度/student/decoder；段和可以不等于E2E，不能代替E2E。
- 关闭评测特征和存图后测纯生成；同步边界/数据搬运明确。
- 同一个质量arm的模型、bank、实际步数和Decoder必须对应；policy的在线梯度计入。
- 记录显存、实际accept步数、batch末尾padding/引擎fallback。batch8 TRT engine不能冒充batch1 engine；无batch1引擎先列torch官方路径或预算内构建，不偷偷套batch8/8。

## 6. 放行门禁（只针对工程，不按成绩取消实验）

`preflight.json` 至少含：

```json
{
  "source_revision": "待锁定实际commit",
  "registry_hash": "待生成",
  "frozen_assets_verified": false,
  "data_overlap_resolved": false,
  "cpu_contract_tests": "PENDING",
  "gpu_numerics": "PENDING",
  "eval750_contract": "PENDING",
  "teacher_and_student_roundtrip": "PENDING",
  "runner_fault_recovery": "PENDING",
  "runtime_and_storage_estimate": "PENDING",
  "ready_for_serial": false
}
```

全部关键项通过才能ready=true；不是27项局部断言通过就算实现完成。正式运行每arm继续执行轻量完整性检查。

## 7. 实验结束验收

- 从不可变registry对照全部正式arm/教师分片/checkpoint，不从存在的文件glob反推已完成矩阵。
- `ALL_COMPLETE` 必须固定臂全部产物和统计齐全；有明确终止失败可列“队列已遍历”，但研究最终状态仍按缺失项为PARTIAL_COMPLETE。
- 任何人工视觉未核验标PENDING；速度未知留空；48h未授权不续跑。
- 报告明确实现错误修复版本、受影响重评臂和科学负结果；不把三种状态混合。
