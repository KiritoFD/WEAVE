# V8 固定矩阵、工时与串行 Runbook

导航：[总实施方案](<V8_24H_RESEARCH_PLAN.md>) · [实现清单](<V8_IMPLEMENTATION_SPEC.md>) · [测试验收](<V8_TEST_ACCEPTANCE.md>)。

## 1. 注册原则与精确展开

正式arm ID带族/剂量/seed/内容版本，如`V8.H1.M2a.step4.seed42`，不复用历史T/C编号目录。配置冻结后不以科学NO改变队列。每个正式arm完整750；训练teacher manifest与评测清单严格互斥。

参数尚未校准前只生成design registry；全部数值、资产hash、存储需求和GPU微基准锁定后生成不可变runtime registry。注册表自动校验ID唯一、依赖存在、必要产物、更新数、checkpoint数和评测数。

## 2. H0：方向与协议可信度（1.5h）

### 2.1 正式750队列

| ID | 起点/改变 | 数量 |
|---|---|---:|
| H0.B3 / B4 / Q2aDirect | 标准冻结预测器 | 3 |
| H0.Oracle | Q2a + 原正确R1 margin上升，历史参数 | 1 |
| H0.R1Full / R1TopK | B4 + 修复Full/逐图TopK25 | 2 |
| H0.Probe.{R1,S1,S2}.{fwd,rev,wrong} | 同B4起点、单步预算；正确`−∇loss`，错误目标仅改编辑目标 | 9 |
| H0.Legacy4 / True4 | 历史编辑器真实调用链与新契约版本 | 2 |
| H0.noC / disp / E0 / E1 | 重复分支复现，保存终点latent和trace | 4 |
| H0.Random.seed0…4 | 每源共享噪声、四目标同噪声、目标无关 | 5 |
| **H0注册总数** | 全部750，零更新复用H0.B4 | **26** |

随机历史结果只有task级共享/排列独立/图和五分数契约完整时才允许复用。已归档结果缺latent/CLIP/图，需补生成与评分，不以eval存在跳过。26臂包括基线和诊断；诊断不进正式方法前沿。

### 2.2 非750审计任务

资产hash、painter/原始索引/图像hash精确去重、finite-difference测试、每图独立OT、无可接受步保留、运行配置和分支对比、逐图真实ρ分布。ρ低/高档从审计样本实测分位数预登记；负结果不让后续方法消失，真实依赖问题标BLOCKED并修复。

## 3. H1：多模态能量（3.5h）

每个能量单步RMS0.08和4步名义0.32分别完整750，固定B4 z0、标准Decoder，4步每步0.08。

| 能量ID | 实现 | 固定变化 |
|---|---|---|
| M0 | 原E3 | 历史损失与新版editor契约明确分开 |
| M1 | 固定几何权重均值CPT | centroid基线 |
| M2a / M2b | mixture soft-min | 训练距离分位数锁定的两τ |
| M3 | 逐图集合OT | style cost随生成h变化 |
| M3s | 真集合SWD | 固定投影/分位比较 |
| M4 | M2a + bounded specificity | 独立绝对支持保留 |
| M5 | M3 + bounded specificity | 同上 |
| M6 | M2a用幅值/协方差u | 同温度校准原则，特征尺度锁定 |
| M7 | M2a + LL真实统计 | HF/LL分开记录 |
| M8 | M2a + 错误目标库 | z0/request_target不变，只改edit_target |
| **数量** | **11种能量×2剂量** | **22个750** |

M编号为11种，不能因为写M0–M8就误计9臂。H2/H3固定采用M2a作为M2默认路线，不能看M2a/b的750成绩再挑温度；M2b是独立对照。

## 4. H2：几何条件归因（3.5h）

固定M2a/M3、同bank/温度/原型数/4步预算，只改登记条件。

| ID | 能量 | 条件 |
|---|---|---|
| G0 / G1 | M2a / M3 | 无几何、均匀 |
| G2 / G3 | M2a / M3 | 正确源几何 |
| G4 / G5 | M2a / M3 | 同源patch几何顺序打乱，算子不变 |
| G6 / G7 | M2a / M3 | 固定其他源几何错配表 |
| G8 / G9 | M2a / M3 | 正确几何、错误目标库，同z0 |
| G10 / G11 | M2a / M3 | 条件几何、组/原型数固定减半 |
| G12 | M2a | 同原型数量、跨画家balanced库 |
| **数量** | | **13个750** |

画家隔离的真实训练画作能量正对照另列，不代替生成评测。比较G2−G0/G4/G8和OT对应差；所有结果无论正负均输出。

## 5. H3：方向与内容保护（4.5h）

高内容基点B4；Q2a+Oracle与B4+TopK保留为强风格参照。C0/C1/C2是内容方案，不是旧CPT同名实验。

| ID | 风格目标 | 内容方案 | 步数/名义预算 |
|---|---|---|---|
| T0 / T1 / T2 | E3 | RMS / finite-step proxy / JVP-CG | 4 / 0.32 |
| T3 / T4 / T5 | M2a | RMS / finite-step proxy / JVP-CG | 4 / 0.32 |
| T6 / T7 / T8 | M3 | RMS / finite-step proxy / JVP-CG | 4 / 0.32 |
| T9 | R1 | 与预登记M2内容臂T4逐task等终点位移对照 | 4，匹配误差记录 |
| T10 | M2a + weakR1 | finite-step proxy | 4 / 0.32 |
| T11 | M3 + weakR1 | finite-step proxy | 4 / 0.32 |
| T12 / T13 | M2a / M3 | finite-step proxy | 8×0.04=0.32 |
| T14 | M2a错目标 | finite-step proxy，同z0 | 4 / 0.32 |
| T15 | M3打乱几何 | finite-step proxy，同solver | 4 / 0.32 |
| T16 | Q2a正确Oracle | 历史参数 | 强风格基线 |
| T17 | B4修复TopK25 | 历史参数 | 强风格基线 |
| **基本数量** | | | **18个750** |

两档ρ明确展开：T0–T8、T10–T15各跑rho_lo/rho_hi，两档共15个额外750；T9用T4.rho_lo已测终点位移；T16/T17不更改历史预算。因此H3合计**33个750注册项**。T16/T17若与H0完全同配置/hash可关联复用，不重复执行但保留独立注册ID。

每步记录单项目标与E_used、gradient cosine、投影后能量、接受理由、实际路径长度/终点RMS，B1/B8一致。C2代理差也必须跑，负结果不取消。

## 6. H4：教师与全部学生（8h）

### 6.1 固定教师任务

| 教师 | 定义 | 源×其他目标 | 任务数 |
|---|---|---:|---:|
| A | 核验后的R4c历史编辑教师，精确版本登记 | 3000×4 | 12000 |
| B | M2a + finite-step内容/位移（预登记T4.rho_lo对应算法） | 3000×4 | 12000 |
| C | M3 + finite-step内容/位移（预登记T7.rho_lo对应算法） | 3000×4 | 12000 |
| **总计** | 来源均为干净训练池 | | **36000** |

A/B/C尽量共用同一个3000源manifest，风格/画家/结构平衡、绝不含750源/参考；若clean pool不足，标资源/数据问题并给缺口，不缩规模。teacher质量NO不取消其可学习性实验。

Teacher-A登记为核验历史R4c，不把true4或其他更好结果偷偷替换；若为修复实现必须登记版本差异、受影响教师全量重建。轨迹缓存含每step状态/方向/步长/accept，policy不能只读取终点。

### 6.2 训练与checkpoint

| ID | 模型/监督 | 教师 | Updates | Seed |
|---|---|---|---:|---:|
| D0 | 142K自由残差+hard ClipRMS、原R4c直接loss | 无终点监督 | 4000 | 42 |
| D1 | 142K delta | A | 4000 | 42 |
| D2 | 142K delta | B | 4000 | 42 |
| D3 | 142K delta | C | 4000 | 42 |
| D4 | 142K共享4步policy | B轨迹 | 4000 | 42 |
| D5 | 约500K delta容量对照 | B | 4000 | 42 |
| D6 | 142K共享4步policy | B轨迹 | 4000 | 43 |
| **总计** | **7个臂（输入日程中的“6个”以D0–D6表为准）** | | **28000** | |

每臂0/1000/2500/4000保存，非零3个checkpoint完整750，共**21次750**。零初始化若schema/数值精确=B4，用B4的750作明确复用；若不等价必须补评0，最多另7次。有效batch240，bf16，warmup100，microbatch调整不变总更新数和effective batch。

D0直接loss可能比缓存监督慢，policy还有在线风格梯度；不以历史37min估全部臂。固定batch过拟合/学生重载在正式启动前完成。

## 7. H5：稳健性与部署（2h）

以下在冻结registry中锁定，不等看成绩后挑最好：

- 20 routes/5风格/ER/Ukiyo-e/seen-unseen painter：从所有正式750输出完整统计。
- 六个预登记干预750：M2正确几何的独立bank seed、M3独立bank seed、M2同画派另一painter子库、M3同画派另一painter子库、固定D2 same-style不编辑、固定D4 same-style不编辑。若D2/D4资源失败，保留依赖BLOCKED而非用其他最好学生替换。
- 同源四目标/正确-错误干预案例来自完整评测，不另用200筛选。固定案例每目标至少平坦/轮廓/细节三种，所有正式arm输出；最差G/内容损伤公开选择规则。
- 纯生成batch1/batch8 staged/E2E真实测速：B3/B4/Q2a+Oracle/R1Full/TopK/legacy R3c/R4c、M2/M3、D0–D6候选对应版本。主图未知速度留空，S10为另列部署子表。
- 人工视觉表尚未看就PENDING，不用“程序读取图”替代人工审核。

上述6个新增750是将输入“六组固定诊断”具体化为可执行补充，其余统计/图/profile也必须独立登记job。不会把计算统计的6组诊断误算成6个新的模型。

## 8. H6：合并与验收（1h）

从冻结registry汇总全部候选、诊断和失败，不按已有结果选择性报告。

三套标准独立：
1. 原P4.3：相对B3 ΔDINO-S≥0.020且CI下界>0；DINO-C≥0.8125；LPIPS≤0.4712；Top1≥B3−0.02；≥4/5目标DINO-S非负；Minimalism误判不增>5pp；Same150不明显退化（具体容忍值preflight锁定）。
2. 特异性：相对B4 ΔG/ΔI CI下界>0、正确目标/几何优于同算子反事实、绝对目标支持改善。
3. 挑战：DINO-S≥0.45、DINO-C≥0.80、LPIPS≤0.4712，且特异性和视觉成立；学生保留教师收益且成本低。

实现DONE、科学NO/PASS、视觉PENDING分别列。方法假设失败不会改变固定矩阵。

## 9. 总工作量与是否能装进24h

| 工作 | 固定规模 |
|---|---:|
| H0正式750注册 | 26 |
| H1正式750注册 | 22 |
| H2正式750注册 | 13 |
| H3含两档ρ | 33 |
| H4非零checkpoint | 21 |
| H5固定补充干预 | 6 |
| **基本750注册项** | **121** |
| 教师任务 | **36000** |
| optimizer更新 | **28000** |

可复用的完全同配置项仍保留registration和引用，实际执行次数由去重cache key自动计算；预检完整B4也可复用。121是本细化方案展开后的注册量，不等于121独立架构；未包含失败重评/零checkpoint不等价补评/预注册扩展。

工时按实测：`T = Σ n_eval×eval750_sec + Σ teacher_task×teacher_task_sec + Σ updates×update_sec + 预处理/测速/统计/IO`。teacher A/B/C、delta/policy/500K分别计，不用一个平均速率乘所有。

例如eval750=120s，121次约4.03h；teacher_task=0.5s，36000约5h；仅借历史相似模型28000更新约5.76h。三者已约14.8h，尚未计D0在线loss/policy/OT/读写/统计与修复；这是**假设值敏感性分析，不是实际ETA**。teacher_task=1s时仅教师就10h，24h风险明显。预检必须实测再给预算可行性，不能承诺固定工作量必装进24h。

存储最低估算：teacher仅zs/z0/zT三fp16 latent为36000×3×4×64×64×2≈3.54GB，4步state+gradient另约9.44GB（尚不含q/c/metadata）。121×750=90750 PNG，按实测PNG平均大小估磁盘，不能拿一张grid替代逐图存档。

## 10. 24h调度状态机

### 10.1 运行与科学分离

`PENDING → RUNNING → DONE`；错误可为`BLOCKED_BUG/BLOCKED_DATA/BLOCKED_DEPENDENCY/FAILED_NUMERIC/FAILED_RESOURCE/INCOMPLETE_ARTIFACT/INCOMPLETE_BUDGET`。science另列`PASS/PARTIAL/NO/UNASSESSED`。

- 资源/代码错误每arm最多2次修复重试（初次+2次），记录revision/diff/错误/恢复位置；不能自动改科学配置。
- 依赖尚未ready的arm挂起，调度可挑下一个已ready的独立arm；GPU只允许一个活动job。
- H0代码修复不把后续合法标SKIPPED；依赖该错误实现的臂BLOCKED并修复，其他独立路线继续。
- phase时段是调度预算，不因一个阶段延迟就静默删下一阶段；固定队列与24hdeadline共同约束。阶段超预算标OVERRUN、报告影响。

### 10.2 Deadline与恢复

首次launch把`started_at/deadline/run_id`原子保存；恢复保持原deadline。保存每job耗时和已完成update/task计数，不把最后续跑段当累计。

每个teacher分片/训练checkpoint/评测batch之间检查deadline。训练/teacher可停止在安全checkpoint，评价保存task级进度；超过deadline终止本run的进程树、释放GPU锁，生成partial report和剩余队列。不得无限wait子进程把24h变成软上限。

启动新任务前根据最近吞吐与预留报告时间做预算检查，不能跑完长训练才发现没时间产报告。报告导出需另有deadline handler，计算不完至少原子保存完整状态和未完成清单。

### 10.3 串行入口职责

建议入口（待实现）：

```bash
bash sbflow/scripts/run_v8_24h.sh --preflight-only
bash sbflow/scripts/run_v8_24h.sh --freeze-registry
bash sbflow/scripts/run_v8_24h.sh --run --run-id <id>
bash sbflow/scripts/run_v8_24h.sh --resume --run-id <same-id>
bash sbflow/scripts/run_v8_24h.sh --report-only --run-id <id>
```

shell只调用Python registry/runner层，逻辑不得分散到临时补丁。进程受管理，job IDs/进程树/锁持续记录；不修改正在运行的入口文件。SSH使用`ssh 3060`，任务在WSL现有仓库，不能切Windows Python。

preflight-only不启动正式矩阵；freeze-registry必须全部模块/测试通过，缺配置数值就拒绝。report-only即使有FAILED也可运行。

## 11. 预注册扩展与48h剩余方案

基本矩阵全有结果且仍有预算，按顺序运行：
1. M2/M3独立bank seed（若H5已同定义完成，引用不重复）。
2. D1/D2/D3第二seed，各4000更新和1000/2500/4000全750。
3. 固定M2/M3从B3、Q2aDirect出发，4次750，强弱起点对照。
4. painter-disjoint目标子库，固定跨画家稳健性评测。
5. 固定场景视觉补充及profile重复。

不得看NO后另造调参组合填满预算。超过24h留下按原顺序的pending jobs、依赖、实测剩余秒、存储需求、失败修复清单；**用户授权额外24h才开新deadline**，不默认允许。

## 12. 完成判定和交付

- `ALL_COMPLETE`：所有必需arm达到约定更新/任务且完整产物验收，统计/结果链齐全；若部分失败只可说“队列已遍历”，研究状态PARTIAL_COMPLETE。
- `PARTIAL_COMPLETE`：已有可信结果、但任一必需实验或人工审核/正式测速未完成。
- `INCOMPLETE`：缺关键protocol/数据/基线，使现有结果无法进入同协议表。
- 无结果时不写假报告；有部分结果也必须输出所有注册ID的状态，不只生成成功arm表。

固定交付：manifest.json、arms.csv、config与hash、原始eval/trace/PNG/latent、teacher/ckpt、runtime、双前沿、视觉表、失败registry、剩余队列与结果主报告 `docs/v8/V8_24H_RESULTS.md`（运行后才创建，不在设计阶段链接不存在的结果文件）。
