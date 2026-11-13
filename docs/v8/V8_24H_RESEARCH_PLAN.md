# WEAVE V8：24h 全矩阵实施方案

## 1. 本次交付与执行决定

**这是实施与测试设计，不是已实现或已启动实验的声明。** 本次只新增文档，不修改模型、不启动 GPU、不覆盖历史结果。

依据：
- [24h 全矩阵研究计划](<../WEAVE_V8_24H_FULL_MATRIX_PLAN.md>)。
- [阶段总结与研究综合](<../WEAVE_RESEARCH_SYNTHESIS_20261112.md>)。

执行顺序固定为：**完成全部模块 → 单元/接口测试 → 3060 数值与完整评测冒烟 → 冻结配置与队列 → 单卡串行 H0–H6**。禁止边跑正式矩阵边补尚未实现的阶段，也禁止以“文件可编译”代替全流程验收。

设备：单卡 RTX 3060 12GB、WSL；本地工作区 `G:\WEAVE`，执行仓库 `/mnt/i/Latent_Style/WEAVE`。主质量固定标准 SD-VAE、B4、R1、原始 RGB 零点与完整 tasks750。S10 仅作独立部署子表。**（⚠ 2026-11-14：该口径已作废——一切解码一律用 S10，见 AGENTS.md 决策协议与 docs/HIGHLIGHTS.md）**

### 1.1 两份输入发生冲突时的优先级

24h 矩阵计划是更具体、较新的执行契约，优先于总结文档中的条件化训练建议：

- Teacher-A/B/C 和 D0–D6 全部预注册并执行；教师质量 NO 不取消学生可学习性实验。
- **科学结果 NO 不改变队列**；实现错误、数据污染、数值/资源错误才阻断受影响臂。
- 一个臂依赖缺失只能标 `BLOCKED_DEPENDENCY`，不得伪装成科学 NO 或“合法跳过”。其他独立臂继续。
- 24h 为单次运行累计墙钟硬上限，重启不重置。48h 只是可能的后续续跑方案，**没有用户明确授权不自动运行第二个 24h**。
- 提前完成基本矩阵时仅按预注册扩展顺序增加科学信息，不空等，不随机调参。

### 1.2 开发前必须纠正的历史解释

1. 原 Oracle 的 `+∇margin` 正确；需要最小化损失的探针应使用 `−∇loss`。不得把所有 R1 路线一起否定。
2. R3c/R4c 是测试时编辑结果；N1/N2 学习式残差失败，不能证明整个学习范式无效。
3. 同 seed 的目标无关随机扰动跨目标函数相同，是正确负对照；需要检验同源噪声共享、不同 seed 不同、task 排列不变。
4. 指标完全相同不证明文件复用。配置、分支、终点 latent、接受轨迹都要核查。
5. 本地 [旧编辑器 step 实现](<../../tools/v7r/v7r_common.py#L265-L310>) 中 `zr=z_cur.detach().clone()`，所以使用 zr 更新**本身不证明不累积**。须检查调用链是否每步传回上一终点。H0 同时保存 legacy4 与契约修正版 true4；即使累积无 bug，也保留两者对照并注明实际修改是什么。
6. 固定几何权重的平方距离只能回归加权均值；“调用 Sinkhorn”不等于多模态风格传输。
7. 随机扰动提高 DINO-S/G 都不能单独证明指定画派控制。必须同时报告目标支持、ΔG、ΔI、正确/错误目标和视觉证据。

## 2. 文档与工程布局

本目录四份实施文档：

| 文档 | 职责 |
|---|---|
| [本实施方案](<V8_24H_RESEARCH_PLAN.md>) | 总决定、工程接口、实现顺序、执行前置条件 |
| [模块实现清单](<V8_IMPLEMENTATION_SPEC.md>) | 每个模块要实现的函数、数学目标、数据与输出契约 |
| [测试与验收清单](<V8_TEST_ACCEPTANCE.md>) | 具体单测、GPU 冒烟、全量评测校验、启动门禁 |
| [矩阵与串行执行](<V8_MATRIX_AND_RUNBOOK.md>) | 全部 arm、工作量、预算、状态机、恢复和交付 |

仓库已存在 `tools/v8/` 的 Bridge/DSBM 实验代码，**本轮不复用其目录名或覆盖其实现**。建议新增：

```text
 tools/v8_24h/
   core/          common.py protocol.py registry.py artifacts.py
   objectives/    features.py bank.py mixture.py transport.py specificity.py
   editing/       legacy.py editor.py content.py interventions.py audit.py
   learning/      teacher.py models.py losses.py train.py
   reporting/     evaluate.py statistics.py diagnostics.py speed.py report.py
   tests/         test_protocol.py test_objectives.py test_editor.py
                  test_learning.py test_runner.py
 sbflow/scripts/run_v8_24h.sh
 runs/v8_24h/
   manifest.json arms.csv registry.json frozen_config.json
   audit/ energies/ geometry/ corrector/ teachers/ students/
   eval/ figures/ runtime/ state/ logs/
```

每目录约 4–5 个代码文件，单文件小于 1500 行；入口只做调度，不复制数学算法。测试文件按测试域组织，不把几百个断言塞进主模型。历史运行目录只读。

## 3. 通用接口与结果契约

### 3.1 Latent/目标契约

所有对象显式记录 `latent_space`：标准 VAE 编码产生的未缩放 posterior mean 与 WEAVE 模型使用的 scaled latent 不得混用。适配器负责唯一一次转换，并记录 scaling factor、Encoder/quant_conv/post_quant_conv/Decoder 来源与 hash。

- `SourceBatch`：task/source ID、源 latent、原始 RGB 或其路径、源画派、请求目标、原始索引。
- `StartBatch`：B4 或 Q2a 的冻结输出 z0，始终显式保存。
- `Energy.evaluate(z, context)`：返回**逐图**支持能量、区别能量、LL/HF 分量与诊断。不得提前对 batch 求均值导致线搜索耦合。
- `Editor.run(z0, source, request_target, edit_target, config)`：返回终点与逐图逐步轨迹。请求目标与实际编辑目标分离。
- `Evaluator.score(outputs, task_manifest)`：输出完整逐任务分数，不做训练反传。
- `Student.forward(...)`：返回 z0 + 硬限幅残差，明确是否包含在线风格梯度。

所有优化只在 latent 上；DINO、CLIP、CSD、LPIPS、RGB Decoder 仅作评测，不进入训练损失。R1 作为已冻结的小型引导器，允许 latent 梯度。

### 3.2 每个正式 arm 的最小产物

```text
 config.json      全部解析后参数、种子、起点、损失/约束、实际编辑目标
 provenance.json  git commit + dirty diff hash + 模型/数据/任务/参考 hash
 status.json      运行状态、科学判定、错误类别、重试次数
 latents.pt       按 task_id 对齐的 z_source/z0/z_final、空间与缩放契约
 trace.jsonl      逐图逐步损失、梯度、候选接受、约束、累计位移
 eval.json        n_total=750 / Cross600=600 / Same150=150 + 所有 rows
 features.pt      DINO/CLIP/CSD 特征、五风格分数、task_ids
 runtime.json     纯生成与评价/存储分开；batch1/batch8 的实测结果
 images/          正式输出按 task_id 保存 PNG；可脱离生成器补算 CLIP
```

不允许只留拼图或 feature cache 而不留输出图。PNG 用统一 RGB 量化口径；磁盘不足是资源问题，不允许静默关闭保存。学术主表与重新从 PNG 评分的小量化差异须记录。

教师另有训练任务 manifest，逐源原始 hash、画家 ID、跨风格四目标完整性。权重与大数组保留在运行存储；Git 提交轻量配置、结果摘要、hash、源码、文档，不提交真实训练画作。

## 4. 实现顺序：先实现完，不先跑正式实验

### 阶段 A：数据与评测底座

实现 protocol/registry/artifacts、训练池精确去重、latent 适配器、标准评测、G/I、PNG/latent 输出与任务 ID 校验。先在合成输入和已有结果上验证统计公式。

**退出条件**：全量 750 能生成/评分/落盘/续跑；CLIP 不再是占位；Same150 字段兼容有明确 schema；Oracle/B4 资产路径已实查而非猜测。

### 阶段 B：数学目标和优化器

实现多模态 bank、M0–M8 能量、逐图 OT、SWD、有界区别项、legacy 与 true editor、三内容方案和反事实。先做双峰/置换/有限差分等数学测试，不使用任务750选温度。

**退出条件**：所有 M/G/T 配置均解析成功，有 GPU 实际运行路径；错误目标和几何打乱只改变登记的一个因素。

### 阶段 C：教师与学生

实现 A/B/C 教师缓存与轨迹、142K delta/policy、500K 对照、D0 直接损失训练、硬 ClipRMS、分带监督、固定 batch 过拟合、checkpoint 全量评测接口。

**退出条件**：D0–D6 均能完成最小训练—保存—重载—评测闭环；非零 checkpoint 全 750 的任务已在队列里，而非运行结束临时添加。

### 阶段 D：调度与最终报告

实现 24h 持久时钟、GPU 串行锁、job/进程状态、最多两次修复重试、partial report、预算队列、真实测速、双前沿/3D 前沿、视觉与失败案例清单。先注入一个失败 arm 和一个超时 arm 验证不会“全部完成”。

**退出条件**：通过 [启动验收](<V8_TEST_ACCEPTANCE.md>)，生成不可变注册表；然后才允许正式启动。

## 5. 正式启动前检查

- 冻结文件清单存在且 hash 一致；训练/参考/评测重叠已处置并落盘。
- 所有固定 arm 在注册表中存在，无 UNKNOWN/TODO/占位函数/静默 fallback。
- runbook 的实验数、教师任务数、训练更新数与注册表自动计数一致。
- 在真实 GPU 上实测一个 mixture edit、一个 OT edit、一个 delta/policy 更新、一个 750 eval，算出工作量 ETA 与存储需求。
- 预检结果全部写入 preflight.json；未经验收不能以“先启动剩下以后补”放行。
- 先同步到独立工作树并校验 commit/hash，再运行串行入口；正式运行期间不原位改写正在执行的 shell 或 Python 文件。修复生成新版本与 revision，当前任务结束后受控切换。

## 6. 本轮文档交付边界

这四份文档明确了实现和测试工作，但不把此前 CPT 的 27 个局部单测当 V8 已验收。尚未实现的新项包括真实 mixture/OT、多画家无污染 bank、目标切换评测、完整教师轨迹、policy 学生、24h 不重置调度以及实际质量对应的测速。执行完成后生成单一结果主报告，不能再把旧状态快照拼到正文里产生前后矛盾。

具体实验队列与结果判定见 [矩阵和 Runbook](<V8_MATRIX_AND_RUNBOOK.md>)；具体实现见 [实现清单](<V8_IMPLEMENTATION_SPEC.md>)；放行标准见 [测试清单](<V8_TEST_ACCEPTANCE.md>)。
设计冻结前待锁定参数由训练池校准和 GPU 冒烟确定，不以正式750结果反复搜索。所有尚待锁定项必须在冻结配置中变成具体数值，否则不能启动。

## 7. 需要在实现预检中锁定的参数

包括 prototype 数量/半容量版本、温度两档、OT ε/迭代数/残差阈值、LL 权重、区别项 margin 与负梯度上限、内容代理 δ/CG 系数、两档 ρ、D0 损失配方、policy 状态编码、bf16 microbatch、存储与缓存分片。依据和锁定方式见其他三份文档；它们是明确的实现任务，不是允许运行时自由调参的空白。

不得把固定矩阵中的方法换成更容易实现的替代物：例如“组中心 Sinkhorn”不能替代“生成 Patch 集合对目标原型的逐图 OT”，零内容输入不能替代真实 canonical 内容条件，丢掉轨迹不能假称学习了优化策略。

## 8. 最终交付

轻量 Git 交付：源码、注册配置、完整 arm 状态表、统计摘要、报告、复现 hash。运行存储交付：教师缓存、权重、750 输出图/latent、全部 trace 和 feature cache。报告分别回答目标真实性、几何条件贡献、内容保护、学习能力、困难风格、实际效率；科学 NO 保留，缺失实验明确列出。

推荐结果报告路径：`docs/v8/V8_24H_RESULTS.md`。该文件只能在有实际结果时创建，不在设计阶段写假结果。

## 9. 一句话工程原则

**先把全部固定实验的真实执行路径实现并验收，再冻结排队；单卡串行执行的是完整注册矩阵，不是若干已实现的分支加一个打印 ALL_COMPLETE 的脚本。**

补充：本方案不自动创建新长期实验目标，也不接管现有 GPU 会话；用户批准执行实现或启动时，再按本实施包推进。

## 10. 实施方案验收项

读者应能从文档直接找到：每模块要写什么、每数学目标怎么校验、每个 arm 的定义、750 完整性、教师和 optimizer 更新规模、失败如何继续、何时停止、如何恢复、怎样报告不完整。缺任一项都应先补实施包，再动 GPU。

## 11. 版本记录

初版依据用户指定两份输入编制；未修改输入原文。原计划 a311bf8 是研究审计基点，不表示本地 HEAD 必须回退到该 commit；正式 run 必须记录实际实现 commit 和基点差异。

## 12. 历史隔离

旧 V6/V7/V7R/NEXT 与已有 V8 Bridge 工作都只读。可复用已核验的 Encoder/B4/R1、数据读取和评测基础，但不直接继承可疑梯度符号、跨图 Sinkhorn、错误拒绝策略、空 CLIP 字段、猜测路径及未校验的 done 标志。

## 13. 团队交接

模块可独立开发，但接口和 schema 先固定；正式 GPU 运行由一个串行队列唯一控制。交接必须同时含代码版本、测试证据、未解决问题和所需依赖，不以“开发完成”代替整个实验完成。

## 14. 决策记录

本轮明确采用 V8 全矩阵的独立科学路线，不再采用 CPT 的性能 Gate 裁剪训练。实现验收是硬门禁，研究结果不是执行门禁。这个区分必须同时体现在注册表、调度器和最终报告中。

## 15. 产物缺失处理

若某臂运行退出码为零却缺任何正式产物，标 `INCOMPLETE_ARTIFACT`，不能标 DONE。partial report 仍生成，并列出如何补齐该臂；其他独立臂按固定队列继续。

## 16. 参考关系

本目录是两份研究文档的工程细化，不另造研究方向，也不把 OT/语义匹配称为首次创新。是否形成贡献由全量同协议、目标特异性、跨画家、视觉与效率证据共同判断。

## 17. 规模说明

24h 容不下时不减少教师数、不删非零 checkpoint、不改任务集。留下精确未完成 arm 与工时，用户决定是否继续。详见 runbook 的预注册矩阵与工时计算。

## 18. 参数与文件规模

142K policy 参数计数包括全部可训练状态编码和步长/接受头；500K 对照报告实数。代码、文档和实验注册分层，不让单文件超过1500行，也不为保存某一时刻生成巨量临时 patch 脚本。

## 19. 不预期结果

0.4383/0.8125/0.4712 是联合验收门槛，0.45/0.80 是挑战目标，二者都不是性能预测。任何预检、固定 batch 测试或小样本运行不进入正式候选榜单。

## 20. 执行冻结

预检后导出注册表 checksum 与运行 config checksum，正式任务启动时再核验。若修复改变数值算法，原输出保留并登记 superseded，受影响正式臂按同750重跑；禁止把不同版本最优数字偷偷拼入同一表。

## 21. 编号空间

阶段内同名 C0/C1/C2 是内容方案编号，不与旧 CPT 的 C1–C6混用。正式 arm ID 带 V8 和族前缀，例如 V8.H3.T3.rho_lo；禁止仅用“T3”覆盖历史目录。

## 22. 当前范围

实施文档已将架构、数据、数学、测试、队列和验收拆开。执行尚未开始，实际进展以后按 registry 的产物证据报告，不用本设计里的预算时段冒充已经完成的墙钟。

## 23. 异常分类

科学 NO、实现 BLOCKED_BUG、依赖 BLOCKED_DEPENDENCY、数值 FAILED_NUMERIC、资源 FAILED_RESOURCE、预算 INCOMPLETE_BUDGET 分开。自动报告不得把失败异常写成“该方法无效”。

## 24. 视觉边界

程序只生成固定案例与审核表，不代替人工判断。若无人审核，标 visual_review=PENDING；不能以 metric 正值填写“无伪影、风格成立”。

## 25. 总结

本计划按“先全部实现并验收，再串行”落地；研究独立性、真实多模态能量、逐图约束和可审计的结果链是本轮与旧脚本的关键区别。
