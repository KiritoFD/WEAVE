# CLEAVE 第一阶段训练定型｜RTX 3060 · 16小时预算

STAGE 1 DEPLOYMENT STATUS (2026-11-28): Code and plan are deployed to the existing NVMe workspace I:\Latent_Style\WEAVE. No new WEAVE directory was created and no old experiment files were cleaned. Stage 1 has NOT started. Remote preflight is ready=false: P2 checkpoint, V36 cache/memmaps, offline VGG16 and ConvNeXt are missing. E4500, P1 and SD0 files exist; the P1 target is smaller than the verified source and SD0 SHA256 is unchecked. Large-file transfer timed out. Remote compilation and plan mode passed; 18 local focused tests passed. No training, profile or quality evaluation ran.

依据：[整体报告](../../OVERALL_REPORT.md)、[V37后续结论](../../paper-audit/POST_12H_CONCLUSIONS.md)、[超参数审计](HYPERPARAMETERS_AND_MUON.md)。机器可读配置：[stage1_config.json](stage1_config.json)；执行入口及文件职责：[代码说明](../../../../exp/stage1_recipe/README.md)。本文与配置为本轮执行依据，不把历史V37的E1反向配对当作本轮E1。

## 1. 要回答的三个问题

1. A任务代理latent使用目标原画Q32的训练/推理不一致，是否限制实际编辑能力？
2. 修正契约后，单独增强B画法损失，是否改善E-only或E→P的等内容前沿？
3. One-shot P独立调整LR或Q32噪声，能否以较低成本改善生成配方？

V37 E0/E1均新增8402步而未通过替换门；A/B/C局部梯度投影81.14%/18.21%/.65%；Oracle草稿S约.5179高于正式One-shot的等内容.4914；固定A3更换P参考使Top-1从43.83%降至19%。这些结果支持优先检查监督及条件使用，而不是先扩容、换优化器或换Style Encoder。参考跟随率不等价于正确画法重绘，DINO-S也不是纯笔触指标。

闭集E/P、公共Encoder、S10B与现役One-shot基线冻结。Muon、架构扩容、多原型Style Encoder、Encoder/CFG蒸馏和F4机制补强均不进入第一阶段；F5保持取消。Qwen/F7为独立任务，本队列不接管、不终止、不追加。

## 2. 完整矩阵与时间边界

| 阶段 | 内容 | 预算 | 累计上限 |
|---|---|---:|---:|
| S0 | 数据/源码契约、LPIPS入口修复、Fast-5 CI审计 | 1h | 1h |
| S1 | 冻结基线选起点、micro/checkpoint吞吐标定 | 1h | 2h |
| E0 | 原监督短程控制，目标Q32 | 90min | 3.5h |
| E1 | A改用代理Q32 | 90min | 5h |
| E2 | 代理Q32 + B画法项×1.25 | 90min | 6.5h |
| P0 | 标准LR共同起点续训 | 90min | 8h |
| P1 | 两组LR减半 | 90min | 9.5h |
| P2 | 胜出LR，Q噪声减半，从共同起点重新分叉 | 90min | 11h |
| R0 | 最有价值的一对复核/延长，或一种备用对照 | 2h | 13h |
| I0 | Seen-Dev单变量推理校准 | 1h | 14h |
| F0 | 完整test750及独立OOD、CI与报告 | 2h | 16h |

每90min臂约75min训练、15min保存和校准。S1以同组最慢臂p95耗时并留10%余量确定共同新增更新数；共同micro/checkpoint配置按稳定samples/s择优。2/3/4/5s每步对应约2250/1500/1125/900步，仅是4500秒排程除法，不是3060实测。

`--steps`表示共同新增更新快照点，达到时保存`matched.pt`；之后继续到同一训练截止，保留`equal_time.pt`。正式因素对照使用matched权重；等时间结果单列，不混为等更新实验。未到共同更新数不能进入比较。低于600更新输出`reduce_arms_required`并停止发放正式训练；**不允许因E无收益取消P阶段，也不强行完成低更新全矩阵**。需要减臂时先修订配置与新窗口，不自动缩小正式任务集。

全局16h截止和每阶段截止均写入manifest，恢复不能重置。提前完成可以提前进入后续阶段，但每阶段自身预算不超表中上限。截止导致不完整产物时标记失败/未完成，不把进程exit0或旧PID当作实验已完成。

## 3. S0：监督契约与评测修复

### 3.1 代理条件必须与实际输入一致

本轮E1/E2的A路径为：`proxy RGB → SD1.5 encoder → scaled z_proxy`，`proxy RGB → 原ConvNeXt/Stiefel → Q_proxy`。模型内部c64、亮度和源结构均从z_proxy生成；目标原画仅用于监督终点，参考图仅用于风格条件。E0保留目标Q32作历史契约控制。

所有臂固定相同作品采样与代理随机种子。缓存按seed、更新、作品ID、逻辑slot及实现/先验版本签名保存；微批切分不改变代理。第一次按需生成，后续复用，**不在第一小时重编码76,319张画作**。S0只审计一个完整A6批，输出目标Q/代理Q的RMS、余弦、空间误差及latent/c64差异与条件来源。B/C条件同样核查。

共同E起点必须是**4500步online权重**。默认指向历史V37的`checkpoints/initial4500.pt`完整初始文件；它目前不在本地，执行节点必须核验`step=4500`且包含`ecore`。不能用只含`ecore_ema`的历史快照静默替代online；也不能用V37 12902步权重替代4500步初始。P1/P2 checkpoint与V36全量训练cache也未在本地就位，不声称真实加载已验证。

### 3.2 LPIPS Bug分析报告

| 项目 | 定位与处理 |
|---|---|
| 错误现象 | F1/F6把uint8源图直接传给要求reference RGB01的`FeatureStack.lpips_vs`，Identity可得到明显非零 |
| 根因 | 生成分支内部已除255，reference分支依赖调用方规范化；两侧尺度不一致 |
| 最小修复 | F1/F6三处调用改为`reference_uint8.float()/255.0`，生成图仍按原uint8契约传入 |
| 回归验证 | CPU fake-LPIPS复现旧错误，并执行真实调用表达式确认Identity为0；真实LPIPS网络Identity检查由未来补算入口执行 |
| 证据保管 | 只把重算LPIPS及相关源图聚类CI写到新目录；不覆盖旧评分，不重新生成图片，不重算无关DINO/CSD/CLIP |

当前本地仅有旧汇总，缺F1完整spec/raw与F6完整输出。补算入口会先列出缺资产并报错，**尚未完成实际LPIPS重算**。支持外部已生成RGB任务清单`--repair-spec`；F6今后的运行会额外保留实际uint8输出，避免只有少量预览图无法补算。旧F1/F6 LPIPS继续暂停引用。

### 3.3 Fast-5严格CI复核

现存`3060_profile_summary.json`的Fast-5 s100 Same150为Δ=−1.333pp、CI[−4.000,+1.333]pp，旧字段却标PASS。严格要求CI下界≥−2pp时，应为**inconclusive，未证明非劣**，不是明确退化，也不是通过。Fast-5其余三档通过；Optimized六档通过。当前`eval_gate.py`已经按CI严格判定，冲突来自历史汇总，不能通过修改原始CI或放宽门槛消除。

S0 CPU审计已输出当前可用快照；原始数据不覆盖。重新有逐任务结果时可按同源图重算bootstrap；仅重判现存CI不会冒称新增bootstrap。现役报告已增加更正说明。

## 4. S1：共同起点与训练配置

| 设置 | Reference E | One-shot P |
|---|---|---|
| 生成核参数 | 12.67M | P45.22M + Style MLP约.933M |
| 精度 | BF16 autocast，FP32参数/优化器状态 | 同左 |
| 逻辑batch | A6/B4/C4；B有两参考分支 | 96 |
| micro候选 | 1/2/4 | 4/8/16 |
| checkpoint候选 | 开/关；图像感知路径按需重算 | 开/关 |
| 梯度clip | 1.5 | 1.0 |
| 显存目标 | ≤10GiB，需真实profile | 同左 |

P1 42k与P2 50k先在同一Seen-Dev300、固定参考/噪声的τ=0/.35/.50/.65/.80比较，依据括区内等内容风格表现选择共同起点。无重叠或证据缺失则明确失败等待处理，不能按步数直接偏爱50k。当前代码对四个C锚比较风格均值增益，平分保留42k；此规则是新设定，记录在`p_initial_selection.json`。选中的训练阶段决定T0/T1/T2/T3固定配比：P1为.30/.40/.15/.15，P2为.10/.30/.30/.30；所有续训臂相同。

为了不混淆问题，**研究增益仍相对原正式P2 50k + Reference-A3基线报告**；另保存每臂相对其共同训练起点的变化。Seen-Dev采用实例参考嵌入指标，正式test750采用目标画派库DINO-S/五类Top-1，二者不混作同一数值协议。

profile每个micro/checkpoint/臂在独立进程从同权重及随机状态起步，计入实际完整逻辑更新；只在未来显式执行，profile权重不晋升为训练结果。P采样按完整96逻辑batch同步完成，再切micro，避免异步prefetch超前的sampler/RNG债务。S1初次缓存、加载与测量也受1h预算约束；预算不足明确退出，不把配置清单当实测。

## 5. Reference E三臂

共同4500-online；新建AdamW，LR1e-4，betas(.9,.99)，wd1e-4，100新增更新warmup后恒定；双EMA .995/.999从同一online初始化。100步warmup是本轮新设定，不是历史V37配方。输出残差、源结构门控和Style Encoder不变。

| 臂 | A Q32 | A总倍率 | B画法倍率 | B颜色/内容/边缘倍率 |
|---|---|---:|---:|---:|
| E0 | 目标原画 | 1 | 1 | 1 |
| E1 | 代理源图 | 1 | 1 | 1 |
| E2 | 代理源图 | 1 | 1.25 | 1 |

E2只增强已有ranking、reference距离、Gram、Patch-SWD子项；Oklab、灰度内容、Sobel和latent幅度项保持原权重。特征损失仍可能包含色彩，必须结合灰度/同色盘输出判断。不复用V37逆向B，不生成那10,000组固定教师代理。

每250更新记录真实加权A/B/C梯度范数、投影/余弦、B画法与内容梯度余弦、灰度内容误差、残差幅度及频谱；梯度比例不以均衡为目标。每臂训练后两份EMA分别测E-only λ=.40/.60/.80/1.00与E+冻结P τ=.35/.50/.65/.80。固定探针另给正误参考、同色盘物理响应和少量RGB/灰度裁剪，响应不能自动变成画法质量通过。

## 6. One-shot P三臂

本阶段独立于E结果，必须保留。共同初始权重使用S1选中的V36 EMA；双EMA从这一共同权重初始化，新建AdamW，betas(.9,.999)，wd.01，batch96，100新增更新warmup后恒定，clip1.0。Content/style dropout=.30/.15、Y16扰动=.40；gamma/layout及Noise-to-Data FM目标沿用实际旧实现。

| 臂 | P LR | Style MLP LR | Q噪声σ | 起点 |
|---|---:|---:|---:|---|
| P0 | 1e-4 | 1.5e-4 | .05 | 共同初始 |
| P1 | 5e-5 | 7.5e-5 | .05 | 同上 |
| P2 | Seen-Dev胜出LR | 对应×1.5 | .025 | 同上，不继承P0/P1新增权重 |

P2的LR在P0/P1生成前沿筛选后写入有效配置；没有选择证据时拒绝启动。双EMA=.999/.9995。监控P/Style相对参数更新量、嵌入和速度响应；保留τ=0纯噪声生成、τ=.50/.65编辑验证。C升而S降只可能是沿曲线移动，必须用匹配内容比较判断。

## 7. R0与I0的自适应规则

`r0.mode=auto`按E1/E0、E2/E1、P1/P0、P2/对应LR控制的直接Seen-Dev等内容收益择最大一对，执行第二种子；默认从共同初始重训，每臂约3000s训练+600s校准。E/P任务间不强行比较不同数据指标。两臂本次共同更新不足600时写`insufficient_r0_budget`，不伪装已复核，保留I0/F0。

| 显式R0模式 | 执行 | 证据名称 |
|---|---|---|
| second_seed | 一对新seed，共同初始 | 独立第二种子复核 |
| extend | 同一对原matched完整状态，等新增更新延长 | 继续训练，不称独立复现 |
| backup_e | E3代理Q32、A×.75 vs E1 | 新任务权重因素 |
| backup_p | P3 Style/P LR比2.0 vs原1.5 | 新分组LR因素 |
| auto无支持收益 | 冻结契约诊断，不盲目长训 | 未获得新训练收益 |

备用只选一种。计划、pair、seed、checkpoint和截止被冻结到`r0_plan.json`，resume不重新选择。`confirmation=false`或反向退化不声称复现成功；第二种子明确否定的对应候选从后续正式筛选去除。extend/backup若有支持收益可作为新候选，但仍需F0。

I0只在Seen-Dev，对入选候选τ=.50分别比较原配置、wj×.9、wj×1.1、wc×1.1、A3 αY=.75、A3 αC=.75。E→P路线调E内部A3，P+A3路线调解析起点，其他变量不变；不做低NFE搜索。初筛按S减去相对原点的内容偏移作显式代理评分，再将胜出项扩至四档τ做等内容复查，不能仅凭单点得分采用。

若E/P均有支持收益，可先在Seen-Dev测试组合，相对更强单独路线至少两个锚点ΔS≥.005才列为互补候选；不默认拼接。默认最多一个候选组进入F0，优先保留完整任务规模。原六步积分器与闭集推理配置不变。

## 8. F0：完整验收与采用门

| 对象 | 正式工作点 |
|---|---|
| 原V36 P+A3 | τ=.35/.50/.65/.80 |
| 入选Reference E-only | λ=.40/.60/.80/1.00 |
| 入选E+冻结P、入选P+A3 | 各自τ=.35/.50/.65/.80 |
| E+新P | 仅Seen-Dev互补门通过 |

每正式Core臂必须完整test750、750唯一任务、150源图各5目标，Same150真实通过网络。Artist-OOD为1000、Movement-OOD为500完整实例参考任务，不用dev200或子集替代。基线复用必须校验checkpoint内容SHA、评测配置、种子、源/参考/错参考ID与实现来源；同名路径不构成同协议证明。

| 匹配DINO-C | 原基线S约值（历史插值） | 第一阶段研究门槛 | 第二阶段方向 |
|---|---:|---:|---:|
| .60 | .4945 | .505 | .520 |
| .65 | .4810 | .491 | .510 |
| .70 | .4680 | .478 | .495 |
| .75 | .4550 | .465 | .480 |

实际采用相对本轮**完全同协议重新确认的基线**计算，不将表中四舍五入值用于CI。只在两曲线观测括区内插值；2000次源图聚类配对bootstrap，每次重插值，至少1000个有效draw，否则标不确定，不填数。

采用需要同时满足：

- 至少两个内容锚ΔS≥.01且CI下界>0；或至少一个锚Top-1≥+5pp且CI下界>0、S非劣。
- Same150各档非劣；源图聚类CI下界≥−2pp。
- Artist/Movement-OOD各至少一个共同括区锚，S与2-way非劣；默认CI下界分别≥−.005/−2pp。这是对“无明显泛化退化”的保守操作化，不是已观测事实；无括区不自动通过。
- 灰度、同色盘和固定样例支持实际画法改善，且无新增伪影；review必须引用相应产物。`physical_response_only`探针本身不满足质量门。

默认S非劣门为CI下界≥−.005。缺视觉review时保留基线，机器不会自动批准。review模板字段`gray_style_improved`、`color_matched_improved`、`no_new_artifacts`及证据说明由后续实际看图后填写；本轮不预填true。若2h不够，减少候选，不缩减正式样本。正式输出包括逐任务JSON、配对CI、实际计数、阶段状态、报告及分数据集前沿；失败与缺失保留。

完成测量后通过统一绘图入口`--stage1-results`接入最新SVG和交互HTML，独立OOD分面，不混为原Cross600。公开文献前沿与商业产品、闭集参照和开放域候选分开。后续执行者在对话中用`html_preview`再`html_render`展示DINO-C/S与DINO-C/Top-1；本轮没有新训练点，因此不画虚构曲线。

## 9. 运行、恢复与停止

以下前两条为CPU准备；最后一条仅供未来独占3060窗口使用，**本轮未执行**。

```powershell
python -m exp.stage1_recipe.queue --config docs/system/optimization/training/stage1_config.json
python -m exp.stage1_recipe.evaluation.audit --config docs/system/optimization/training/stage1_config.json --out runs/stage1_recipe/preparation/s0_cpu_audit.json
python -m exp.stage1_recipe.queue --config docs/system/optimization/training/stage1_config.json --preflight
# Qwen/F7结束后，在已有正确环境、仓库根执行；不要现在运行
python -m exp.stage1_recipe.queue --config docs/system/optimization/training/stage1_config.json --execute
```

3060 PowerShell包装入口为 `exp/stage1_recipe/deploy_3060.ps1 -Mode Audit`；它顺序执行只读资产盘点及CPU checkpoint/cache头验证，Plan只打印预算，Execute模式调用队列自身的占卡门与锁。远端缓存与起点须经传输校验后再执行Audit。准备阶段仅运行CPU checkpoint加载，不启动训练或安装自动任务。

执行模式才查询显卡。必须为单张可见3060、无其他compute worker、显存/利用率低于配置阈值；忙则直接退出，不启动16h倒计时、不杀未知进程。拿到锁后才创建manifest；同输出目录不能重复启动。所有GPU子进程需队列显式设置`WEAVE_STAGE1_GPU_ALLOWED=1`，默认plan/help不探卡。队列设置HF离线环境，不主动下载依赖/模型；缺资产直接报错，环境沿用已验证训练环境。

恢复使用同一配置/同一命令，保留原绝对deadline。活worker存在时拒绝重复启动。崩溃遗留锁只有核验owner已死后才能用`--clear-stale-lock`清理；Windows通过read-only process handle检查，禁止`os.kill(pid,0)`。改变配置、初始权重或seed必须使用新输出根，不能冒充原run续跑。

安全退出：在输出根创建`STOP`文件；队列向当前训练run写STOP，等待完整更新后的原子保存。最后手段只停止自己拥有的child tree，绝不停止Qwen/F7等外部进程。恢复前移除本次自己创建的STOP文件，并检查checkpoint、stage状态和剩余预算。

每个完整checkpoint含online、两个EMA、AdamW、scheduler、Python/NumPy/torch CPU/CUDA RNG、sampler cursor/曝光、初始来源、config hash及deadline。训练中断在半个逻辑batch时保留最后一个完整原子checkpoint，不把半步RNG覆盖进去。queue状态同时观察child存活、step增长、heartbeat时间和日志更新，stalled不被标成正常完成。大checkpoint/逐图输出留在执行节点，不自动scp回传。

## 10. 本次实现验收与第二阶段边界

本地Stage1专项CPU回归18项通过。3060 I盘已通过Python编译与计划模式；资产预检会拒绝缺失资产。I盘目前缺P2、训练cache/memmap及离线VGG/ConvNeXt权重，完成同步并重跑 `deploy_3060.ps1 -Mode Audit` 后才具备S0起跑条件。

第一阶段之后才根据证据选择长训、损失/任务权重进一步搜索、dropout/时间采样、分组LR/beta/decay/Muon或新的风格表征。若生成前沿已有实质进步，优先扩大可靠配方的训练预算。没有前沿收益时不把训练loss、参考跟随率或梯度比例当作采用理由。
