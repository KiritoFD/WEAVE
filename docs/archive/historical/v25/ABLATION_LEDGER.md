# V25 实验结论账本（DOC-01，2026-11-20）

> **任务**：[agent2.md §八](../19_pull_together/agent2.md) A09/DOC-01 交付物之二。三栏：**有效（带 CI）/ 失败（带归因）/ 未决（带解锁条件）**。
> **证据分级口径**（承 [audits/a08_science_review.md](audits/a08_science_review.md)，落实 agent2.md A09"纠正历史过度归因"要求）：
> - 【代码证实】= 源码/逐位校验可复核的事实（file:line 或 hash 级）；
> - 【受控实验】= 单变量/配对设计支持的结论（含 A01 配对 Bootstrap CI，n_boot=1500/seed=2026/150 源聚类）；
> - 【观测性】= 相关性/几何观察，不能反向归因；
> - 【待验证】= 合理猜测，未做实验。
> 全部数值出处均为已提交文档/JSON；Round 1 点估计读数以 [04_round1_results.md](../19_pull_together/04_round1_results.md) 顶部 CI 修正块为准。

## 一、有效（带 CI / 受控支持）

| # | 结论 | 分级 | 关键数值 | 出处 |
|---|---|---|---|---|
| E1 | **Round 2 `Tr_T1`（drop_c 30%）vs `Tr_T0` @D6h 非劣且 Top-1 +1.00pp（CI [+0.0017,+0.0200] 不含零）**，守卫全过；@D5 压线 INCONCLUSIVE（dino_s 下界 −0.0051 vs 带 −0.0050）但 Top-1 +3.17pp 显著 | 【受控实验·CI】 | seed 复验方向一致（seed2 d5 56.67% vs seed1 57.17%；内容逐位一致） | [audits/a01_round1_ci.md](audits/a01_round1_ci.md) 组 5/6；[GPU4090_RESULTS.md](GPU4090_RESULTS.md) Round 2 |
| E2 | **γ=0 精确回退成立**（机制 B0 构造性回退）：学生@γ0 实测 = M0_prism1_l100 的 0.74403/0.44303/0.50571，residual=0.0 逐位 | 【受控实验】 | 一手核对 mechanism_b.json | [audits/a08_science_review.md](audits/a08_science_review.md) §9.2 |
| E3 | **D6 τ-shift 内容上限实测 0.7055**（`B_d6_t072`，τ_mean 0.762 触顶）：τ_min 0.66→0.70→0.72 的 ΔC +0.00345/+0.00132（后者低于 0.002 饱和阈值）；τ_min 路线到 0.73 正式关闭 | 【受控实验】 | D6 存在死区 τ_min∈[0.50,0.58]（纯内容代价无风格收益） | [audits/a04_strength_report.md](audits/a04_strength_report.md) §六.2/3 |
| E4 | **四条强度旋钮在 (DINO-C, DINO-S) 平面全部严格单调**，各臂均在族内前沿上（唯一异常 D6 Top-1 +0.00167 = 1 任务计数噪声）；strength 映射已交付（s=0.5 → λ≈1.218/τ_min≈0.528/γ≈0.630/τ0≈0.506，descriptive 插值） | 【受控实验·点估计】 | 19 实测臂 | [audits/a04_strength_report.md](audits/a04_strength_report.md) §六.1/8 |
| E5 | **O3_self_only 单变量：跨风格监督不可或缺**——去掉跨风格对（100% 自配对重建）Top-1 35.33%→12.33%（跌至错误标签底噪声 8–12%），内容↑风格崩；"少改即安全"先验量化成立 | 【受控实验】 | t55 同构（10.33% vs 24.00%） | [GPU3060_RESULTS.md](GPU3060_RESULTS.md) ② |
| E6 | **教师保守性归因分解**：Top-K 筛选仅贡献超额的 5.7%（rejected 均值 0.94254 精确重构——2400 候选全部高于 C1 带 [0.80,0.90] 上界），目标函数（w_c=2.2 + 22 步 Adam 预算 ≈1.2σ 位移上限）贡献 94.3%；恒等控制覆盖 O3−label_free 差距的 88%（D-S）~96%（Top-1）；corr(latent_L2, mid_cos)=−0.926（4090 导出实测） | 【受控实验】 | label_free 学生单位内容牺牲的风格收益为负（style_per_content −0.147 vs O3 +0.368） | [audits/a03_teacher_quality.md](audits/a03_teacher_quality.md) §2.2/§三 |
| E7 | **C1 两档诊断：w_c 降档不出频带**——w_c=1.2 keep 0/900（mid_cos≈0.957，gain 0.0017）、w_c=0.7 keep 0/150；**48 步反转**：mid_cos 0.9538→0.9630（加预算反而更深收敛进近恒等盆地）→"少改即安全"是**优化结构问题**（w_c/步数/阈值已定量排除；lr 未独立扫描） | 【受控实验】 | A08 裁决：支持结构重设计（init/信任域/fiber 引导三提案），但"一切超参已排除"表述过度 | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) C1 段；[audits/a08_science_review.md](audits/a08_science_review.md) §八 |
| E8 | **A05 特征几何**：795D 判别饱和于 PCA-64（5-way 92.0% ≈ full 91.8%，k=32 已 88.0%）；判别几乎全由 DINO768 块承载（单块 92.5%）；Oklab27 是互不可预测的独立低秩调色通道（R² 0.034/0.164）；u128 有效 4 维（PR=4.0，Fisher 29.73）→ u 侧无需也不可再简化；SG-5 ER 参考失位（最近质心=Minimalism，margin −0.11）；train→eval fiber 分布位移（PCA-k95 基仅解释 65.4% 方差） | 【受控实验】+【观测性】 | ER–Roc 质心最相似 +0.42 / Min–Roc 最对立 −0.71 | [audits/a05_feature_geometry.md](audits/a05_feature_geometry.md) §〇 F1–F7 |
| E9 | **A06 路线结构**：`Min→Uki` 全 5 臂垫底（Top-1 0.000–0.033，锁源 0.90–1.00）；共性失效两条轴 `*→ER` 与 `Min→*`（5/5 或 4/5 臂重合）；源锁定率与 Top-1 反向绑定（PRISM-1 lock .487/Top-1 .443 vs D6 族 .34/.55）；A01 Minimalism 守卫路线级定位：cs_j−full 的 −7.5pp = `Roc→Min` −13.3pp + `Imp→Min` −10.0pp | 【受控实验·复算逐位一致】 | 方向不对称：`Uki→Min` 1.000 / `Min→Uki` 0.000 | [audits/a06_route_report.md](audits/a06_route_report.md) §〇/§四.3 |
| E10 | **血缘链代码级事实**：V20→V22/V24/V25 的 D6/D5 solver dict 逐位一致（hash 相同）；V25 教师缓存与评测主配置 (ckpt, solver, branches, gate) 四元组对齐、唯一缺口是锚；Round 2 AB 训练锚 = z_src 干净 latent 与评测 T4 锚**确定不一致**（设计内已知）；评测种子 R1/R2 分裂（20261185 vs 20261119） | 【代码证实】 | solver hash：D6H `f52948ae3451` | [audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md) §二/§五 |
| E11 | **评测管线非 GPU 开销**：单臂 ~50s 中 CPU/I/O ≈1.69s（PNG 关）/4.92s（PNG 开），占 3.4%/9.8%（估算）；~738MB raw batch 往返是最大非 GPU 项；逐任务小文件是 Windows 反模式（67× 字节差） | 【观测性·估算】 | B04 全部为合成/标定口径 | [audits/b04_io_report.md](audits/b04_io_report.md) §8 |
| E12 | **恢复语义**：raw 层 finish 双闸（任务数/重复 id）保证部分批次不可能记为完成；eval.json 写入非原子（毫秒级 H2 窗口）；R4 型 accumulating JSON 无 merge-on-start（单臂重启覆盖丢记录）；`ResultWriter`（done 完整性闸 + JSONL 权威史）已交付 | 【代码证实】 | 原子化提案 A/B/C 未动现有代码 | [audits/b06_recovery_report.md](audits/b06_recovery_report.md) §1/§4 |
| E13 | **机制 A 判负可交付**（带两项限制）：A1 PASS≈0、A3 ≡常数门（ΔC CI [−0.0004,+0.0002]）、A2_em 能量锁定后仍 FAIL（ΔS −0.0083 [−0.0103,−0.0063]）——预注册通过线（ΔD-S≥+0.008 或风格持平下 ΔLPIPS≤−0.010）无一臂达到；与 A06 跨臂共享失效互洽 → 联合结论"两模型混合优势来自全局统计差异本身，不是空间分配"成立 | 【受控实验·CI】 | A08 附录 A 独立重算与定案表逐位一致 | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) 机制 A；[audits/a08_science_review.md](audits/a08_science_review.md) §9.1 |
| E14 | **B03 血缘检查器交付并三方对齐**：`ArtifactRegistry`（append-only JSONL，param_hash 与 A02 solver_hash / exp.v22.config_hash 同一序列化约定，三方逐位对齐；`probe_legacy` unknown-not-raise；invalidate 日志）；ref_feats 格式版本 v1/v2 如实记录（2D 无批维历史条目兼容但不静默放过）；154 测试全绿 | 【代码证实】 | 接入提案 3 项（教师缓存最高优先）；steps/lr 须入 params_hash（F8 事故） | [../../src/v25/artifact_registry.py](../../src/v25/artifact_registry.py) docstring；[../19_pull_together/PROGRESS.md](../19_pull_together/PROGRESS.md) L75 |
| E15 | **Round 3 τ 曲线单调可行**：O3 单 strength（τ0）在 [0.40,0.60] 单调权衡无异常段（t40 0.5457/0.5304 → t60 0.7274/0.4883），`PRISMRef.transfer(strength)` 滑块接口可行（但曲线带 wj 混杂，非纯 τ0） | 【受控实验·点估计】 | 同权重零训练补扫 | [GPU3060_RESULTS.md](GPU3060_RESULTS.md) ①；[audits/a04_strength_report.md](audits/a04_strength_report.md) §六 rejected |

## 二、失败（带归因）

| # | 结论 | 分级 | 归因 | 出处 |
|---|---|---|---|---|
| F1 | **z_src 锚判负**：Top-1 7.17%、DINO-S 0.4374、LPIPS 0.2245 ≈ 输出≈源图（近恒等退化） | 【受控实验】 | 源图锚使条件语义反转——锚提供的是"风格目标"而非"内容起点" | [04_round1_results.md](../19_pull_together/04_round1_results.md) §三 P1-3 |
| F2 | **机制 A 空间自适应门控家族判负**（定案，按铁律不重训第三种门控） | 【受控实验】 | 两模型增量 Δz 的空间重分配在该预算（1000 upd）与水位（γ=0.55, D-C≈0.736）下无增量；学习门控在平滑约束下退化为常数门 | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) 机制 A 段 |
| F3 | **机制 B0 判负——修正中状态（A08：不可交付为最终科学结论）**。γ≥0.25 风格随 γ 下降（0.4352→0.3872）、残差 0.231(g25)→0.632(g70) | 【受控实验→受损】 | 三条主证全部受损：①Same150 96.7%→64.7% 崩塌 = measure 路径缺 same-task bypass 的**伪迹**（mechanism_b.py:303 bypass 了 gw 副本但 :382 传未 bypass 张量）；②残差混入 B4(训练)/T4(评测) 教师 lineage 差；③2000 upd 欠拟合（lat 0.014 未收敛）相容。成立部分仅 γ=0 回退（E2） | [audits/a08_science_review.md](audits/a08_science_review.md) §二/§9.2 |
| F4 | **"删 Hybrid、单模型 D6 替代"判负（当前口径）**：04 文档 P0-1 "B≳C" 初步读法在 LPIPS≤0.51 约束下不成立（D6 全族 LPIPS 0.518–0.537 违约）；D6 内容上限 0.7055 < 0.73；C-Top1 前沿由 Hybrid 主导（g25–g85） | 【受控实验·点估计】+预注册 CI 缺口 | 双模型必要性的**正确证据**是 D6 τ-shift 实测饱和（E3），不是 B0 失败 | [audits/a04_strength_report.md](audits/a04_strength_report.md) §六.5/6；[GPU4090_RESULTS.md](GPU4090_RESULTS.md) 问题 2 |
| F5 | **"B4+canonicalizer 锚 = T4 等价物"断言被推翻**：B4↔T4 闸门实测 distinct-scale（divergence_ratio 1.21 > 0.60；task_l2_p50 28.0 ≈ T4-vs-src 28.7）；该配方的原始出处是 V19 错标签反事实臂的锚 | 【代码证实】+【受控实验】 | 书面断言 vs 代码事实（A02 表 (b) 预警兑现）；机制 A/B 教师读数与评测读数跨 lineage | [audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md) §二 (b)；[GPU4090_RESULTS.md](GPU4090_RESULTS.md)（闸门结果） |
| F6 | **C1 两档 w_c 教师生产失败**（keep=0） | 【受控实验】 | 非 w_c 单变量可解：损失盆地极小值本身即近恒等解（48 步反转，E7）；结构重设计三项（init=ref_anchor / trust-region / color-guide）已实现为 [../../exp/v25/reference_teacher_v2.py](../../exp/v25/reference_teacher_v2.py) 开关 | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) C1；PROGRESS L117-121 |
| F7 | **V24 M3/M4/M5/M6 伪锚教师链禁用**：avg_pool 伪 z0 锚、z_p1 充当 D6 锚的双重伪造链 | 【代码证实】 | precompute 已显式用真 z0_tr 取代；其衍生缓存禁止再作教师 | [audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md) 表 (c) |
| F8 | **缓存血缘事故**：`ref_teacher_pairs_v2_texpr.pt` 按"档名"复用命中 24 步旧包（steps 未纳入缓存键） | 【代码证实】 | B03 registry 接入时须把 steps/lr 纳入 params_hash | [../19_pull_together/PROGRESS.md](../19_pull_together/PROGRESS.md) L121 |
| F9 | **Round 1 初步读法的两项修正**（点估计→配对 CI）：①P1-4 "fixed 门控可删"被 **FAIL** 否定（Δdino_c CI [−0.0187,−0.0113] 整体越带下界，Ukiyo-e −10pp）→ 密度门控保留；②P0-2 "cs_j 删 full"两个水位均 **INCONCLUSIVE**（Minimalism −7.50pp@D6h / −21.67pp@D5 触守卫）→ 默认保留 full | 【受控实验·CI】 | 04 文档初步读法只比较了插值风格指标，未配对检验内容指标 | [audits/a01_round1_ci.md](audits/a01_round1_ci.md) §二/§五 |

## 三、未决（带解锁条件）

| # | 未决项 | 当前状态 | 解锁条件 |
|---|---|---|---|
| U1 | **D6 vs Hybrid @ DINO-C≈0.73 配对判定** | UNDECIDED；τ_min 补臂路径已死（E3 饱和），B0 教师曲线载体失败 | 余下杠杆：改 D6 其它参数（tau_hi 上调/分支结构）补实测臂后用 A01 工具出 CI；或接受 PRISM-1/Hybrid 对 0.73–0.75 的既有实测覆盖并正式放弃该判定（A08 §四 R5 建议总控更新指引） | 
| U2 | **cs_j 是否删 full（20 vs 40 NFE）** | INCONCLUSIVE（F9②守卫触发） | 若坚持删：按 A06 建议对 `Roc→Min`/`Imp→Min` 路线单独设守卫阈值（而非最差类均值）；或直接保留 full |
| U3 | **fixed 常数门重标定（`P14b_d6h_fixed_g69`）是否可替密度门控** | 对齐后追平并略优（0.6838/0.5122/0.5293/53.83% vs original 0.6782/0.5121/0.5350/55.00%） | 对该真实配对跑 A01 配对 CI（工具/协议已就绪，未执行） |
| U4 | **PRISM-1 锚替代 T4 锚** | INCONCLUSIVE（三指标骑带边 + τ 未对齐 + Ukiyo_e −18.33pp）；对齐臂 P13b 仍略逊 −0.005~−0.007 | 无急迫解锁——维持 T4 默认、PRISM-1 锚仅作免 T4 快速档；如需重判须 τ 对齐 + 配对 CI |
| U5 | **`V25_Tr_T3 vs Tr_T0` 配对 CI 未算** | runs/v25/measured/ 缺 Tr_T3 评测产物（Tr_T2 可随时补算） | GPU 产出后 `compare_arms` 直接补算 |
| U6 | **机制 A2 归因在锚对齐前不得引用**；机制 B0 主证待修复重评 | R12 补救脚本已就绪（零训练） | [../../exp/v25/r12_anchor_reeval.py](../../exp/v25/r12_anchor_reeval.py) 在 4090 跑 R1（B4 锚 ctx750 重评 7 臂）+ R2（bypass 修复 γ 扫描）→ a01 配对 CI |
| U7 | **O3_label_free ΔS+0.157 的参考归因**（"颜色级迁移尚存" vs "参考无关通用漂移"） | INCONCLUSIVE（A08 §七：swap 0.131 + ΔS +0.157 联合更支持漂移解释） | SG-25 参考消融控制（drop_style=1 或参考换池均值 fiber，2×125 对，3060 分钟级） |
| U8 | **C1 lr 替代解释未排除**（唯一残留超参） | A03 显式冻结 lr=0.055 | lr 探针：w_c=0.7、steps=48、lr∈{0.02,0.11}、150 对（~25 min 4090，可并入结构重设计首批） |
| U9 | **跨 Round seed-unmatched 配对 CI**（20261185 vs 20261119） | 决策级比较禁止直接混算 | 同种子重跑一侧或显式标注 seed-unmatched（a02 约束 4） |
| U10 | **O3 臂路线矩阵缺失**（One-Shot 路线级失效模式未知） | 无逐任务记录，A06 如实降级为聚合口径（Top-1 0.24） | 4090 侧为 O3 臂补 measure 逐任务产物（A06 工具一行式扩展已预留） |
| U11 | **128D 之外：795D vs ConvNeXt v16 冗余判定** | BLOCKED（convnext_tiny 权重本地不存在）；行为侧旁证 O1→O2→O3 各有增量 | `--export-cnx` 导出后跑 CCA/ridge/logistic 对照（concat 增益 <1pp ⇒ 可简化） | 
| U12 | **B05 双接口端到端冒烟** | 引擎注册表/强度映射/配置 hash 已交付并钉住；transfer 执行路径未跑（未跑项在案） | 最终单样本双接口冒烟 + checkpoint manifest 补登（见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §六） |

## 四、历史过度归因纠正表（A08 口径逐条）

| 历史表述（出处） | A08 裁定 | 现行正确口径 |
|---|---|---|
| "Same150 崩塌为主证"（机制 B0 归因②，PROGRESS L93 问题表） | **伪迹**——measure 路径缺 bypass（mechanism_b.py:303 vs :382），96.7%→64.7% 是评测 artifact 不是模型失效 | 撤回该主证；B0 判负降级为"本窗口判负，带已记录重试协议"（R2 修复后重评才有意义） |
| "学生不跟踪教师混合曲线"（B0 主证） | **混杂**——残差把学生拟合误差与 B4/T4 教师 lineage 差混为一路；完美跟踪训练教师的学生在该协议下也会随 γ 长出残差 | 不可作为 B0 主证；须 R12 加记 B4 锚教师曲线残差后归因 |
| "双模型必要"（证据=B0 失败） | **证据换源**——B0 失败不作必要性证据 | 引用 D6 τ-shift 内容上限 0.7055（实测饱和，E3）+ LPIPS 包络口径（F4） |
| "机制 A 判负"（名义 A2 读数） | **部分伪影**——名义 A2 的互换 ~85–90% 是 λ_c 标定残差（Δ=+0.037）→ 有效能量漂移（0.55→0.456）的强度效应 | 判负可交付，但引用 A2 必须用 em 臂；"全局统计差异"归因按 A1+A3+em+A06 联合证据引用 |
| "C1 不是超参问题（w_c、步数均已排除）" | **表述准确但勿泛化**——lr 未独立扫描；预算反转逻辑不覆盖"更大 lr 逃逸局部盆地"版本 | "w_c/步数/阈值已定量排除 + 48 步反转指向损失盆地形态；lr 未测"（U8） |
| "颜色级迁移尚存"（O3_label_free SG-25 ΔS+0.157，GPU3060_RESULTS L38） | **INCONCLUSIVE**——swap≈0.13 时 ΔS 可由参考无关漂移产出 | "参考响应弱"（0.131 vs 0.442 同协议）可交付；参考归因待 U7 消融 |
| "fixed 门控可删 / cs_j 可删 full"（04 文档初步读法） | **CI+守卫否定**（F9） | 密度门控保留（FAIL）；full 默认保留（INCONCLUSIVE） |
| "教师筛选偏向近恒等造成 label_free 风格差"（A03 任务原命题，若"筛选"指 Top-K） | **被拒绝**——筛选份额仅 5.7% | 若指教师管线整体保守（目标+预算+筛选）则被证实（≥88–96%，边际为负）（E6） |

## 五、Round 1 四判定点·最终态对照

| 判定点 | 04 文档初步读法（点估计） | 配对 CI + 守卫最终态 | 现行决策 |
|---|---|---|---|
| P0-1 单模型 vs Hybrid | B 同水位风格反超 C（+0.008~+0.013 插值） | 未出判定 CI；LPIPS 包络反转（A04 §六.5）；0.73 水位 D6 不可达（E3） | Hybrid 保留默认；单模型 D6 为压缩候选（U1） |
| P0-2 CFG 分支 | cs_j ≈ full，可删 full | #1/#3 均 INCONCLUSIVE（Minimalism 守卫） | 保留 full（U2） |
| P1-3 锚点 | PRISM-1 锚 ≈ T4；z_src 判负 | #4 INCONCLUSIVE；z_src 判负维持（F1） | T4 默认；PRISM-1 锚=快速档 |
| P1-4 Gate | fixed_g=0.5 带内，密度门控可删 | #2 **FAIL**（ΔC CI 整体越带） | 密度门控保留；重标定臂待 CI（U3） |

> 复算入口：`python -m tools.v25_cpu.a01_noninferiority`（全部 CI）、`python -m tools.v25_cpu.a04_strength_analysis`、`python -m tools.v25_cpu.a06_route_diagnostics`（命令详见各 audit 报告 §复算）。谱系侧档案见 [MODEL_LINEAGE.md](MODEL_LINEAGE.md)。
