# A08：机制 A/B/C 结果独立科学审查（2026-11-20）

> **TASK**: A08（Slot A，第四批，REVIEW-SCIENCE 角色，[agent2.md §九](../../19_pull_together/agent2.md)）｜**STATUS**: DONE
> 方法：只读既有材料 + CPU 一手复算（无 git/CUDA/ssh，未改任何既有文件）。除文档核对外，本审查**独立重跑了**以下验证：①`tests/v25/test_stats.py` 16 passed（一手）；②`tests/v25/test_mechanism_c.py -k "label_free or inert"` 12 passed（一手）；③用 `tools/v25_cpu/a01_noninferiority.compare_arms` 对机制 A 四臂 vs A0 **独立重算配对 CI**（600 任务 / 150 源聚类，数值与 GPU4090_RESULTS 定案表逐位一致，见附录 A）。
> 三态口径：PASS = 该威胁**不推翻**对应结论（结论可保留）；FAIL = 该威胁**实质损害**对应结论；INCONCLUSIVE = 现有证据不足以裁决。

## 〇、结论速览

| # | 审查项 | 裁决 | 一句话依据 |
|---|---|---|---|
| 1 | 机制 A 混杂变量（λ_c 标定残差 Δ=+0.037） | **PASS** | 名义 A2 读数 ≈90% 可由有效能量漂移（E_mean 0.456 vs 0.55）解释，但预注册能量匹配对照 A2_em（E≡0.55 构造性锁定）仍 FAIL，且内容校正后仅平价（−0.0004）——判负稳健，归因带残留限制 |
| 2 | 训练/评测教师一致性 | **FAIL** | 训练侧 z_d6_tr 用 **B4 锚**（precompute_train_latents.py:78-83,117-119），评测侧 ctx750 再生用 **T4 锚**（mechanism_a.py:548 → exp/v22/t4.py:86-117）；`assert_same_teacher` 只比对 spec dict（anchor 是自由文本），不绑锚张量；与 PROGRESS.md L64"评测侧用 B4 锚再生"的声明相矛盾。B4↔T4 已实测 distinct-scale（ratio 1.21） |
| 3 | Bootstrap 工具正确性 | **PASS** | 重采样索引矩阵逐位相等（test_stats.py:305-306 ↔ metrics.py:116）+ CI 与 `bootstrap_source_ci(draws=None)`/`paired_cluster_ci` 双向 1e-12 等价（test_stats.py:307-323）；方向/三态/配对守卫/真实数据锚定共 16 测一手复跑通过 |
| 4 | 外推当实测 | **PASS** | 04 文档所有插值/外推均带"初步读法/≈/外推需 CI 验证"标注且被顶部 CI 修正块覆盖；A04 全部插值标 descriptive only、外推标 UNMEASURED；D6@0.73 各文档一致记 UNDECIDED（但两处补救指引已过时，见 §4） |
| 5 | 内容水位对齐（0.7361–0.7466） | **PASS** | 错位方向**利好** A2（内容盈余部分掩盖风格亏损）；用本工作点自身前沿斜率（0.99/0.72，A04 §二.3）校正后 A2_em @A0 内容水位 ≈ 0.4740 vs A0 0.4744（−0.0004）——至多平价，距通过线 +0.008 差 ≥0.0084，不翻转 |
| 6 | One-Shot 无类别输入 | **PASS** | `oh = torch.zeros(...)` 无条件恒零（p0_o1_o3_value.py:102），`style_id` 形参在 forward 体内零使用——对任意权重结构性惰性；推理传 sid0≡0（r3_oneshot_minimal.py:129,229,285）；B02 inspect 断言（mechanism_c.py:138-152）+ legacy 槽位 torch.equal 实证 12 测一手通过 |
| 7 | 视觉结论与指标自洽（swap 0.131 vs ΔS +0.157） | **INCONCLUSIVE** | "参考响应弱"成立（同 τ/wj 协议 0.131 vs 0.442）；但 ΔS +0.157 与 swap 0.131 联合更支持"**参考无关的通用漂移**"而非"颜色级迁移尚存"——现有数据无法区分，缺参考消融控制 |
| 8 | C1 结论链条（结构问题 vs 超参） | **PASS** | 阈值替代解释已被 A03 定量排除（2400 候选全部高于带上界，筛选份额 5.7%）；步数/预算被 48 步反转排除（步数↑ → mid_cos 反常↑）；w_c 两档排除；唯 lr 未独立扫描（A03 §五显式冻结 lr=0.055）——"w_c、步数均已排除"的原文表述准确，泛化为"一切超参"略过 |

---

## 一、混杂变量：λ_c=0.3 标定残差是否驱动 A2 的"同前沿互换"

**结论：混杂存在且已定量分解；预注册 FAIL 裁决对它不敏感。**

事实链（一手复算，附录 A）：

- 标定目标 = A0 全局混合同 γ 的探针 mid-cos（`mechanism_a.py:497-529`，`calibrate_lambda_c` 在 64 个训练对上取 `candidates=(0.3,0.6,1.0)` 中最近者）。λ_c=0.3 探针 Δ=+0.037（PROGRESS.md L78"候选网格偏高但可接受"）——即 A2 的训练目标系统性偏保守（更保内容）。
- **该残差的能量级体现**：`runs/v25/mechanism_a/mechanism_a.json` 逐臂 `edit_energy_mean`——A0=0.55（≡γ），**A2_learned=0.45571**（E_p50=0.55016），A2_em=0.55（`energy_matched_gamma` 构造性锁定，`mechanism_a.py:185-194`），A3=0.54989（常数门）。
- **定量分解**（用 A04 §二.3 Hybrid γ 曲线在本工作点 C∈[0.732,0.747] 的实测局部斜率 dC/dγ=−0.0973、dS/dγ=+0.0963）：A2 有效能量 0.55→0.456（Δγ=−0.094）预测 (ΔC,ΔS)=(+0.0092,−0.0091)；实测 (+0.0105,−0.0111)。**名义 A2 的互换 ≈85–90% 是强度漂移伪影**（标定残差 → 门控学出欠编辑 → 有效 γ 下移），残余 (+0.0013,−0.0020) 才是空间再分配成分。
- **决定性对照**：A2_em（能量逐任务精确匹配 0.55，消除全部强度成分）独立重算 CI：ΔC +0.0080 [+0.0047,+0.0114]、ΔS −0.0083 [−0.0103,−0.0063]、LPIPS +0.0036——**仍 FAIL**（预注册通过线 ΔD-S≥+0.008 或风格持平下 ΔLPIPS≤−0.010，GPU4090_RESULTS.md 机制 A 定案段；两条件均远未达）。互换方向点上前沿（ΔS/ΔC=1.04 ≈ 局部前沿斜率 0.99）——是"沿前沿互换"而非"离开前沿的退化"。
- 佐证：A1_heuristic（**不经过任何标定**、输入不含 dz）PASS≈0；A3_smooth 同管线+平滑约束退化为常数门（ΔC CI [−0.0004,+0.0002]，一手重算逐位一致）——门控可学的空间信号本身 ≈0，与标定无关。

**残留限制（不影响裁决，影响归因表述）**：em 臂消除总能量混杂但不改变 M 的空间形态——若 λ_c 残差把"高内容区少编辑"烙进 M，A2_em 的互换仍可能部分是标定塑形的空间保守性。"该门控家族在完美标定下能否达到 +0.008 增益"未经训练无法回答；但 A1/A3 的零信号 + em 平价（§五）使"存在未被发现的帕累托增益"先验极低。**最终表述建议**：机制 A 判负可交付，"两模型优势来自全局统计差异本身"的归因按"A1/A3/em 联合证据"引用，避免把名义 A2 的互换当作空间效应引用。

## 二、训练与评测教师一致性 —— **FAIL（本轮审查最重要的发现）**

**代码即记录（repo 内唯一可审计证据）显示训练/评测锚不一致：**

| 侧 | 锚 | 证据 |
|---|---|---|
| 训练侧 z_d6_tr | **B4**（B4_distill50 + canonicalizer(c1_gentle)） | `exp/v25/precompute_train_latents.py:33`（B4_CKPT）、`:75-83`（z_anc_tr = b4(canon(z0,c1_gentle),…)）、`:117-119`（run_core_flow 第 6 参 = z_anc_tr） |
| 评测侧 z_d6_750 | **T4**（AnisotropicS10Teacher 2 步优化） | `exp/v25/mechanism_a.py:548`（`build_or_load_t4_latents_750`）→ `:553-556`（run_core_flow 第 6 参 = z_t4）；`exp/v22/t4.py:11-24,45-83,86-117`（T4_CONFIG，全文件无 B4 路径）；机制 B 复用同一构建（`mechanism_b.py:47-53,359`） |

三个加重因素：

1. **断言锁不住锚**：`assert_same_teacher`（`mechanism_a.py:278-287`）只比对 TeacherSpec **dict**；`anchor` 字段是自由文本 `"B4_distill50 + canonicalizer(c1_gentle) [T4-equivalent]"`（`precompute_train_latents.py:37`），评测侧从不按 spec 构建锚——断言对"实际传入 run_core_flow 的锚张量"零约束，评测侧校验是恒真。
2. **与书面声明矛盾**：PROGRESS.md L64（B4↔T4 闸门 distinct-scale 后）声明"评测侧按 TeacherSpec **用 B4 锚再生 z_d6_750**"——代码未实现；A02 报告约束 2 明确禁止"以 teacher_spec 写了 T4-equivalent 为由跳过"，闸门判 distinct-scale（ratio 1.21，task_l2_p50 28.0 ≈ T4-vs-src 的 28.7，PROGRESS.md L64）后既未换 T4 型锚重造教师、也未在评测侧补 B4 锚臂——恰是 A02 §五 RISKS 预警的失配场景。
3. **量级非平凡**：闸门实测 B4 与 T4 锚互差 ≈ 锚-源距离同量级 → dz_750^T4 与 dz_tr^B4 分布差异实质性存在。

**影响分解**：

- **机制 A**：A0/A1/A2/A3 共享同一 ctx750 → 臂间配对比较内部自洽；但 **A2/A3 的门控在 eval 时消费 OOD 输入**（训练见 B4-Δz，评测见 T4-Δz），其 FAIL 读数含未量化的 OOD 惩罚。判负本身靠 A1（dz 无关）/A3（塌缩）/A2_em 斜率算术支撑（§一/§五），**不翻转**；但"A2 归因"在锚对齐前不能引用。
- **机制 B0（更严重）**：学生训练目标是 B4 锚教师混合 z*(γ)（`mechanism_b.py:16,233`，z_d6_tr 来自缓存），评测残差却对 **T4 锚教师曲线**计算（`mechanism_b.py:359,304`）——`runs/v25/mechanism_b/mechanism_b.json` 的 res_mean 0.231(g25)→0.632(g70) **把学生拟合误差与教师 lineage 差异混为一路**；一个完美跟踪训练教师的学生在该协议下也会随 γ 长出残差。B0"不跟踪教师"的主定量证据被混杂。
- **附带发现（B0 第三条主证是伪迹）**：`eval_student_gamma_sweep` 在 `mechanism_b.py:303` 对 **gw 副本**施加 same bypass，但 `main` 在 `:382` 把**未 bypass 的** `z_by_gamma` 传给 `eval_latent_v25` → Same150 在 γ>0 时测的是学生原始输出（残差网络在 same 任务上非零），96.7%→64.7% 的崩塌是 **measure 路径缺 bypass 的 artifact**，不是模型失效。PROGRESS.md L93 已把它列为归因②"协议缺口"，但 L59 问题表仍引用"Same150 崩塌为主证"——后者应撤回。

**裁决**：FAIL。"训练与评测教师一致"当前不成立（按 repo 代码记录），且该断言结构（自由文本锚 + dict 比对）无法防止复发。补救 R1/R2 见 §十（均为零训练）。

## 三、Bootstrap 工具与统计基础等价性 —— PASS

- **RNG 逐位一致**：`shared_bootstrap_draws(n)`（a01_noninferiority.py:162）= `np.random.default_rng(2026).integers(0, n, size=(1500, n))`，与 `bootstrap_source_ci` 内部构造（`src/eval/metrics.py:115-116`）逐位相等——`test_stats.py:305-306` 以 `np.array_equal` 断言（bitwise，非 approx）。
- **CI 双向等价**：`test_stats.py:307-323` 断言 compare_arms 的 CI 同时等于 `bootstrap_source_ci(draws=None, seed=2026)` 与 `paired_cluster_ci`（metrics.py:169-188，配对差 + 源聚类），容差 abs=1e-12（float64 尾数级）。跨指标共享重采样索引的性质与配对差逐 replicate 联合分布保持（a01 报告 §一声明）成立：paired_cluster_ci 逐指标同 seed 同形状 → 同 draws 矩阵。
- **语义守卫一手复跑通过**（16 passed）：带边界含等号 PASS（:104-114）、三指标 FAIL 判据（:116-132）、LPIPS 低优方向（:146-162）、重复/缺失/NaN/聚类完整性 ValueError（:188-231）、worst-class 与 same-blind 守卫降级 PASS→INCONCLUSIVE（:326-353）、真实 Round-1 数据锚定（:356-364，Δdino_c mean≈−0.0038）。
- **已披露的参数差异**（非缺陷）：包装层 N_BOOT=1500/SEED=2026（a01_noninferiority.py:51-52）vs 基础层默认 4000/20261118——a01 报告 §一已声明，等价性测试在包装层参数下验证，自洽。
- **A08 独立复算交叉验证**：用该工具重算机制 A 四臂 CI（附录 A）与 GPU4090_RESULTS 定案表逐位一致——工具链端到端可信。

## 四、外推当实测检查 —— PASS（附两处过时指引记录）

逐处核对：

- **04_round1_results.md**：P0-1 插值表全部用"≈/外推"标注并注"外推需 CI 验证"（§三表格内）；顶部 CI 修正块 ⑤ 明确"下文各'初步读法'均为点估计口径，以本修正为准"；D6 vs Hybrid @0.73 记"未决"（修正块 ④）。
- **a04_strength_report.md**：全部插值臂标 **descriptive only**、映射表外推区标 **UNMEASURED**（§三、§七.5"面板不呈现任何插值臂读数点"）；§六.5 明确纠正 04 文档"B≳C 可删 Hybrid"读法在 LPIPS 约束下不成立。
- **D6@0.73 未决项处理一致性**：04 修正块 ④、a01 §四.1、GPU4090_RESULTS.md L10、a04 §六 rejected/inconclusive 四处口径一致（UNDECIDED，不计算不外推）；`B_d6_t072` 实测 0.7055 后 τ_min 路径正式关闭（A04 §六.2，PROGRESS.md L55"无法到达 0.73"）——没有任何文档把 0.73 读数当实测。

**记录项（不判 FAIL）**：两处**补救指引已过时**——04 修正块 ④"需 4090 补 τ_min≈0.72 真实臂后由 A01 出 CI"与 GPU4090_RESULTS.md L38"将在机制 B 教师曲线实测覆盖 0.73 后出 CI"：前者已被 A04 实测关闭（0.72 → 0.7055 饱和），后者因 B0 判负而失去载体。UNDECIDED 状态本身正确，但该未决项当前**没有任何活着的最短解决路径**（余下杠杆：改 D6 其它参数/分支结构补臂，或接受 PRISM-1/Hybrid 对 0.73–0.75 的既有实测覆盖并放弃该判定）。建议总控更新指引（本审查不改既有文件）。

## 五、内容水位对齐对"A2 无帕累托增益"的影响 —— PASS

水位（mechanism_a.json）：A0 0.73614 / A1 0.73786 / A2 0.74662 / A2_em 0.74415 / A3 0.73606。

- **错位方向**：A2/A2_em 内容**高于** A0——沿前沿这对应风格**亏损**，即原始对比**低估** A2 在同内容下的风格。错位对 A2 有利，若校正后仍无增益，结论只强不弱。
- **幅度**（本工作点实测局部斜率 0.99，A04 §二.3 g40→g55）：把 A2_em 拉回 A0 内容水位 0.73614 需退 ΔC −0.0080 → 收回风格 +0.0079 → 校正后风格 ≈ 0.46610+0.0079 = **0.4740 vs A0 0.47442（−0.0004，平价）**。预注册通过线要求同内容 ΔD-S≥+0.008——距通过线差 ≥0.0084。
- 名义 A2 的水位错位主要由 §一的能量漂移解释（Δγ=−0.094 预测 ΔC +0.0092，已解释实测 +0.0105 的 ~88%）——水位错位与标定残差是同一现象的两面。
- **裁决**：错位方向利好 A2、校正幅度 ≤0.008 且校正后仅平价——"A2 无帕累托增益"稳健。附带说明：A2 的 LPIPS 实为显著改善（−0.0073，CI [−0.0085,−0.0061]，一手重算），但未达预注册 −0.010 线且风格/守卫双失——判 FAIL 正确，报告引用时宜完整陈述（避免"全面更差"的过度表述）。

## 六、One-Shot 无类别输入 —— PASS

- **结构性惰性**：`SLFUNetV24OneShot.forward`（`exp/v24/p0_o1_o3_value.py:97-103`）接受 `style_id` 形参但函数体内**零使用**；`oh = torch.zeros(B, self.N_STYLE, device=dev)`（:102）无条件恒零后进 cond_mlp（:103）——与权重无关，任何 checkpoint 下 sid 均无效应。
- **训练侧**：`train_o_arm` 虽把 `sid_all[ir]` 传入 forward（p0_o1_o3_value.py:252,289），但该参数是死参；风格条件全部来自参考侧无标签量（`u_style = adapter(u_all[ir], u_all[ic])`、`u_glob = fib_n_all[ir]`，:255-257）。
- **推理侧**：`sid0 = torch.zeros(B, dtype=torch.long)`（`exp/v25/r3_oneshot_minimal.py:129,229,285`）。
- **机械断言**：`assert_label_free_interface` 用 inspect 扫 9 个机制 C 接口签名禁 sid/tid/onehot/class 等 9 类参数名（`exp/v25/mechanism_c.py:138-152`）；legacy 槽位 torch.equal 实证（`tests/v25/test_mechanism_c.py:320-337`，sid=0 vs sid=3 输出逐位相同）。一手复跑 `pytest -k "label_free or inert"` → **12 passed**。
- 边界说明：参考特征（fiber/ConvNeXt v16/DINO）当然携带风格**信息**——审查命题是"无类别 **ID** 输入"，成立；不得引申为"无风格信息"。

## 七、"参考响应弱"（swap 0.131）与 SG-25 ΔS 的联合解读 —— INCONCLUSIVE

- **成立的部分**：swap_dist 协议（SG-5 五参考 × 10 内容、τ=0.45/wj=2.2 固定、与 O3 既有 0.442 同口径可比，`r3_oneshot_minimal.py:202` 注释与 `:203-256` 实现）下 label_free **0.1311** vs O3 **0.442**——"对参考更换的响应只有对照的 ~30%"是同协议实测，"参考响应弱"成立。
- **不自洽的部分**：GPU3060_RESULTS.md L38 把 ΔS +0.157 解读为"颜色级迁移尚存"。但两指标的联合更支持另一种解释：ΔS = mean(out·ref) − mean(src·ref)（未归一 DINO 点积，`r3_oneshot_minimal.py:266-268,283`）；swap 低意味着**内容平均后不同参考的输出几乎重合**——一个**参考无关**的通用漂移方向只要与 OOD 参考池共享方向正相关，即可在 swap≈0.13 的同时产出 ΔS≈+0.15。即数据无法区分"+0.157 是逐参考的颜色级迁移"与"+0.157 是先验漂移恰好朝参考池方向"；后者与 label_free 教师近恒等（mid_cos 0.965，A03）＋学生学到的"少改即安全"先验**更自洽**。
- **无法定案的旁证**：`vgg_gram` 0.00054/0.00055（输出 vs 参考 gram MSE，:292）无 src 基线行、无 O3 对照值——单独不可解释，不构成反证。
- **裁决**：INCONCLUSIVE。"参考响应弱"可交付；"颜色级迁移尚存（来自参考）"不可交付，需 §十 R3 的参考消融控制裁决。

## 八、C1"优化结构问题而非超参"推理链 —— PASS（表述限定 + lr 项记录）

- **阈值过紧替代解释：已被 A03 定量排除**（审查问题点名项，答案是"有"）：A03 §2.2 精确重构 rejected 均值 0.94254——**全部 2400 个候选（含被 Top-K 淘汰的 400 个）都高于 C1 带上界 0.90**，纯带过滤保留 0 对；筛选对超额的份额仅 5.7%（(0.96463−0.96095)/(0.96463−0.90)），其余 94.3% 在目标函数。放宽阈值无候选可救——阈值不是 binding 约束。
- **步数/预算替代解释：被 48 步反转排除**（PROGRESS.md L119）：24 步 mid_cos 0.9538 → 48 步 **0.9630**（反常**上升**）。若目标函数想要大编辑而预算不足，加预算应使 mid_cos **下降**（向编辑端逼近）；实测反向收敛 → 损失盆地的极小值本身就是近恒等解——"预算不足"版本被证伪。
- **w_c 替代解释：两档排除**：w_c=1.2 keep 0/900（PROGRESS L67）、w_c=0.7 keep 0/150（L105）。
- **lr 替代解释：未独立扫描**（唯一残留）：A03 §五参数块显式"lr = 0.055 不变（变量隔离：只改 w_c 与步数）"。PROGRESS L120 原文"不是超参问题（**w_c、步数均已排除**）"作为字面陈述准确；但推广到"一切超参"时 lr 是漏项。预算反转逻辑覆盖"位移不足"版本的 lr 过小（lr×steps 是同一预算量），不覆盖"更大 lr 逃逸局部盆地"版本（非凸未测）。
- **方向支持**：三条结构重设计提案（z_opt 改参考侧统计初始化 / 内容项有界信任域 / 参考色彩 fiber 引导项，PROGRESS L120）恰好各自对准剩余未排除因素（init→盆地选择、目标形态→梯度结构、引导项→风格梯度可得性）——证据支持"下一步做结构重设计"的决策；"结构问题已证明"的强表述应降为"w_c/步数/阈值已定量排除 + 48 步反转指向损失盆地形态，lr 未测"。

## 九、总体裁定

### 9.1 机制 A 判负：**可交付为最终结论**（带两项已量化限制）

- 机制级命题"该预算（1000 upd）/该工作点（γ=0.55, D-C≈0.736）/该门控家族下，空间自适应门控无预注册帕累托增益"成立：A1 PASS≈0（一手 CI 重算）、A3 逐位退化为常数门、A2/A2_em FAIL（一手 CI 重算，Minimalism 守卫 −9.2pp 触发）、与 A06 路线级跨臂共享失效互洽。
- 限制一（§一）：λ_c 标定残差使名义 A2 读数 ~85–90% 为强度漂移伪影——引用 A2 时必须用 em 臂；限制二（§二）：学习门控臂的 eval 消费 OOD 锚输入（B4 训练 / T4 评测）——A2 归因在 R1 补评前不引用。
- "两模型混合优势来自全局统计差异本身"作为联合结论（A1+A3+em+A06）可交付；单靠 A2 不够。

### 9.2 机制 B0 判负：**不可交付为最终科学结论**（可交付为"本窗口判负，带已记录重试协议"）

- 三条主证全部受损：①Same150 崩塌 = measure 路径缺 bypass 的伪迹（mechanism_b.py:303 vs :382，§二）；②残差增长 0.23→0.63 混入 B4(训练)/T4(评测) 教师 lineage 差异（§二）；③风格随 γ 下降与已记录的欠拟合（2000 upd，lat 未收敛）相容。
- **成立的部分**：γ=0 精确回退（residual=0.0000 逐位；学生@γ0 实测 = M0_prism1_l100 的 0.74403/0.44303/0.50571，mechanism_b.json 一手核对）——工程结论可靠。
- **"双模型必要"（四核心问题之 2）的引用口径**：应引用独立实测的 D6 τ-shift 内容上限 0.7055（B_d6_t072，A04 §六.2 实测饱和），**不应**引用 B0 失败本身作为必要性证据。

### 9.3 C1 结构性重设计方向：**被证据支持**

A03 的 94.3%/5.7% 目标-筛选分解 + 恒等控制覆盖 88–96% + corr(latent_L2, mid_cos)=−0.926 + 两档 w_c keep=0 + 48 步反转，共同指向"当前目标函数的极小值即近恒等"——结构重设计（init/信任域/fiber 引导）是证据导向的下一步；lr 探针（R4）作为廉价加固。

## 十、需要补救的实验/工程清单（按优先级；全部最小设计）

| # | 对应项 | 设计 | 成本 | 裁决力 |
|---|---|---|---|---|
| **R1** | 2、1 | **B4 锚 ctx750 重评（零训练）**：按 `precompute_train_latents.py:75-83` 配方对 test750 z0 构建 B4 锚（`b4(canon(z0,c1_gentle),0,canon,tid)`），传入 `run_core_flow` 再生 z_d6_750，复用现有 `gate_A2_learned_g55.pt`/`gate_A3_smooth_g55.pt` 走 `eval_mechanism_a_arm` 重评 A0/A1/A2/A2_em/A3 → a01 配对 CI | 1 次 ctx750 构建 + 7 臂评测（4090 <1h，无训练） | 直接裁决"A2 FAIL 是否在锚对齐 lineage 下成立"；同时把机制 A 归因的 OOD 混杂归零 |
| **R2** | 2（B0） | **修 measure 路径 bypass**：`mechanism_b.py:382` 改传 bypass 后张量（同 :303 掩码覆写）；用 R1 的对齐 ctx750 对现有 `student_B0.pt` 重跑 γ 扫描评测（评测-only），并加记一条 **B4 锚教师曲线残差**（区分学生误差 vs lineage 差） | 代码 1 行 + 评测-only（<30 min） | 使 B0 的残差证据可解释；重试协议（3000 upd）仅在其后才有意义 |
| **R3** | 7 | **SG-25 参考消融控制**：label_free 在 t40/t50 各加一次 `drop_style=1`（或参考换池均值 fiber）→ 得 ΔS_prior；若 ΔS_prior ≈ +0.15，"+0.157 颜色级迁移"改判为先验漂移 | 2 × 125 对（3060 分钟级） | 一手裁决 ΔS 的参考归因 |
| **R4** | 8 | **lr 探针**：w_c=0.7、steps=48、lr∈{0.02,0.11}、150 对（并入结构重设计首批即可；若做，则"非超参"表述完全闭合） | ~25 min（4090） | 排除最后的 lr 替代解释 |
| **R5** | 4 | **文档指引更新（总控执行，本审查不改既有文件）**：04 修正块 ④ 与 GPU4090_RESULTS.md L38 的 D6@0.73 解决路径已死（τ_min 饱和实测关闭 + B0 失败），重指向"改 D6 其它参数补臂或接受现有覆盖放弃判定"；PROGRESS.md L59 问题表 ②中"Same150 崩塌为主证"撤回（伪迹，见 §二） | 纯文档 | 消除结论链中的失效指引 |

## 附录 A：A08 独立重算记录（CPU-only，a01 工具，600 任务/150 源聚类）

```
compare_arms(arm, 'MechA_A0_global_g55')，方向 = arm − A0：
A1_k50 : ΔC +0.0017 [+0.0011,+0.0023]  ΔS −0.0004 [−0.0008,−0.0000]  LPIPS −0.0002  → PASS≈0
A2     : ΔC +0.0105 [+0.0074,+0.0135]  ΔS −0.0111 [−0.0131,−0.0091]  LPIPS −0.0073 [−0.0085,−0.0061] → FAIL（守卫 Minimalism违）
A2_em  : ΔC +0.0080 [+0.0047,+0.0114]  ΔS −0.0083 [−0.0103,−0.0063]  LPIPS +0.0036 [+0.0026,+0.0046] → FAIL（守卫 Minimalism违）
A3     : ΔC −0.0001 [−0.0004,+0.0002]  ΔS +0.0002 [−0.0000,+0.0004]  LPIPS +0.0001  → PASS≈0（≡常数门）
```
与 `docs/v25/GPU4090_RESULTS.md` 机制 A 定案表及 PROGRESS.md L84-87 逐位一致。

一手测试：`pytest tests/v25/test_stats.py -q` → 16 passed；`pytest tests/v25/test_mechanism_c.py -k "label_free or inert" -q` → 12 passed。

关键证据文件：`exp/v25/mechanism_a.py`、`exp/v25/mechanism_b.py`、`exp/v25/precompute_train_latents.py`、`exp/v22/t4.py`、`exp/v25/r1_inference_ablation.py`、`exp/v25/r3_oneshot_minimal.py`、`exp/v24/p0_o1_o3_value.py`、`exp/v25/mechanism_c.py`、`src/eval/metrics.py`、`tools/v25_cpu/a01_noninferiority.py`、`tests/v25/test_stats.py`、`tests/v25/test_mechanism_c.py`、`runs/v25/mechanism_a/mechanism_a.json`、`runs/v25/mechanism_b/mechanism_b.json`、`runs/v25/measured/MechA_*/eval.json`、`docs/v25/audits/a01_round1_ci.md`、`a02_teacher_lineage.md`、`a03_teacher_quality.md`、`a04_strength_report.md`、`docs/19_pull_together/{04_round1_results,PROGRESS,agent2}.md`、`docs/v25/{GPU4090_RESULTS,GPU3060_RESULTS}.md`。
