# V25 论文材料整理与叙事草稿（DOC-02，2026-11-20）

> **任务**：[agent2.md §八](../19_pull_together/agent2.md) DOC-02（Slot A）。**只创建本文件**，不改 [paper.tex](../icme2027/paper.tex) 本体。
> **定位**：为论文方法节/实验节/讨论节准备**证据与叙事草稿**。铁律：**只准备证据，不提前宣称新机制成功**——机制 B0 重评（R12）、C1 教师重生成、M1–M9 延迟补测均未完成，相关位置一律标注"未完成/禁止预写"。
> **数值出处**：全部为已提交文档/JSON；结论分级沿用 [ABLATION_LEDGER.md](ABLATION_LEDGER.md) 证据标签：【代码证实】/【受控实验】/【受控实验·CI】/【观测性】/【待验证】。
> **与 paper.tex 的关系**：[paper.tex](../icme2027/paper.tex) 现有叙事 = PRISM-1（单步）+ PRISM-Flow 双主引擎口径；本文把 V25 的四引擎统一框架（含 Hybrid 与 Reference Adapter）整理成可并入的草稿素材，数值冲突处以 A07/A04 审计口径为准。

---

## 〇、总边界声明（写进论文前必读）

1. **NFE 勘误**（[audits/a07_cost_model.md](audits/a07_cost_model.md) §2 勘误，【代码证实】）：上游任务文本"d6_flow (cs_j 20 NFE)"与代码不符——**D6 实际 16 次 velocity 评估**（8 步 × RK2 ×2）；"20 NFE"恰是 **O3_ref**（10 步 × 2）。且 NFE 都不是真实张量工作量：cs_j ×2 / full ×4 的 CFG batch 放大后，image-equivalent 前向为 32 / 65 / 40（详见第四节）。
2. **延迟数字三个口径**（[audits/a07_cost_model.md](audits/a07_cost_model.md) §6/§9）：计时段（e2e_ms，不含 PNG+裁判）≠ 墙钟 ≠ 协议批摊销；跨文档比较必须先对齐口径。est 区间未实测，**写入论文前必须走 M1–M9 清单**（第四节）。
3. **One-Shot 评测边界**（[GPU3060_RESULTS.md](GPU3060_RESULTS.md) 头注）：SG-25 125-pair 是**阶段诊断，非 StyleGallery 论文完整协议**；O3 臂无逐任务记录，路线级失效模式未知（[audits/a06_route_report.md](audits/a06_route_report.md) F7）。详见第六节。
4. **未完成实验清单**（第九节）：M1–M9、C1 教师链（lr 探针 + 结构重设计 v2）、B0 R12 重评、双接口冒烟（U12）——这些位置的任何"成功结论"都禁止预写。

---

## 一、统一 PRISM 方法定义（方法节草稿素材）

> 四引擎共享同一前端（SD0-VAE encoder + DINO/ConvNeXt 特征栈）与同一后端（S10 解码 [decPrune]，10.66M 含 post_quant_conv，见 [audits/a07_cost_model.md](audits/a07_cost_model.md) §1）。用户接口为双入口单滑块（[../../src/v25/config.py](../../src/v25/config.py) docstring）：`PRISMCore.transfer(content, style, strength)`（三引擎）与 `PRISMRef.transfer(content, reference, strength)`（参考条件引擎）。

### 1.1 记号表（方法节 Notation 段草稿）

| 记号 | 含义 | 出处 |
|---|---|---|
| $x_s,\ z_s=\mathcal{E}(x_s)$ | 源图与其 SD0 latent（repo 存储为 scaled，$\times 0.18215$） | [paper.tex](../icme2027/paper.tex) §3 开头 |
| $\tau$ | 目标规格：画派类 $y_t$ 或单参考图 $x_{\mathrm{ref}}$ | [paper.tex](../icme2027/paper.tex) §3 |
| $(\Pi_\perp,\Pi_\parallel)$ | Stiefel 商空间 / 风格纤维对偶分解 | [paper.tex](../icme2027/paper.tex) §3.2 |
| $e(x_s\!\to\!\tau)=u_\tau-\phi_\eta(\Pi_\parallel(x_s))$ | 免源标签 universal chord | [paper.tex](../icme2027/paper.tex) 式 (universal-chord) |
| $\mathcal{D}_\star$ | S10 剪枝 decoder（10.66M，自带 post_quant_conv） | [audits/a07_cost_model.md](audits/a07_cost_model.md) §1 |
| $\lambda$ / $\tau_{\min}$ / $\gamma$ / $\tau_0$ | 四引擎强度旋钮（λ、γ 正向；τ_min、τ0 反向） | [audits/a04_strength_report.md](audits/a04_strength_report.md) §一 |
| NFE / img-eq | velocity 评估次数 / CFG batch 放大后的图像等效前向 | [audits/a07_cost_model.md](audits/a07_cost_model.md) §2 |
| T4 / B4 锚 | 评测侧默认锚 / 训练侧教师锚（distinct-scale，禁互相归因） | [MODEL_LINEAGE.md](MODEL_LINEAGE.md) §二 |

### 1.2 四引擎定义与公式

**引擎 1 · PRISM-1（单步编辑，[MODEL_LINEAGE.md](MODEL_LINEAGE.md) N0）**

- 定义：1.50M 台账口径（live 逐参数 1.280M net + 0.305M StyleFiberEncoder，[audits/a07_cost_model.md](audits/a07_cost_model.md) §1）的空间路由单步前馈网络；锚 = latent chroma-shift（α=0.20，λ=1.0 配方内固定）；无 CFG 分支、无门控、无迭代。
- 公式（沿用 [paper.tex](../icme2027/paper.tex) 式 eq:prism-1step）：
  $$\hat z_{s\to\tau}(\lambda)=z_0^\perp+\lambda\,g_{\mathrm{id}}(e)\cdot\Delta_\theta\!\big(z_0^\perp,\ P_Y z_s,\ C_{64},\ Q_{32};\ e\big),\quad g_{\mathrm{id}}(e)=\tanh^2(\|e\|_2/\sigma_e)$$
- NFE=1，CFG ×1，fresh 总前向 6，协议批总前向 3（[audits/a07_cost_model.md](audits/a07_cost_model.md) §2 表，依据 `exp/v24/p0_m0_m2.py` L31-63、`src/v25/config.py` L161-167）。
- 实测读数（cross600）：λ=1.0 → DINO-C 0.7440 / DINO-S 0.4430 / LPIPS 0.5057 / Top-1 44.33%（[audits/a04_strength_report.md](audits/a04_strength_report.md) §二.1 `M0_prism1_l100`）。
- 定位：极速高内容基座 + Hybrid 混合分量（源锁定率 0.487 五臂最高，[audits/a06_route_report.md](audits/a06_route_report.md) F3）。

**引擎 2 · Flow 编辑（V20 SLFUNet，`cs_j` / `full` 双分支）**

- 定义：22M（live 21.947M，`SLFUNet(base_dim=144)`，[audits/a07_cost_model.md](audits/a07_cost_model.md) §1）分段视界流，仅 16×16 瓶颈层注入结构语义；T4 锚；original 密度门控 sigmoid((dsd−mid)·550)。
- 公式骨架：RK2 积分的速度场 $v_\theta$，solver D5 τ∈[0.28,0.66] / D6 τ∈[0.40,0.74] / D6H τ∈[0.48,0.78]（[MODEL_LINEAGE.md](MODEL_LINEAGE.md) N1，solver hash D5 `07904c2bb16b` / D6 `e14eb16ca1e9` / D6H `f52948ae3451`）。
- NFE=**16**（8 步 × RK2 ×2，`exp/v22/core.py` L28-32、L185-199）；CFG：`cs_j` 双分支 ×2 → **32 img-eq**，`full` 四分支 ×4 → **65 img-eq**（含 P1）（[audits/a07_cost_model.md](audits/a07_cost_model.md) §2；full 的 ×4 = `z_cur.repeat(4,...)`，L153-180）。
- 实测读数：`M0_v20_d6`（D6/full）0.6688/0.5107/0.5374/55.67%；`P02_d6h_csj` 0.6782/0.5121/0.5350/55.00%（[audits/a04_strength_report.md](audits/a04_strength_report.md) §二.2；[../19_pull_together/04_round1_results.md](../19_pull_together/04_round1_results.md) P0-2）。
- **未决**：cs_j（20 img-eq 等效 NFE 减半）是否可替 full——A01 两个水位均 INCONCLUSIVE（Minimalism 守卫 −7.50/−21.67pp），默认保留 full（[ABLATION_LEDGER.md](ABLATION_LEDGER.md) F9②/U2）。**论文不得写"cs_j 已验证可替代 full"。**

**引擎 3 · Hybrid（PRISM-1 λ=1.0 + V20 D6 双模型串行线性混合）**

- 定义：两模型 latent 缓存线性混合 $\hat z_{\mathrm{hyb}}=\gamma\,z_{\mathrm{flow}}+(1-\gamma)\,z_{\mathrm{p1}}$，权重独立（参数 23.227M = SLFUNet + P1，[audits/a07_cost_model.md](audits/a07_cost_model.md) §1）；协议评测路径消费教师 latent 缓存，eval 侧仅 decode（1 前向/图，`exp/v25/r1_inference_ablation.py` L156-158）。
- NFE 口径：名义 16（D6 段）× CFG ×4 → fresh 65 img-eq；协议批 **1**（缓存混合，[audits/a07_cost_model.md](audits/a07_cost_model.md) §2 表 hybrid 行）。
- 实测读数（γ 曲线，[audits/a04_strength_report.md](audits/a04_strength_report.md) §二.3）：γ=0.25 → 0.7527/0.4481/47.50%；γ=0.55 → 0.7321/0.4744/52.67%；γ=0.85 → 0.6910/0.5003/56.00%。
- 论文定位证据：双模型必要性的**正确证据**是 D6 τ-shift 内容上限实测 0.7055 饱和（[ABLATION_LEDGER.md](ABLATION_LEDGER.md) E3）+ LPIPS≤0.51 包络下 D6 全族违约（F4），**不是 B0 失败**（A08 裁定证据换源，[ABLATION_LEDGER.md](ABLATION_LEDGER.md) §四）。

**引擎 4 · Reference Adapter（O3，One-Shot 参考条件）**

- 定义：`SLFUNetV24OneShot` 22.227M（含 RefAttnV24 0.281M）+ RefAdapter 0.066M + StyleFiberEncoder 0.305M = 22.598M（[audits/a07_cost_model.md](audits/a07_cost_model.md) §1）；三通路并联：Q=灰度 DINO 16×16 布局、V=RGB ConvNeXt v16 局部、u_glob=795D fiber 全局门，置信度门 $R=g\,R+(1-g)\,u_g$；10 步 RK2 + wj 引导（默认 wj=2.2，t45 canonical）；`style_id` 形参零使用（`oh=torch.zeros`，结构性免标签）（[audits/a08_science_review.md](audits/a08_science_review.md) §六）。
- 公式骨架：τ0 桥 $z=(1-\tau_0)\,\epsilon+\tau_0\,z_{\mathrm{src}}$（张量算子，`exp/v24/o3_sg25_eval.py` L169，[audits/a07_cost_model.md](audits/a07_cost_model.md) §4）。
- NFE=**20**（10 步 × RK2 ×2）× CFG ×2 → **40 img-eq**；fresh 47，协议批 42（flow40 + style tiny + dec；ref 原型缓存）（[audits/a07_cost_model.md](audits/a07_cost_model.md) §2）。
- 实测读数：t45 → 0.5845/0.5245/0.4213/35.33%；t40 → 0.5456/0.5304/40.83%（[GPU3060_RESULTS.md](GPU3060_RESULTS.md) ①）。
- **边界**：参考响应 swap 0.131 vs O3 0.442 的归因未决（第六节）；C1 教师重生成进行中——**新教师（reference_teacher_v2）的任何读数不存在，禁止预写**。

### 1.3 参数量与共享栈汇总（方法节表草稿）

| 引擎 | 活跃参数 (M) | NFE | img-eq 前向 (fresh / 协议批) | 出处 |
|---|---:|---:|---:|---|
| prism1 | 1.280（台账 1.5M） | 1 | 6 / 3 | [audits/a07_cost_model.md](audits/a07_cost_model.md) §1/§2 |
| d6_flow (cs_j) | 21.947（台账 22M） | 16 | 35 / 33 | 同上 |
| hybrid | 23.227 | 16 | 68 / 1 | 同上 |
| o3_ref | 22.598 | 20 | 47 / 42 | 同上 |
| 机制 B0 student（未完成） | 1.642（B0 口径） | 1 | 6 / 3 | 同上（**ckpt 未产出，禁止写论文**） |
| 共享：S10 decoder | 10.660 | — | 1 | 同上 |

---

## 二、架构示意（文字版，可直接交绘图者的 spec）

> 每段 = 一条数据流。建议论文图布局：左侧共享栈（Encoder + 特征 + fiber），中间四引擎并列，右侧共享 S10 解码；strength 滑块从底部统一注入。

**Spec A · 共享前端（所有引擎）**：输入 $x_s$（512²）→ SD0-VAE encoder 得 scaled latent $z_s$；灰度 DINOv2-S 16×16 token（结构侧）+ RGB DINO 空矩 + ConvNeXt-T 局部 token（O3 侧）+ 795D fiber（Oklab 27D + RGB DINO 768D）→ $\phi_\eta$ 球面编码到 $\mathbb{S}^{127}$ → universal chord $e=u_\tau-\phi_\eta(\Pi_\parallel(x_s))$。目标规格 $\tau$ 二选一：类原型 $u_{y_t}$（查表）或参考图 fiber（1-shot）。

**Spec B · PRISM-1（单步）**：$z_s$ → chroma-shift 锚（α=0.20）得 $z_0^\perp$ → Semantic-Chord Router 在 16×16 交互 $Q_{32}$ 与 $e$ → 1.28M 单步 net $\Delta_\theta$ 一次前向 → $z_0^\perp+\lambda g_{\mathrm{id}}(e)\Delta_\theta$ → S10 解码。标注点：NFE=1、无迭代、17.0 ms/img @4090（协议批，[docs/performance/MASTER_LEDGER.md](../performance/MASTER_LEDGER.md) §1.1）。

**Spec C · Flow（V20）**：$z_s$ → T4 锚（AnisotropicS10Teacher 2 步，评测侧；hash `04007b4f0355`）→ original 密度门控 sigmoid((dsd−mid)·550) → SLFUNet 分段视界流：前段亮度锁定布局稳定、后段自由笔触合成，16×16 瓶颈注入结构 → RK2 8 步积分 → S10 解码。CFG 分支二选一：cs_j（双分支 ×2）/ full（四分支 ×4）。标注点：NFE=16、img-eq 32/65。

**Spec D · Hybrid**：同一 $z_s$ 并行进 PRISM-1（λ=1.0）与 V20 D6 流 → 两路 latent 缓存 → 线性混合 $\gamma z_{\mathrm{flow}}+(1-\gamma)z_{\mathrm{p1}}$ → S10 解码。标注点：双模型串行、γ 为强度旋钮、协议批 eval 仅 1 前向（decode）。

**Spec E · Reference Adapter（O3）**：$x_{\mathrm{ref}}$ 侧（可缓存）：RGB DINO ×2 + ConvNeXt ×1 提参考 token/原型 → $x_s$ 侧：灰度 DINO 布局 Q、ConvNeXt 局部 V、795D fiber 全局门三通路并联 → RefAttnV24 跨注意力（16×16/32×32）→ 10 步 RK2 + wj 引导，τ0 桥起整流 → S10 解码。标注点：NFE=20、img-eq 40、参考侧特征可缓存（M6 待实测收益）。

---

## 三、论文级消融表（从 [ABLATION_LEDGER.md](ABLATION_LEDGER.md) 48 条中选摘）

> 选择标准：论文级 = 单变量可控 + 有 CI/守卫 + 有外部对照价值。每条标注证据分级与 CI 状态；"论文用途"指建议写进论文哪张表/哪段。

### 3.1 有效条目（15 条有效中选 9）

| # | 结论 | 分级 | 关键数值（出处字段） | 论文用途 |
|---|---|---|---|---|
| E1 | Round 2 训练配方 `Tr_T1`（drop_c 30%）非劣且 Top-1 +1.00pp | 【受控实验·CI】 | CI [+0.0017,+0.0200] 不含零（[audits/a01_round1_ci.md](audits/a01_round1_ci.md) 组 5/6）；seed 复验方向一致（seed2 d5 56.67%） | 训练配方鲁棒性段；注意 T0–T3 间差异"全部在噪声边缘"（[GPU4090_RESULTS.md](GPU4090_RESULTS.md) Round 2），最终取舍待 CI，不得写成"drop_c 显著更优"的整体结论 |
| E3 | D6 τ-shift 内容上限实测 0.7055 饱和 | 【受控实验】 | `B_d6_t072` DINO-C 0.7055；ΔC +0.00345/+0.00132（后者低于 0.002 饱和阈值）；死区 τ_min∈[0.50,0.58]（[audits/a04_strength_report.md](audits/a04_strength_report.md) §六.2/3） | 双引擎必要性主证据（替代被 A08 撤回的"B0 失败"证据） |
| E4 | 四条强度旋钮 (DINO-C, DINO-S) 全部严格单调，各臂在族内前沿 | 【受控实验·点估计】 | 19 实测臂（[audits/a04_strength_report.md](audits/a04_strength_report.md) §六.1/8）；唯一异常 D6 Top-1 +0.00167 = 1 任务计数噪声 | strength 接口合法性（第七节叙事） |
| E5 | 跨风格监督不可或缺（O3_self_only 单变量） | 【受控实验】 | Top-1 35.33%→12.33%（跌至错标签底噪声 8–12%）；t55 同构 10.33% vs 24.00%（[GPU3060_RESULTS.md](GPU3060_RESULTS.md) ②） | One-Shot 方法的核心消融 |
| E6 | 教师保守性归因分解：Top-K 筛选仅 5.7%，目标函数 94.3% | 【受控实验】 | rejected 均值 0.94254 精确重构；corr(latent_L2, mid_cos)=−0.926（[audits/a03_teacher_quality.md](audits/a03_teacher_quality.md) §2.2/§三） | 教师设计讨论（第八节故事）；引用时勿说"筛选造成保守"（该原命题被拒，[ABLATION_LEDGER.md](ABLATION_LEDGER.md) §四） |
| E8 | 特征几何：795D 判别饱和于 PCA-64；判别几乎全由 DINO768 块承载；Oklab27 独立低秩；u128 有效 4 维 | 【受控实验】+【观测性】 | 5-way 92.0% ≈ full 91.8%，k=32 已 88.0%；R² 0.034/0.164；PR=4.0（[audits/a05_feature_geometry.md](audits/a05_feature_geometry.md) §〇 F1–F7） | fiber 维度选择的方法学支撑 |
| E9 | 失效路线是跨臂结构性的（非臂特异） | 【受控实验·复算逐位一致】 | `Min→Uki` 5/5 臂垫底；`*→ER`/`Min→*` 两轴（[audits/a06_route_report.md](audits/a06_route_report.md) §〇/§四.3） | 失败模式段（第五节三故事） |
| E13 | 机制 A 空间自适应门控家族判负 | 【受控实验·CI】 | A1 PASS≈0；A3 ≡常数门（ΔC CI [−0.0004,+0.0002]）；A2_em FAIL（ΔS −0.0083 [−0.0103,−0.0063]）；预注册通过线无一臂达到（[audits/a08_science_review.md](audits/a08_science_review.md) §9.1） | honest ablation："两模型混合优势来自全局统计差异本身，不是空间分配"；引用 A2 必须用 em 臂 |
| E15 | O3 单 strength（τ0）曲线单调可行 | 【受控实验·点估计】 | [0.40,0.60] 单调无异常（t40 0.5457/0.5304 → t60 0.7274/0.4883，[GPU3060_RESULTS.md](GPU3060_RESULTS.md) ①） | PRISMRef 滑块接口可行性；**必须带 wj 混杂标注**（非纯 τ0 单变量） |

（未入选但可引用：E2 γ=0 构造性精确回退 residual=0.0 逐位【受控实验】——建议放 Hybrid 方法定义的"可控性"脚注，不放消融表；E10/E12/E14 为工程内部事实，不进论文。）

### 3.2 失败条目（9 条失败中选 4，作 honest ablation）

| # | 结论 | 分级 | 归因（出处字段） | 论文写法 |
|---|---|---|---|---|
| F1 | z_src 锚判负 | 【受控实验】 | Top-1 7.17%、LPIPS 0.2245 ≈ 输出≈源图；锚提供"风格目标"而非"内容起点"（[../19_pull_together/04_round1_results.md](../19_pull_together/04_round1_results.md) §三 P1-3） | 锚点设计的反例消融：锚语义可移动 Top-1 ~48pp（[MODEL_LINEAGE.md](MODEL_LINEAGE.md) §二） |
| F4 | "删 Hybrid、单模型 D6 替代"判负（当前口径） | 【受控实验·点估计】+预注册 CI 缺口 | D6 全族 LPIPS 0.518–0.537 违约 LPIPS≤0.51 包络；内容上限 0.7055 < 0.73；C-Top1 前沿由 Hybrid 主导（[audits/a04_strength_report.md](audits/a04_strength_report.md) §六.5/6） | 双引擎设计选择的依据段；如实注明 CI 缺口（U1 未决） |
| F5 | "B4+canonicalizer 锚 = T4 等价物"断言被推翻 | 【代码证实】+【受控实验】 | divergence_ratio 1.21 > 0.60；task_l2_p50 28.0 ≈ 锚-源距离 28.7（[audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md) §二 (b)） | 训练/评测 lineage 一致性审计的方法学段（防"静默换锚"类 bug） |
| F9 | Round 1 初步读法两项修正（点估计→配对 CI） | 【受控实验·CI】 | ①fixed 门控可删被 **FAIL** 否定（Δdino_c CI [−0.0187,−0.0113] 整体越带）；②cs_j 删 full 两水位 **INCONCLUSIVE**（[audits/a01_round1_ci.md](audits/a01_round1_ci.md) §二/§五） | 方法学卖点：配对 bootstrap CI + 预注册守卫如何推翻点估计读法 |

**禁止引用清单**（引用即违反 A08 裁定，[ABLATION_LEDGER.md](ABLATION_LEDGER.md) §四）：机制 B0 的"Same150 崩塌主证"（伪迹）与"学生不跟踪教师混合曲线"（混杂）；"双模型必要（证据=B0 失败）"（证据换源）；"颜色级迁移尚存"（O3 ΔS +0.157 的 INCONCLUSIVE 归因）；机制 A 名义 A2 读数（~85–90% 是 λ_c 标定残差伪影，须用 em 臂）。F3（机制 B0 判负）整体处于"判负-修正中"状态，R12 重评前不得作为最终科学结论引用。

---

## 四、真实 RGB→RGB 速度比较（性能口径，全依 [audits/a07_cost_model.md](audits/a07_cost_model.md)）

### 4.1 三档口径定义（§6）

- **cold 首次加载**：权重读盘 + 反序列化 + CUDA ctx + 首图 warmup；
- **warm 热缓存稳态**：协议路径缓存齐备后的每图延迟；
- **同参考复用**：O3 侧缓存 ref 特征（2×DINO + 1×CNX）后的增量收益。

### 4.2 现在可以写进论文的数字（measured，带口径标注）

| 数字 | 值 | 口径 | 出处 |
|---|---|---|---|
| PRISM-1 @4090 | 17.0 ms/img（58.7 img/s） | 协议批形态（latent/特征全缓存） | [docs/performance/MASTER_LEDGER.md](../performance/MASTER_LEDGER.md) §1.1（V19_S2） |
| V20 D6 家族 | 76.6–134.1 ms/img | **full 分支口径族锚**（D1..D7） | MASTER_LEDGER.md（V20.D1_core_slf_hiS） |
| V21 1-Shot | 135.4 ms/img | 协议批 | MASTER_LEDGER.md（vs StyleGallery 30,400 ms，224×） |
| O3 计时段 @3060 batch=1 | 1251.6 / 1245.4 ms（t40/t50）；峰值显存 2111.8 MB | **计时段**（`torch.cuda.synchronize` 包住生成段，不含 PNG+裁判） | runs/v24/o3_sg25_eval.json `sg25[*].e2e_ms/peak_vram_mb` |
| O3_label_free 计时段 | 1672.4 / 1529.5 ms；2135.0 MB | 同上（与 O3 差 +23~34% **未定因，M7**） | runs/v25/r3_oneshot_minimal.json `label_free_blind.sg25[*]` |
| T_IO 摊销 | 2.3 ms/img（PNG 关）～6.6 ms/img（PNG 开） | 协议 750 臂 | [audits/b04_io_report.md](audits/b04_io_report.md) §8 |
| 单臂评测墙钟 | ~50 s/臂；CPU/I/O 占 3.4%/9.8% | 评测口径（含裁判） | [audits/b04_io_report.md](audits/b04_io_report.md) §8 |
| O3 阶段占比 | flow ~81–90%，固定栈 ~10–17% | **derived/est 反推**（§4），仅可作归因叙述，不可当实测引用 | [audits/a07_cost_model.md](audits/a07_cost_model.md) §4 |

### 4.3 禁止直接入论文的数字（est/换算，补测前只能标 "est"）

- PRISM-1 ~29 ms @3060：**跨卡换算记录值，非同协议实测**（M4 补测）；AGENTS.md 环境备忘。
- d6_flow cs_j 协议批 45–80 ms/img：**derived 折半**（由 full 族锚 134.1/64 单价反推），未经直接实测（M2）。
- hybrid fresh 85–150 ms/img、cold 8–25 s、同参考复用省 38–65 ms：全部 est（M3/M5/M6）。
- paper.tex 现有 "226× / 224× throughput" 倍数：分母 StyleGallery 30,400 ms 与分子 V20 D1 full 口径可对齐，但 **cs_j 主配置一旦换用即需重测**（M2/M3）。

### 4.4 M1–M9 必测清单（投稿前 GPU 实验，[audits/a07_cost_model.md](audits/a07_cost_model.md) §8）

| ID | 内容 | 替换哪个估算 |
|---|---|---|
| M1 | o3_ref 计时段内部分段计时（VAE enc / 4×DINO+CNX / 20 vel evals / S10 dec），synchronize 包段 | §4 全部 est/derived 区间 |
| M2 | d6_flow(cs_j, D6H) 协议批 750 任务延迟 + 峰值显存 | **主表引擎唯一缺的直接锚点** |
| M3 | hybrid fresh 路径延迟（68 img-eq 口径） | est 85–150 ms/img |
| M4 | PRISM-1 @3060 同协议批口径重测 | 43× 比值分母（1252/29） |
| M5 | 全引擎 cold 首图延迟 | est 8–25 s |
| M6 | O3 同参考复用收益实测 | est 38–65 ms（B02 cache 落地依据） |
| M7 | label_free vs O3 计时段差 +0.28–0.42 s/img 定因 | R7 前必须解释 |
| M8 | 评测裁判墙钟分离（CSD/DINO/CLIP/LPIPS 750 图秒数） | 分离推理/评测口径 |
| M9 | prism1/d6/hybrid 协议批推理峰值显存 | 12GB 3060 部署边界（现仅 o3 batch=1 2.1GB、V20 训练 4.06GB） |

### 4.5 "NFE ≠ 端到端速度"四层放大（论文性能讨论段素材，[audits/a07_cost_model.md](audits/a07_cost_model.md) §5）

1. CFG batch 放大：20 NFE（O3）→ 40 img-eq；16（D6）→ 32（cs_j）/65（full 含 P1）；
2. 非 flow 固定栈：VAE enc + DINO×4 + CNX + S10 = 7 前向（est 129–229 ms/图）；
3. batch 利用率：batch=1 单价 24–28.1 ms/img-eq（derived）vs 协议批 2.10 ms/img-eq（D1 134.1/64）≈ **12× 批化摊销**；
4. 计时段外开销：I/O + 裁判 + PNG。
综合：O3 vs PRISM-1 @3060 = 1252/29 ≈ 43×（分母口径存疑，M4），而 NFE 比 = 20×——**论文不得用 NFE 单独解释延迟差异**。

---

## 五、失败模式叙事（图材料，全依 [audits/a06_route_report.md](audits/a06_route_report.md)）

**图引用**：[audits/a06_route_heatmap.png](audits/a06_route_heatmap.png)（2×3 分面，5 臂各一格 5×5 路线 Top-1 热力图 + O3 占位格明确标注 NO PER-TASK ROWS）。JSON 全量：[audits/a06_route_diagnostics.json](audits/a06_route_diagnostics.json)。

**故事 1 · `Min→Uki` 全场共同最弱路线**【confirmed】：5 臂全部 bottom-1（Top-1 0.000–0.033，源锁定 0.90–1.00——输出几乎全部停在 Minimalism）；反向 `Uki→Min` 反而是最强路线（3 臂 1.000）。极端方向不对称是"生成端存在强方向性"的路线级证据（与 V10.2 铁律 "Separation Margin ≠ Generative Direction" 呼应）。

**故事 2 · `*→ER` 难目标轴**【confirmed】：`Imp→ER` 5/5 臂锁定源 0.97、`Roc→ER` 5/5（锁回 Rococo 0.40–0.67，离开者多数落入 Impressionism 而非 ER）、`Min→ER` 4/5。ER 是最难目标（逐目标 Top-1 .15–.23）：其外观签名（素描性、低饱和）与源差距最小、判别裕度低。附 F5 修正：ER–Roc 质心最相似（+0.42）不是互相混淆，是 **Roc→ER 单向拉回**——特征空间几何不能替代配对路线评测。

**故事 3 · Ukiyo 孤立质心（生成端难、条件端易）**【confirmed·观测性】：Ukiyo-e 795D 质心孤立（A05 F4，最大质心余弦 +0.02）：入 Uki 4 路线一致地差，出 Uki 4 路线良好（`Uki→Min` 1.0）——质心孤立只伤害**生成端**，不伤害条件端。PRISM-1 对孤立目标的生成短板比 D6 族更尖锐（其最弱 5 路线含两条 Uki 目标）。

**写作警示**（[audits/a06_route_report.md](audits/a06_route_report.md) §七）：路线 n=30，单格 Top-1 二项标准误 ~±9pp——跨臂比较只看 5/5、4/5 结构性重合与 >10pp 差，不得对单格数字过度解读。O3 臂聚合 Top-1 .24 **无逐任务记录**，路线级失效模式未知（U10，未伪造）。

---

## 六、One-Shot 能力边界（评测边界声明，必须写进论文）

1. **SG-25 125-pair 是阶段诊断，非完整协议**（[GPU3060_RESULTS.md](GPU3060_RESULTS.md) 头注）。五画派 test750（Cross600 读数 + Same150 恒等控制）才是全部质量结论的协议；SG-25 OOD 读数（如 ΔS +0.157、gram 0.00055）只能作边界行为诊断，不得与 StyleGallery 论文口径对比。
2. **swap 参考响应发现与未决归因**（A08 第 7 项，[audits/a08_science_review.md](audits/a08_science_review.md) §七）：同协议下 label_free swap_dist **0.1311** vs O3 **0.442**（[GPU3060_RESULTS.md](GPU3060_RESULTS.md) ③）——"参考响应弱"成立（可交付）；但 ΔS +0.157 与 swap 0.131 联合**更支持"参考无关的通用漂移"**而非"颜色级迁移尚存"（ΔS 未归一 DINO 点积；参考无关漂移方向只要与 OOD 参考池正相关即可同时产出两读数）。归因未决（U7），解锁条件 = SG-25 参考消融控制（drop_style=1 或参考换池均值 fiber，2×125 对，3060 分钟级）。**论文写法**：报告 swap 0.131 vs 0.442 差异 + 明确标注归因 INCONCLUSIVE；禁止写"仍保留颜色级迁移能力"。
3. **O3 无逐任务记录**：路线矩阵缺失（U10）；SG-5 ER 参考失位（最近质心=Minimalism，margin −0.11，[audits/a05_feature_geometry.md](audits/a05_feature_geometry.md) F 项）。
4. **O3 τ 曲线 wj 混杂**：t40 wj=2.3 / t60 wj=1.9（[audits/a04_strength_report.md](audits/a04_strength_report.md) §六 rejected）——O3 强度曲线不得作为纯 τ0 单变量响应引用。
5. **label_free 教师链在修复中**（C1，[GPU3060_RESULTS.md](GPU3060_RESULTS.md) 队列下一步）：当前 label_free 学生行为 = "参考不敏感 + 少改即安全"（swap 0.131 + 近恒教师 mid_cos 0.965 三证据链一致）；新教师（reference_teacher_v2）读数不存在，禁止预写改进结论。

---

## 七、strength 单滑块接口叙事（全依 [audits/a04_strength_report.md](audits/a04_strength_report.md)）

**语义**：用户 `strength s∈[0,1]`，0 = 最弱编辑（DINO-C 最高实测臂），1 = 最强编辑（DINO-S 最高实测臂）；以 DINO-C 水位归一的分段线性映射，每段标注实测依据臂；s<0/s>1 无实测臂支持（**UNMEASURED**），默认 clamp。τ_min 与 τ0 是反向旋钮（旋钮↑=编辑↓），接口内部反向取值。

**各引擎最有效区间**（ΔS/|ΔC| 最大段，共同模式：保守端效率最高）：

| 引擎 | 旋钮 | 最有效区间 | 效率 ΔS/\|ΔC\| | s=0.5 默认档 |
|---|---|---|---:|---:|
| PRISM-1 | λ | [0.75, 1.00] | 0.216 | λ≈1.218 |
| Flow (D6) | τ_min（反向） | [0.66, 0.70] | 0.606 | τ_min≈0.528 |
| Hybrid | γ | [0.25, 0.40] | 1.97 | γ≈0.630 |
| O3 | τ0（反向） | [0.55, 0.60] | 0.290 | τ0≈0.506 |

（全部 descriptive 插值，出处 [audits/a04_strength_report.md](audits/a04_strength_report.md) §三/§六.7/8；JSON：[audits/a04_strength_curves.json](audits/a04_strength_curves.json)。）

**论文写作要点**：①同 s 跨引擎内容水位不同（各引擎归一化基准不同），跨引擎切换按水位对齐表插值；②D6 死区 τ_min∈[0.50,0.58]（纯内容代价无风格收益）应作为接口设计注记；③O3 映射内嵌 wj 混杂（第六节第 4 条）；④端点混杂：τ_min=0.40 端点臂 tau_hi=0.74 与 B 臂 0.78 不一致（§二.2 注意）。

---

## 八、教师诊断故事（讨论节候选：C1 否定性证据链一段草稿）

> 用途：方法节"教师设计"或讨论节"为什么保守教师不可修"段落。全部为已实测证据；U8（lr）如实标注未排除。

**草稿**：参考条件教师的第一版（O2/O3 链）产出系统性保守的伪目标：输出与源 latent 的中间余弦 mid_cos≈0.965，远高于设计频带 [0.80,0.90]。归因分解（[audits/a03_teacher_quality.md](audits/a03_teacher_quality.md) §2.2，【受控实验】）显示这不是筛选偏差——Top-K 筛选仅贡献超额保守性的 5.7%（被拒候选均值 0.94254 精确重构，2400 候选全部高于频带上界），目标函数本身（内容权重 w_c=2.2 + 22 步 Adam 预算 ≈1.2σ 位移上限）贡献 94.3%；恒等控制可覆盖 label_free 学生与 O3 差距的 88–96%。随后的两档降权实验（【受控实验】，[GPU4090_RESULTS.md](GPU4090_RESULTS.md) C1 段）证伪"降 w_c 即可修复"：w_c=1.2 时 900 对 keep=0（mid_cos≈0.957、gain 0.0017），w_c=0.7 时 150 对 keep=0；加大预算反而更深收敛进近恒等盆地（48 步 mid_cos 0.9538→0.9630，**反转**）——表明损失盆地的极小值本身即近恒等解（[ABLATION_LEDGER.md](ABLATION_LEDGER.md) E7/F6）。信任域与锚初始化的重设计开关已实现（`exp/v25/reference_teacher_v2.py`，init=ref_anchor / trust-region / color-guide 三项）；唯一未排除的超参解释是学习率（A03 冻结 lr=0.055，U8 未决）——**"一切超参已排除"的表述过度，现行口径为"w_c/步数/阈值已定量排除，lr 未测"**（A08 裁定，[ABLATION_LEDGER.md](ABLATION_LEDGER.md) §四）。该证据链的论文价值：教师端"少改即安全"先验可被学生放大为参考不敏感（swap 0.131，第六节），提示参考条件生成的瓶颈在目标函数结构而非模型容量。

---

## 九、投稿前待补实验清单（GPU 队列，禁止预写其结论）

**M 系（延迟/显存，[audits/a07_cost_model.md](audits/a07_cost_model.md) §8）**：M1–M9（见第四节表）。优先级：M2（主表锚点）> M7（label_free 定因）> M4/M9（口径与部署边界）。

**C 系（教师链，[GPU3060_RESULTS.md](GPU3060_RESULTS.md) 队列下一步 + [ABLATION_LEDGER.md](ABLATION_LEDGER.md) U7/U8）**：

| ID | 内容 | 解锁什么 |
|---|---|---|
| C-A | C1 lr 探针：w_c=0.7、steps=48、lr∈{0.02,0.11}、150 对（~25 min 4090，[audits/a08_science_review.md](audits/a08_science_review.md) U8） | 排除/确认最后一个超参解释 |
| C-B | reference_teacher_v2 结构重设计（init=ref_anchor / trust-region / color-guide）运行 + 教师质量诊断（A03 口径） | 伪目标生产解锁（keep 频带达标才进入 R3） |
| C-C | 2000 对参考条件伪目标生成 → O3 新教师训练（R3/R4） | label_free 学生重训 |
| C-D | SG-25 参考消融控制（drop_style=1 或参考换池均值 fiber，2×125 对） | U7 swap 归因判定 |

**双接口与协议收尾**：

| ID | 内容 | 出处 |
|---|---|---|
| S-A | B05 双接口端到端冒烟（`PRISMCore.transfer` / `PRISMRef.transfer` 单样本）+ checkpoint manifest 补登 | [ABLATION_LEDGER.md](ABLATION_LEDGER.md) U12；[REPRODUCIBILITY.md](REPRODUCIBILITY.md) §六 |
| S-B | R12 零训练重评（B4 锚 ctx750 重评 7 臂 + B0 bypass 修复 γ 扫描）后 a01 配对 CI | [../../exp/v25/r12_anchor_reeval.py](../../exp/v25/r12_anchor_reeval.py)；U6 |
| S-C | `P14b_d6h_fixed_g69` 重标定门控的配对 CI | U3 |
| S-D | O3 臂逐任务 measure 产物（4090 侧），解锁路线矩阵 | U10；[audits/a06_route_report.md](audits/a06_route_report.md) §七.1 |
| S-E | 跨 Round 同种子重跑一侧或显式标注 seed-unmatched | U9（评测种子 20261185 vs 20261119 分裂，[audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md)） |
| S-F | D6 vs Hybrid @0.73 判定：改 D6 其它参数（tau_hi 上调/分支结构）补实测臂出 CI，或正式放弃该判定 | U1（[audits/a08_science_review.md](audits/a08_science_review.md) §四 R5） |

---

## 十、护栏与复算

```bash
python -m pytest tests/test_markdown_links.py tests/test_no_absolute_paths.py -q
```

复算入口（CPU-only）：`python -m tools.v25_cpu.a04_strength_analysis`、`python -m tools.v25_cpu.a06_route_diagnostics`、`python -m tools.v25_cpu.a07_cost_model`（命令详见各 audit 报告）。

---

## 交付块（agent2.md §十二）

- **TASK**: DOC-02 — 论文材料整理与叙事草稿
- **STATUS**: DONE
- **FILES_CHANGED**: `docs/v25/PAPER_MATERIALS.md`（新增，唯一所有权文件）
- **KEY_RESULTS**: 9 项全覆盖——①四引擎统一定义（含 NFE 勘误 d6=16/o3=20 与参数量表）；②五段绘图 spec；③消融表（有效 9 + 失败 4，逐条证据分级 + 禁止引用清单）；④速度三档口径 + 可用/禁用数字分表 + M1–M9；⑤失败模式三故事 + 热力图引用；⑥One-Shot 边界（SG-25 阶段诊断声明 + swap 0.131 vs 0.442 归因 INCONCLUSIVE）；⑦strength 单滑块叙事（四引擎最有效区间 + s=0.5 默认档）；⑧C1 教师诊断故事草稿（lr 未排除如实标注）；⑨待补实验清单（M1–M9 + C-A..C-D + S-A..S-F）。
- **TESTS**: `python -m pytest tests/test_markdown_links.py tests/test_no_absolute_paths.py -q` — 全绿（见最终回复确认）
- **结论**: confirmed = 全部数值均为已提交文档/JSON 实测读数；rejected/未决项均按 ABLATION_LEDGER 口径标注，未预写任何未完成实验的成功结论。
- **RISKS**: ①M2（cs_j 协议批延迟）缺失使主表延迟列暂只能用 full 口径族锚；②cs_j/full 与 D6-vs-Hybrid 两项判定未决（U1/U2），论文设计选择段须如实写 CI 缺口；③O3 路线矩阵与 swap 归因未决（U7/U10），One-Shot 节只能写到边界声明层。
- **NEXT_TASK_SUGGESTION**: 若 GPU 队列有空档，优先 M2（主表延迟锚点）与 C-A（lr 探针，~25 min）——两者分别解锁主表与教师故事的最小缺口。
