# V11 实施方案

**配套理论**：`docs/v11/V11_THEORY.md`
**目标**：先让 V10 的结果具备可解释性（P0），再判定各机制真实有效性（P1–P2），
最后把任务建模改对（P3–P4）。

**总原则**

- **不再按 V10 设置跑完 40 组。** 现有 6 组中，D 组（19–26）与 E 组（27–33）结论已失效，
  F 组 Monge（37–38）目标函数错误，B 组 exp10 符号不一致。继续跑只会得到无效数字。
- 每个修复必须有**数值验证**（梯度对照、等价性测试），不能只测 loss 非零。
- 已有 `exp01/model_equiv1000/2000/3000/4000.pt` 可复用，P1 不必重训前缀。

---

## 优先级总表

| 优先级 | 工作 | 目的 | 预计 |
|---|---|---|---|
| **P0** | 修 6 处工程缺陷 + 梯度验证 | 让现有结果可解释 | 3–4 h |
| **P1** | 重跑真正的五风格分布监督 + 忠实 α-DSBM | 判定两种方法是否有效 | 2–3 h GPU |
| **P2** | EDM-only vs 完整模型 | 确认非线性运输是否超越低阶统计 | 1 h GPU |
| **P3** | 独立训练五风格条件生成 | 分离"能否生成"与"能否迁移" | 2–3 h GPU |
| **P4** | 内容约束的风格分布运输 | 修正任务建模 | 需设计 |

---

## P0：工程缺陷修复（必须先做完）

### P0-1 修复 `swd2_chunked` 梯度切断

**文件**：`sbflow/style_loss.py`

**现状（错误）**

```python
with torch.no_grad():
    xg = x[torch.randperm(n)[:k]]   # ← 在 no_grad 内创建，已脱离计算图
...
xs = xg[s:s+chunk]                   # ← 仍然脱离
```

**正确实现（全局排序 + 固定秩匹配 + 分块反传）**

1. 用 `torch.no_grad()` 只计算**索引**与**参考分位数**，不在此作用域内切片生成张量
2. 在 `no_grad` 外，用同一组索引从 `x` 取子集 → 保留 `grad_fn`
3. 对完整 batch 计算投影得分，做**全局排序**，记录每个样本的秩 `rank`
4. 分块重算前向：每块用其秩去索引**固定参考分位数** `yq[rank]`，反传后梯度累加
5. 与完整 batch 的解析实现 `swd2_full` 做**梯度数值对照**

**验收标准**

| 检查 | 通过条件 |
|---|---|
| 梯度存在 | `loss.requires_grad == True` 且 `x.grad` 非全零 |
| 与全 batch 等价 | 相对误差 `< 1e-3`（loss 与梯度各自） |
| 分块无关性 | `chunk=32` 与 `chunk=128` 结果一致（误差 `< 1e-4`） |
| 反例保护 | 若人为 `detach`，测试必须**失败** |

**新增测试**：`sbflow/tests/test_swd_grad.py`

---

### P0-2 修复评测的全局 SWD/MMD

**文件**：`sbflow/score_cpu.py::aggregate`

**现状（错误）**

```python
tgt = sub[0]["target"]              # 只取第一个目标的参考集
real = refs["styles"][tgt]["csd"]   # 却被用于全部 200 张
```

**正确实现**

- `by_direction` / `by_target`：保持现有逻辑（每个 block 内目标唯一，本来就对）
- `overall` 的 `csd_swd` / `csd_mmd2_unbiased`：改为**按目标风格分组 → 每组与自己的参考集
  算距离 → 五目标等权平均**
- 其余标量指标（`target_top1`、`target_margin`、`csd_dT`、`csd_dS`、`dino_c`、`lpips`）
  是逐样本均值，不受此 bug 影响，保持原样

**验收标准**

- `overall["csd_swd"]` 必须等于五个 `by_target[*]["csd_swd"]` 的算术平均（误差 `< 1e-9`）
- 加断言防止回归
- 在已有 6 组的 `style_scores.json` 上**重算**并记录新旧值差异

---

### P0-3 分布监督覆盖双向、五目标

**文件**：`sbflow/train5.py`（分布监督分支）

**现状（错误）**

- 恒从 `mz0` 沿规范方向 $a\to b$ 积分
- `groups=mpb`（恒为较大编号）→ **Early Renaissance 永远不是目标**

**正确实现**

对每个样本按自身方向分别构造自由 ODE 终点：

| 样本方向 | 起点 | 积分方向 | 终点目标 | SWD 分组 |
|---|---|---|---|---|
| $a\to b$（`dirs=1`） | $z_a$ | $a\to b$ 正向 | $z_b$ 真实 | `pb` |
| $b\to a$（`dirs=0`） | $z_b$ | $b\to a$ 反向 | $z_a$ 真实 | `pa` |

即：**每个样本都要有一个"从自身源风格出发、沿自身方向积分、与自身目标风格比较"的终点**。
实现上用掩码把批次拆成两组，各自积分，再合并计算分组 SWD。

**验收标准**

- 5 个目标风格在 `groups` 中出现次数近似相等（各占约 20%）
- 构造只含 `dirs=0` 的 batch，验证 `ld` 仍非零且**梯度非零**
- 构造目标风格为 `Early_Renaissance` 的 batch，验证它确实被作为 SWD 目标

---

### P0-4 忠实实现 α-DSBM 的方向专属损失

**文件**：`sbflow/online_bridge.py::online_loss_mixed`

**论文 Algorithm 1**

1. 用**反向 SDE** 得 $\hat x_0$，在 $(\hat x_0,x_1)$ 上监督**前向漂移** $b_+$
2. 用**前向 SDE** 得 $\hat x_1$，在 $(x_0,\hat x_1)$ 上监督**反向漂移** $b_-$

**现状（不等价）**

```python
l_fwd = mixed_bridge_loss(model, x0_hat, z1, ..., dirs, ...)   # 内部按 dirs 各取部分方向
l_rev = mixed_bridge_loss(model, z0, x1_hat, ..., dirs, ...)   # 同上
```

`mixed_bridge_loss` 内部用 `torch.where(fwd, bplus, bminus)`，导致
$(x_0,\hat x_1)$ 这一对也被用来监督部分样本的 $b_+$，与论文配对不符。

**正确实现**

新增两个**方向固定**的损失函数：

```python
def mixed_bridge_loss_fwd(model, z0, z1, pa, pb, stats, ...)  # 全部样本监督 b+
def mixed_bridge_loss_rev(model, z0, z1, pa, pb, stats, ...)  # 全部样本监督 b-
```

然后

```python
l_fwd = mixed_bridge_loss_fwd(model, x0_hat.detach(), z1, pa, pb, ...)   # (x̂0, x1) → b+
l_rev = mixed_bridge_loss_rev(model, z0, x1_hat.detach(), pa, pb, ...)   # (x0, x̂1) → b−
```

**注意**：`pa,pb` 是规范对，但 $\hat x_0$ 属于风格 $a$、$x_1$ 属于风格 $b$，
所以前向漂移用规范方向；反向漂移的局部时间 $\tau=1-t$，EDM 统计需按反向交换。
这与 `pair_stats_per_sample(..., dirs)` 的现有语义要重新对齐，需单测验证。

**验收标准**

- 构造小批次，数值验证 `l_fwd` 只依赖 $b_+$ 目标、`l_rev` 只依赖 $b_-$ 目标
- 与 V10 旧实现对比，记录 loss 差异（应显著不同）
- `online_direction="both"` 时两方向都参与；`"forward"`/`"reverse"` 时只取对应项

---

### P0-5 修复 `score_current` 反向速度符号

**文件**：`sbflow/train5.py::objective_mixed`

**正确实现**

```python
v_canonical = 0.5 * (bplus - bminus)
v_star = torch.where(fwd_c, v_canonical, -v_canonical)   # 反向取负
s_star = (bplus + bminus) / eps                          # score 不取负
```

**验收标准**

- 构造 `eps→0` 极限，验证 `v_star` 与 `single_velocity` 的 $v^*=(z_1-z_0)$ 一致（正反双向）
- 验证正反两方向在同一 $(z_0,z_1)$ 上速度严格相反
- 新增测试 `test_score_current_reverse_sign`

---

### P0-6 修复 Monge 目标（均值坍塌）

**文件**：`sbflow/train5.py::objective_mixed`（monge 分支）

**现状（数学错误）**：独立端点 MSE → 最优解 $T(z_0)=\mu_j$（理论 §6.2）

**两个可选修正**

**方案 A（推荐，改动小）**：用 **OT 配对的端点**。已有 `paired_batch` 可按 RGB thumbnail
做 mini-batch OT。配对后 $\mathbb E[z_1\mid z_0]$ 才有意义，MSE 不再塌到 $\mu_j$。

**方案 B**：改成 §4 的内容约束 + 风格分布目标（属 P4 范畴，工作量大）。

**实施**：`monge` 分支强制要求 `ot_ratio > 0`，否则**显式报错**而不是静默训练一个
数学上必然坍塌的目标。同时保留 `pivot_cfm` 走 CFM 正确目标。

**验收标准**

- 构造合成高斯两端，验证 `T(z_0)` 的**条件均值**不等于 $\mu_j$（即未坍塌）
- 若 `ot_ratio=0` 且 `objective=monge`，必须 `raise ValueError` 并说明原因

---

### P0-7 用真梯度替代"loss 非零"检查

**文件**：`sbflow/train5.py`

**现状（验证漏洞）**

```python
dist_seen["grad_nonzero"] = float(ld) != 0.0
```

**正确实现**

在第 1 步对 `ld` 单独反传，检查生成器参数上的梯度范数：

```python
g = torch.autograd.grad(ld, [p for p in model.parameters() if p.requires_grad],
                        retain_graph=True, allow_unused=True)
dist_seen["grad_norm"] = float(sum((x.norm()**2 for x in g if x is not None), torch.zeros(())) ** 0.5)
assert dist_seen["grad_norm"] > 1e-12, "L_dist has no gradient to the generator"
```

**验收标准**：若 SWD 梯度被切断（回归 P0-1 的 bug），训练必须**在第 1 步直接失败**。

---

## P1：重跑真正的分布监督与 α-DSBM

**前提**：P0 全部通过。

已执行的四个组（新 ID，不沿用 V10 编号）：

| 组 | 训练内容 | 目的 |
|---|---|---|
| **d1** | 修复后 learned-encoder Global+Local SWD | 直接风格分布监督是否有效 |
| **d2** | 修复后 fixed-latent Global+Local SWD | 学习到的风格表征是否必要 |
| **e1** | 修复后方向专属 SDE4 + EMA teacher | 真正的 α-DSBM |
| **e2** | 双向 ODE4 确定性重配对 | SDE 与 reflow 的区分 |

全部从 `exp01/model_equiv3000.pt` 分叉，追加 1000 步，Batch512 五风格混合，
固定 4000 等效 LR 周期（不重启 warmup）。
对照为 `exp01/model_equiv4000.pt`（bidir 目标未受任何 V11 修复影响，**不重训**）。

### 判定时必须区分的三件事

**1. 训练/推理轨迹不一致（d1/d2）**
可微 ODE 用 **2-step Euler** 产生 L_SWD 的终点，而正式评测用 **4-step Heun**。
若 d1/d2 结果不佳，必须区分：
- 监督目标本身无效
- 还是监督指向的终点与推理终点不一致（2-step Euler ≠ 4-step Heun）

区分办法：用同一 checkpoint 分别在 2-step Euler 与 4-step Heun 下评测，
若两者差异与 d1/d2 的效应同量级，则"轨迹不一致"是主要解释。

**2. ρ 必须用真实 Batch**
合成数据上的 ρ=0.654 / 1.339 只证明链路接通，不代表实际监督强度。
必须在真实训练 Batch 上报告 `dist_grad_norm`、`bridge_grad_norm`、`ρ`。

**3. 分布监督的判定标准（按 `V11_METRICS.md`）**
- 必须 **ΔT 与 Margin 同向改善**，且
- **SWD 向地板 0.0111 靠近**（0.001 级抖动不算），且
- Rococo / ER / Ukiyo-e 有真实收益
- 仅总体 SWD 下降不算有效
- 同时报告 DINO-C / LPIPS 作为内容破坏约束

### 不允许的推断

- **不得**用一次固定 λ=0.1、1000 步的后训练结果否定整个分布监督路线
- **不得**把 exp27（错误实现）当作对 α-DSBM 的否证
- **不得**把 ΔT 描述为概率

**记录**：除原有字段外，必须记录 `dist_grad_norm`、`bridge_grad_norm`、`ρ`、
每目标 SWD 分组样本数、方向覆盖统计。

---

## P1b / P1c：由 d1 结果引出的两条后续（已登记，暂不执行）

d1 的实测：**L_dist 下降 65%（0.0400 → 0.0140），但 CSD-SWD 从 0.02708 变成 0.02723。**
原因是 `DistributionLoss._features` 工作在**潜空间**
（`fixed`：原始潜变量展平；`learned`：StyleEncoder 嵌入），
而评测 SWD 在 **CSD 空间**（768 维风格分类器特征）。
**在潜空间对齐分布，不保证 CSD 空间的分布距离下降。**

### P1b / d3：直接在评测空间监督

把 SWD 换成 CSD 空间 SWD：每步对生成的端点跑冻结 CSD 前向，在 768 维特征上算分组 SWD。

- **优点**：监督目标与评测指标同空间，逻辑上唯一正确
- **代价**：每步多一次 CSD（ViT-L）前向+反向。用 64 个端点的小子集控制开销，
  预计每步 +0.5~1 s
- **风险**：CSD 是判别式特征，梯度可能落在对抗性方向（提高 CSD 相似度但视觉退化），
  必须同时报告 DINO-C / LPIPS

### P1c / d4：把 StyleEncoder 蒸馏成 CSD 的代理

不引入 CSD 到训练回路，而是训练 StyleEncoder 去**回归 CSD 特征**
（或保持 CSD 空间距离），使潜空间 SWD 成为 CSD 空间 SWD 的忠实代理。

- **优点**：训练开销几乎不变；可直接复用 22–26 的既有结构
- **验证**：必须检查 StyleEncoder 空间与 CSD 空间的 SWD **秩相关**
  （在若干生成云上比较），秩相关低则代理无效

**两条路都应在 P3 之后，且只在 P3 显示"能生成目标风格"时才有意义**——
若单域生成都失败，跨域分布匹配就无从谈起。

---

## P2：EDM-only 对照（判定 L2）

**目的**：确认网络残差 $F_\theta$ 是否学到了任何超越低阶统计的东西。

**做法**：训练时**冻结/清零网络输出**，只用 EDM 解析项积分，评测同一 Quick200。

$$
b_\theta(z,t)=b_{\text{analytic}}(z,t)\quad(\text{网络输出置零})
$$

**判定**：

- 若 EDM-only 的 ΔT 与完整模型**接近**（差距 < 0.01）→ **L2 成立**，V10 主要在学统计迁移
- 若 EDM-only 显著更差 → 网络确实学到了东西，问题在别处（指向 L3 或评测）

**实现**：`gen5` 加 `--edm-only` 开关，推理时跳过网络残差项。

---

## P3：独立五风格条件生成（分离能力）

**目的**：把"能否生成"与"能否迁移"拆开。

**做法**：训练**无条件/仅风格条件**的生成模型（不做跨域桥），对每个风格采样并与
该风格真实分布比较。

**判定矩阵**

| 无条件生成 | 跨域迁移 | 结论 |
|---|---|---|
| 好 | 差 | 问题在**内容条件与运输约束**（→ P4） |
| 差 | 差 | 问题在**目标分布建模**（→ 容量/数据/表征） |
| 好 | 好 | V10 的失败纯属 L1 工程问题 |

这是**最有信息量**的一个实验，建议在 P1 之后尽早做。

---

## P4：内容约束的风格分布运输（修正任务建模）

按理论 §4 实现

$$
\min_{P:\,P_0=p_i}\ \operatorname{KL}(P\|R)+\lambda_s D\big((\Phi_s)_\#P_1,p_j^{\text{style}}\big)
+\lambda_c\,\mathbb E_P\|\Phi_c(Z_1)-\Phi_c(Z_0)\|^2
$$

**需要先解决的三件事**

1. **$\Phi_s$ 怎么来**：可用已训练的对比风格编码器，但需验证它在"风格特征空间"里
   的分布距离是否比 CSD 更可优化
2. **$\Phi_c$ 怎么来**：DINO patch 特征 / LPIPS 特征 / 或训练一个内容编码器。
   **不能先验假定 VAE 通道可分**
3. **$\mathcal L_{\text{transport}}$ 的形式**：位移惩罚？路径长度？需消融

**同时准备路线 B（条件采样）**

$$
q_j(z\mid z_x)\propto p_j(z)\exp[-\lambda d_c(z,z_x)]
$$

先用 P3 的共享模型学 $p_j$，再叠加内容引导。**这条路线更容易诊断，建议与路线 A 并行评估。**

---

## 验证与记录要求（所有阶段）

每个实验必须记录，缺一不可：

| 字段 | 说明 |
|---|---|
| `git_sha` | 本地权威 SHA（3060 非 git 仓库） |
| `config` | 完整 resolved 配置 |
| `vae` / `scaling_factor` / `latent_shape` | 固定元数据 |
| `seed` / `batch` / `patch` / `updates` | 训练规模 |
| `exposures` / `per_pair_exposure` | 等效曝光量 |
| `train_seconds` / `peak_vram_mib` | 成本 |
| `first50_sec_per_update` | 前 50 步墙钟速度 |
| `dist_grad_norm` | **新增：分布监督真梯度范数** |
| `swd_group_counts` | **新增：每目标风格 SWD 样本数** |
| `eval` | Quick200 全部指标（含修正后的分组 SWD/MMD） |
| `status` | done / failed / incomplete（含真实原因） |

---

## 实施顺序（本轮立即执行）

1. **P0-1** `swd2_chunked` 梯度修复 + `test_swd_grad.py`（含反例保护）
2. **P0-2** `score_cpu` 分组 SWD/MMD + 重算已有 6 组
3. **P0-5** `score_current` 反向符号 + 单测
4. **P0-6** Monge 强制 OT 耦合 + 单测
5. **P0-7** 真梯度断言
6. **P0-3** 分布监督双向五目标
7. **P0-4** α-DSBM 方向专属损失

每完成一项立即 `git commit`（一个机制一个 commit），并在 3060 上跑对应单测。

---

## 禁止事项

- **不得**在 P0 未通过时启动 P1 训练
- **不得**用"loss 非零"代替"梯度非零"
- **不得**在报告里把分组前/后的 SWD 混用
- **不得**把 exp27 的结果当作对 α-DSBM 的否证（它是错误实现的产物）
- **不得**声称"SD-VAE 是瓶颈"——LPIPS/DINO-C 测的不是往返重建误差
- **不得**在 Monge 目标仍为独立端点 MSE 时继续跑 exp37/38
