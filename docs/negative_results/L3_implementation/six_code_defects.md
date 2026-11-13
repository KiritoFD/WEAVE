# L3 实现缺陷：六处使 V10 结论失效的代码错误

**层**：L3（实现缺陷）
**状态**：**全部已修复并验证**（commit `ef1f3db`）
**影响**：V10 的 D 组（19–26）、E 组（27–33）、F 组 Monge（37–38）、B 组 exp10 结论**失效**

> 这一层最危险，因为**代码能跑、损失非零、日志正常**，但优化的不是声明的目标。

---

## 1. 总表

| # | 位置 | 缺陷 | 失效的结论 |
|---|---|---|---|
| 1 | `style_loss.py::swd2_chunked` | `xg` 在 `torch.no_grad()` 内切片 → **梯度被切断** | "exp24 分布监督无效" |
| 2 | `score_cpu.py::aggregate` | `tgt = sub[0]["target"]`，200 张混合目标图全对一个参考集 | 整体 SWD/MMD 不是五目标指标 |
| 3 | `train5.py` 分布监督 | 恒从 `mz0` 积分规范方向，`groups=mpb` 恒为较大编号 | 只监督 $a\to b$；ER 永不为目标 |
| 4 | `online_bridge.py::online_loss_mixed` | 两配对都调 `mixed_bridge_loss`（按 `dirs` 各取半） | 与论文 Algorithm 1 不等价 |
| 5 | `train5.py` `score_current` | 反向对速度**未取负** | exp10 不是有效对照 |
| 6 | `train5.py` `monge` | 独立端点 MSE → 最优解 $T(z_0)=\mu_j$ | exp37/38 目标函数数学上无效 |
| 7 | `train5.py` 梯度诊断 | `rec = r` 被每个 microbatch 覆盖 | 所有重型组的 ρ 静默丢失 |

---

## 2. 逐条说明

### 2.1 SWD 梯度被切断（最严重）

```python
with torch.no_grad():
    xg = x[torch.randperm(n)[:k]]      # ← 已脱离计算图
...
xs = xg[s:s+chunk]                      # ← 仍然脱离
total = total + (xq - yq).square()...   # ← 无 grad_fn
```

而训练器只检查

```python
dist_seen["grad_nonzero"] = float(ld) != 0.0   # ← 只证明损失值非零
```

**这个检查恰好能被零梯度通过。** 直接解释了
"exp24 加了显式 SWD 监督，SWD 却与基线完全相同（0.0245 vs 0.0245）"。

**修复**：估计器只定义一次（秩匹配 SWD），实现两遍（全批 / 分块）。
分块版在 `no_grad` 内固定全局排序置换与参考分位数，再**带梯度**重算
`x_chunk @ proj` 并按全局秩 scatter。
**验证**：与全批的 loss 与梯度相对误差 **0.0**（chunk 16/32/64/128）。

### 2.2 评测的全局 SWD 用错参考集

```python
tgt = sub[0]["target"]              # 只取第一个目标的参考集
real = refs["styles"][tgt]["csd"]   # 却用于全部 200 张
```

**修复**：`overall` 改为按目标风格分组、五目标**等权平均**。
用修正口径重算已有 6 组：绝对值被低估约 10%（MMD 约 40%），但**相对排序不变**。

### 2.3 分布监督只训练规范方向

恒从 `mz0` 沿 $a\to b$ 积分，`groups=mpb` 恒为较大编号：
- $b\to a$ 方向从未被监督
- **Early Renaissance（编号 0）永远不是目标**

**修复**：新增 `free_ode_endpoint`，每个样本从**自身源风格**沿**自身方向**积分
（反向用局部速度 $-v$）。`groups` 覆盖 `pa` 与 `pb`，五风格各 8/8/8/8/8；
正反 ODE 往返误差 1.2e-7。

### 2.4 在线更新不是方向专属

两种配对都调 `mixed_bridge_loss`，后者按 `dirs` 各取部分方向 →
$(x_0,\hat x_1)$ 这一对也被用来监督部分样本的 $b_+$，与论文配对不符。

**修复**：新增 `mixed_bridge_loss_fwd/_rev` 与 `bridge_loss(mode=)`，
`(x̂₀,x₁)→b⁺ only`、`(x₀,x̂₁)→b⁻ only`。

### 2.5 `score_current` 反向符号错

反向局部时间 $\tau=1-t$，速度应取负：

$$
v_{\text{reverse}}(z,\tau)=-v_{\text{forward}}(z,t)
$$

**但 score 不取负**（空间梯度与时间方向无关）。
V10 交换了 $\tau$ 与 EDM 统计，却对速度**没有取负**。

**修复**：抽出 `score_current_targets()`；$v$ 取负、$s$ 不取负；
$\varepsilon\to0$ 时 $v=z_1-z_0$（误差 2.4e-7）。

### 2.6 Monge 目标数学上无效

独立端点下

$$
\mathcal L=\mathbb E\|T(z_0)-z_1\|^2=\mathbb E\|T(z_0)-\mu_j\|^2+\operatorname{Tr}(\Sigma_j)
$$

最优解

$$
\boxed{T_\theta(z_0)=\mu_j}
$$

**这是目标均值回归，不是 Monge 运输。** 工程完全正常、loss 持续下降，模型也只会
把所有输入映到目标分布均值。

**修复**：`objective=monge` 现在**强制要求** `--monge-ot` 并用 OT 耦合端点，
否则直接报错；`pivot_cfm` 拆出为正确的速度回归 CFM（其最优解不是目标均值）。
数值演示：独立配对塌到 $\mu_j=1.996$，秩耦合恢复斜率 0.510（真值 0.5）。

### 2.7 ρ 被 microbatch 覆盖

梯度诊断只在第 1 个 microbatch 测，但 `rec = r` 被后续 microbatch 覆盖。
所有 `grad_microbatch=64` 的重型组（8 个 microbatch）都丢掉了 ρ。

**修复**：保留在 `dist_seen`，日志时合并。d1 的 ρ 用 2 步探针恢复
（`bridge_grad_norm` 复现到 6 位：0.816872）。

---

## 3. 教训（写进流程）

1. **"损失非零"不等于"梯度非零"。** 必须用 `autograd.grad` 测真实梯度范数。
2. **评测口径要单元测试。** `overall == mean(by_target)` 现在有断言。
3. **目标函数的数学性质要推导。** Monge 的均值坍塌是纸上一行就能看出的，
   但如果只看 loss 曲线永远发现不了。
4. **诊断字段要验证真的写进了产物。** ρ 丢失是因为它只在第 1 个 microbatch 产生。

---

## 4. 相关文档

- 修复说明：commit `ef1f3db`、`a45ad84`
- 理论：`docs/v11/V11_THEORY.md` §6–7
- 验证测试：`sbflow/tests/test_swd_grad.py`、`test_score_aggregate.py`、
  `test_v11_math_fixes.py`、`test_v11_dist_directions.py`、`test_v11_online_directions.py`
