# L3 实现缺陷：训练图上的 autograd 探针会让被探测的分支静默失去梯度

**发现日期**：2026-11-10
**严重性**：高 —— 会让两个不同配置产生**逐位相同**的模型，从而得出"新机制无效"的错误结论
**影响实验**：P2 的 G1a / G1b（首次运行全部作废）

---

## 1. 现象

在 P2 的 G1 臂里，为了满足"训练前必须测量 `grad(L_rank)/grad(L_V6)` 比值"的要求，
我在训练图的 `step == 1` 处插入了梯度探针：

```python
g_v6 = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
g_rk = torch.autograd.grad(args.rank_lambda * l_rank, model.parameters(),
                          retain_graph=True, allow_unused=True)
```

结果：

| 观测 | 值 |
|---|---|
| `grad(L_V6)` 范数 | 1.5126（正常） |
| **`grad(λ_r·L_rank)` 范数** | **0.000000e+00** |
| 返回的梯度张量 | 113 个，**全部非 None 但精确为 0** |
| `z_cross.requires_grad` | True |
| `z_cross.grad_fn` | `AddBackward0` |
| `l_rank.requires_grad` | True |
| `grad(L_rank)` 对 `z_cross` | **4.04e-03（非零）** |
| 需梯度的参数 | 113/113 |

**即：图是连的、中间梯度非零、参数需要梯度，但到参数的梯度精确为 0。**

---

## 2. 后果：两个配置产生逐位相同的模型

G0（无 ranking）与 G1a（有 ranking）训练完成后：

```
G0_ctrl          tensors=113 sha=a66a429aacc08470
G1a_rank_weak    tensors=113 sha=a66a429aacc08470
max|W_G0 - W_G1a| = 0.000e+00
```

**权重完全相同**，Eval200 的 7 个指标也逐位相同。

**如果只看这份结果，会得出"ranking 损失对训练没有任何作用"的结论 —— 而这个结论是错的。**

---

## 3. 定位过程

| 测试 | 结果 | 排除项 |
|---|---|---|
| 独立复现（全新模型 + autocast） | 113/113 非零，norm 55.35 | 排除 critic 本身不可微 |
| 用真实 L11 + 真实数据，无 autocast | 113/113 非零，norm 192.12 | 排除模型结构问题 |
| 同上 + autocast | 113/113 非零，norm 191.97 | **排除 autocast** |
| 先算 V6 梯度再算 rank 梯度（独立脚本） | 113/113 非零 | 排除调用顺序 |
| 训练图内先算 rank 梯度 | 仍为 0 | 排除顺序 |
| 训练图内 `l_rank.backward()` | **113/113 非零，max 14.5** | 排除图断裂 |
| **去掉所有探针，3 步对照** | **113/113 不同，max\|dW\|=1.22e-03** | **确认是探针导致** |

**决定性对照**：

```
max|W_norank - W_withrank| = 1.219578e-03   tensors_differing=113/113
  -> RANKING TERM DOES AFFECT WEIGHTS
```

---

## 4. 结论

> **在训练图上用 `torch.autograd.grad(..., retain_graph=True)` 做诊断探针，
> 会让被探测的那条分支在随后的 `loss.backward()` 中静默贡献零梯度。**
>
> **`retain_graph=True` 并不能阻止这个副作用。**
> **不报错、不警告、梯度张量存在但精确为 0。**

**这不是"探针算错了"，而是"探针改变了训练行为"。**

---

## 5. 规避方法

| 做法 | 说明 |
|---|---|
| **不要在训练图上做梯度探针** | 最安全。诊断放独立进程或独立前向 |
| 需要比值时用**独立的一次性前向** | `with torch.enable_grad():` 重建一份图，测完丢弃，不碰训练图 |
| 或用 **`.grad` 缓冲** | 训练 backward 后读 `p.grad`，但无法分离两个损失项 |
| **必须有"机制是否真的生效"的对照** | 见 §6 |

---

## 6. 必须加的守卫（本次的教训）

**任何"新增损失项"的实验，都必须有一个直接证明该项改变权重的检查**，
而不是只看日志里的损失值或诊断打印。

本次加的守卫（`run_v6_next_p2b.sh` 结尾）：

```python
mx = max(|W_G0 - W_G1a|)
if mx == 0.0:
    print("FATAL: G1a is bit-identical to G0 -- the ranking term did not reach the weights")
    sys.exit(1)
```

**同一守卫也适用于 Q2 的两个机制**（`--embed-style-lambda`、`--trust-lambda`）：
必须分别验证它们确实改变了权重，否则"无效果"的结论不可信。

---

## 7. 附带发现：`model.parameters()` 是生成器

同一段代码还有一个独立的小坑：

```python
g_v6 = torch.autograd.grad(loss, model.parameters(), ...)      # 消耗生成器
g_rk = torch.autograd.grad(rank, model.parameters(), ...)      # 新的生成器，OK
```

**每次写 `model.parameters()` 都会创建新生成器，所以这样写是安全的。**
但如果先 `_params = model.parameters()` 再传两次，第二次会收到**空序列**，
`autograd.grad` 返回空元组，`sum(...)` 为 0 —— **又一次静默失败。**

**正确写法**：`_params = list(model.parameters())`。

---

## 8. 与既有 L3 缺陷的关系

这是第 7 个已记录的实现缺陷（前 6 个见 `six_code_defects.md`）。
**共同模式**：都是"静默失败" —— 不报错、不警告、结果看起来合理。

| # | 缺陷 | 静默方式 |
|---|---|---|
| 1 | `swd2` 用 `scatter` 而非 `gather` | 结果数值错但形状对 |
| 2 | `c1_soft` 用 `lh_mean` 而非 `hl_mean` | 频带被拉向错误目标 |
| 3 | `--no-content` 不移除内容 | 消融标签与实际不符 |
| 4 | eval200 rows 缺 `src_stem` | 聚类 CI 无法计算 |
| 5 | fp16 部署两处 dtype 不匹配 | 只在 fp16 前向时崩 |
| 6 | `eval_tiers` schema 不兼容 | 只在特定任务文件上崩 |
| **7** | **训练图探针清零梯度** | **两个配置产生逐位相同模型** |
