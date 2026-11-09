# V6 24h 计划 · P0 审计报告

**日期**：2026-11-09 22:30
**依据**：`docs/WEAVE_V6_24h_add_subtract_multiply_plan.md` §3
**结论**：**P0 全部通过**，一处计划内条目（N6）经实测判定为空操作并移出队列。

---

## 1. 代码修复

| 项 | 状态 |
|---|---|
| `canonicalize.py` `c1_soft` 的 HL 带均值引用错误（`+ self.lh_mean`） | ✅ **已修**为 `+ self.hl_mean` |
| `A0/gentle` 是否受影响 | ❌ 不受影响（该行只在 `c1_soft` 分支） |
| 历史 `c1_soft` 结果（D1_c1_soft，ΔT +0.0304） | ⚠️ 应重读为"另一个（有 bug 的）变体"，**不是干净的规范化消融** |

回归保护已写入 Smoke16 第 9/10 项：注入 `hl_mean=+3`、`lh_mean=-7` 的假统计，
断言 `c1_soft` 输出的 HL 带被拉向 **+3** 而不是 −7。实测 `hl_out mean=+0.7099`（通过）。

---

## 2. A2 `no_style` 是否同时关闭了 AdaLN 与 PaletteHead

**是。** 置零发生在 `C3UNet.forward` 顶部：

```python
if self.no_style:
    style_id = torch.zeros_like(style_id)
```

之后 `s_emb = style_mlp(style_emb(style_id))`（AdaLN 条件）与
`palette_head(c, style_id)` 都读到全 0 的 ID。

Smoke16 验证：
- 第 5 项：`no_style=True` 时输出对 Style-ID **完全不变**（`allclose`）
- 第 13 项：`palette_head` 本身依赖 Style-ID（否则第 5 项无意义）

损失侧仍用真实 `y_real` / `y_cross`，评测仍按真实目标 `j` 计分 → **符合计划要求**。

---

## 3. A3 `no_canon` 是否只改了 `c_in`

**是。** 训练代码里：

```python
c_in = canon(z_real, mode=args.canon_mode, augment=True)     # ← 受 --canon-mode 影响
...
c_clean = canon(z_real, mode="c1", augment=False)            # ← 硬编码 c1，不受影响
```

`L_canon_inv` 的定义（`c1` 空间的 LL + 0.35·(LH+HL) MSE）**未被改动**。
Smoke16 第 11 项：`canon(mode="none")` 严格等于输入。

---

## 4. B1 分支：已加载，但**数值上可忽略**（P0 最重要的发现）

真实 batch（A0 配置）第 1 步诊断：

| 项 | 值 |
|---|---|
| `L_style_same` | 1.451083 |
| `grad(λ_same · L_same)` | **1.691895** |
| `grad(L_rec)` | 0.178356 |
| **比值** | **9.486** |
| B1 状态 | `loaded`（ckpt 存在） |

### 4.1 风格分布损失的梯度是重建损失的 **9.5 倍**

`w_style_dist = 0.25` 在**梯度**意义上**不是**小权重。
这是理解 A1 结果的关键背景：模型主要被风格分布项驱动，而不是被重建项驱动。

### 4.2 B1 分支贡献 0.048%，关闭后梯度**完全不变**

| 配置 | `L_style_same` | `grad(λ_same·L_same)` |
|---|---:|---:|
| B1 loaded | 1.451083 | **1.691895** |
| `--no-b1` | 1.450386 | **1.691895** |

差值 = 0.000697，即 B1 项约占风格损失的 **0.048%**；梯度范数在 7 位有效数字上完全相同。

**判定：N6（B1_off）是空操作，已移出队列。**
花 42 分钟确认一个已测得的空操作不值得；证据记录于此。

> 这同时回答了计划 §3 的提问："确认 V5 B1 checkpoint 是否确实被 A0 加载"——
> **加载了，但基本没起作用。** 因此 V6 的有效性主要来自
> 自重建 + 跨风格正则 + DWT 局部统计，**不是**辅助风格判别器。

---

## 5. 无损等价性验证

### 5.1 解耦 λ_c 后 A0 完全复现

重构前：`loss += w_cross * (L_cross + w_canon_inv * L_canon_inv)`，即
λ_s = 0.35、λ_c = 0.35×1.5 = 0.525（乘积形式）。
重构后：`loss += lam_s * L_cross + lam_c * L_canon_inv`，默认回落到乘积形式。

**实测 step-1 loss = 1.3165**，与历史 A0_ref 的 1.3165 **完全一致**。
→ 解耦重构在默认路径上**数值等价**。

### 5.2 各臂实际生效的系数

| 臂 | λ_same | λ_s | λ_c |
|---|---:|---:|---:|
| A0 / 默认 | 0.25 | 0.35 | 0.525 |
| `N5_inv_off` | 0.25 | 0.35 | **0.0** |
| `L5_same_off` | **0.0** | 0.35 | 0.525 |

已确认 `grad(λ_same·L_same) = 0.000000`（λ_same=0 时）→ 开关真实生效。

### 5.3 参考池

| 配置 | 实际 top_k |
|---|---:|
| 默认（Core64） | 64/style |
| `--ref-full` | 1024/style |
| `--ref-top-k 256` | 256/style（P0 测试用） |

> 计划写"全训练集或画家均衡子集"，但 8,989 张 Impressionism 单独就约 590 MB 显存，
> 故以 **1024/style 为最大可行池**，上限已记录，不假装是"全量"。

---

## 6. Smoke16（13 项结构检查，全过）

```
PASS  1 forward shape/finite
PASS  2 backward reaches every param
PASS  3 style swap changes output
PASS  4 content swap changes output
PASS  5 no_style is Style-ID invariant
PASS  6 content_mode=none is content invariant
PASS  7 no_inject still content dependent (direct convention z_in==c)
PASS  8 inject_64=False changes output   max|d|=1.03
PASS  8 inject_32=False changes output   max|d|=0.93
PASS  8 inject_16=False changes output   max|d|=0.17
PASS  9 c1_soft runs and is finite
PASS 10 c1_soft HL band pulled toward hl_mean
PASS 11 canon mode=none is identity
PASS 12 direct-mode stem has 4 zero channels
PASS 13 palette head depends on Style-ID
```

> 第 12 项直接验证了零训练压缩的前提：direct 模式下 `[z_in, c, z_in−c]` 有 **4 个恒零通道**。

---

## 7. 真实 Batch 开关连通性（12 项，全过）

每项跑 2 步，全部正常退出并打印正确配置：

| 开关 | 结果 |
|---|---|
| `--no-style` | ok，`no_style=True` |
| `--canon-mode none` | ok，`canon=none` |
| `--base-dim 32` | ok，**0.85M** 参数 |
| `--lambda-cross 0.35 --lambda-canon 0.0` | ok，`lam_c=0.0` |
| `--no-b1` | ok，`b1=disabled_by_flag` |
| `--no-same-style` | ok，`lam_same=0.0` |
| `--core-sampler-off` | ok |
| `--ref-top-k 256` | ok，`top_k=256` |
| `--no-inject-64/32/16` | ok，`inject` 元组对应位为 False |

---

## 8. 队列实现（计划 §3 要求）

`tools/v6_24h/queue.py`：

| 要求 | 实现 |
|---|---|
| 单 GPU 进程排他 | `O_EXCL` 锁文件（含 stale-pid 恢复）+ 启动前 `nvidia-smi --query-compute-apps` 检查，GPU 忙则记 `blocked_gpu` 并停止 |
| 单臂超时 | `1.5 × 42 min = 63 min`，超时记 `incomplete`，**不缩短曝光** |
| 状态实时落盘 | 每次状态转移写 `runs/v6_24h/state.json` |
| 每臂记录 | 配置 JSON（`configs/v6_24h/<id>.json`）、配置哈希、Git SHA、完整命令、耗时、显存、grad_diag、eval200、分目标 ΔT |
| 禁止 watcher 抢占 | 无独立 watcher；队列本身串行 |

---

## 9. 队列内容（15 臂，11.2 h）

| 顺序 | 臂 | 组 | 变更 |
|---:|---|---|---|
| 1 | N1_no_style | 减法Ⅰ | AdaLN + PaletteHead 同时固定 ID |
| 2 | N2_no_canon | 减法Ⅰ | `c_in = z`，`L_canon_inv` 不变 |
| 3 | N3_dim32 | 减法Ⅰ | 0.85M |
| 4 | N5_inv_off | 减法Ⅰ | λ_c 0.525→0 |
| 5 | N4_steps400 | 减法Ⅰ | 800→400 步 |
| 6 | L1_lowC | 减法Ⅱ | (λ_s, λ_c) = (0.35, 0.240) |
| 7 | L2_highS | 减法Ⅱ | (1.20, 0.525) |
| 8 | L5_same_off | 减法Ⅱ | λ_same = 0 |
| 9 | L3_core_off | 减法Ⅱ | Core64 采样 50%→0% |
| 10 | L4_ref_full | 减法Ⅱ | 参考池 64→1024 |
| 11 | S1_no_inject_64 | 加法 | 关 64×64 注入 |
| 12 | S2_no_inject_32 | 加法 | 关 32×32 注入 |
| 13 | S3_no_inject_16 | 加法 | 关 16×16 注入 |
| 14 | N7_dim24 | 减法Ⅰ | 0.48M |
| 15 | N7_dim40 | 减法Ⅰ | 1.31M |

**已移出**：N6_b1_off（§4.2 实测空操作）。

---

## 10. 待办（P0 未完成项，按计划优先级）

| 项 | 状态 |
|---|---|
| 零训练精确压缩（12ch→4ch stem、常量 time 合并、删 coherent target） | ✅ **已完成并验证，见 §12** |
| 数据泄漏检查 | ✅ **已完成，见 §13** |
| 统一 Val200 复评分 A0/T1/A1 + 5×5 混淆矩阵 | 未做 |
| `Screen50` / `Heldout400` / `Same150` 评测层级 | 未实现（现有 `eval200` + `same50` 覆盖主要口径） |

---

## 12. 零训练精确压缩（计划 §4 要求）— 已验证

`tools/v6_24h/fold_direct.py` 把 direct 算子折叠成更小的部署 checkpoint，
**不训练、不改函数类**：

| 折叠项 | 依据 | 参数变化 |
|---|---|---:|
| stem 12ch → 4ch | direct 模式 `z_in == c`，故 `[z_in, c, z_in−c] = [c, c, 0]`：4 通道恒零、8 通道重复 → `W4 = W12[:,0:4] + W12[:,4:8]` | −3,456 |
| 常量 τ 并入 AdaLN bias | τ 恒为 0 → `t_emb` 为常量；每个 `Linear(W,b)` 消费 `cond = s_emb + t_const`，故 `b' = b + W@t_const` | 删除 `time_mlp`（14,016） |

**逐图验证（真实 latents，10 张，五种风格各 2 张）**

| 精度 | max\|Δ\| | mean\|Δ\| | 相对误差 |
|---|---:|---:|---:|
| fp32 | **7.153e-06** | 4.086e-07 | **7.409e-07** |
| fp16 | 7.812e-03 | 4.542e-04 | 8.091e-04 |

参数：**1,881,792 → 1,878,336（−0.18%）**，折叠后 bias 偏移最大 0.735。

**判定：fp32 上等价（相对误差 7.4e-07，属浮点累加顺序差异）；fp16 在 AMP 容差内。**

> 这**不是**"从零训练 4 通道也无损"的声明——只是**等价部署压缩**。
> 训练时 12 通道的冗余权重仍然存在，只是它们作用在恒零/重复通道上。

### 12.1 顺带修掉两个真实的 fp16 部署缺陷

折叠过程中暴露了原代码在**半精度部署**下会崩的两处（训练用 autocast 掩盖了它们）：

| 位置 | 问题 | 修复 |
|---|---|---|
| `sinusoidal_embedding` | 恒返回 fp32，与半精度 `time_mlp` 权重不匹配（`mat1 Float vs mat2 Half`） | `t_emb = time_mlp(emb.to(s_emb.dtype))` |
| `StyleChannelPaletteHead` | `dwt2_haar(z.float())` 恒 fp32，与半精度 `mix_ll` 不匹配（`expected Float but found Half`） | dwt 输出 cast 到参数 dtype |

这两处不修，任何 fp16/bf16 导出都无法前向。

---

## 13. 数据泄漏审计（计划 §3 要求）

`tools/v6_24h/leakage_audit.py`，路径**归一化到 repo 相对路径**后比较
（先前用裸文件名比较会因绝对/相对前缀差异产生假阳性）。

| 检查 | 结果 |
|---|---|
| 训练池总计 15,545 | ✅ |
| **A 评测源图 vs 训练池** | ✅ **overlap = 0**（硬性，通过） |
| A2 评测参考图 vs 评测源图 | ✅ overlap = 0 |
| **B 评测参考图是否在训练池内** | ⚠️ **overlap = 150/150（全部在内）** |
| B2 `csd_proto.pt` 存在且五风格单位范数 | ✅ norms = [1,1,1,1,1] |
| C 每步覆盖五类 | ✅ `per_style_mb = 16`（microbatch 80/5） |
| D 规范化器统计有限 | ✅ 8 个张量 |

### 13.1 这是设计属性，不是 bug，但必须写进所有结论

`tools/pseudo_gt/q0_quality.py::build_prototypes` 明确从**训练池**采样
（每风格 500 张，仅排除 300 张评测源图）构建 `csd_proto.pt`；
而 30 张/风格的评测参考集也位于同一训练池。

**后果**：
1. **绝对 ΔT / SWD 是乐观的**——它们相对的是模型训练过的数据分布。
2. **但臂间比较与 Pareto 排序仍然有效**——所有臂用同一套原型与参考集。
3. `src_ids` 排除了 300 张源图，所以 `cos(src, proto)` 没有被"源图在原型里"抬高，
   ΔT 的**减法**部分是干净的。

**因此**：本报告与后续 24h 结果中，**不得**把绝对 ΔT 当作泛化性能；
只用于同口径的臂间比较。若需泛化声明，必须建立**全新的 source-heldout 参考集**。

---

## 15. 统计方法修正：源图聚类配对 bootstrap（计划 §1 要求）

### 15.1 原实现的问题

`train_eval_v6.py::boot_ci` 对 **200 个任务独立重采样**：

```python
idx = rng.integers(0, len(x), size=(n_boot, len(x)))
```

但 **200 个任务来自约 50–150 张源图**，同一源图的多个任务高度相关
（同一张图被要求迁移到多个目标风格）。把它们当作独立证据 →
**置信区间被系统性低估** → 会高估显著性。

计划 §1 明确要求"按源图聚类的配对 bootstrap"，并警告
"Val200 的 40 张/目标不足以支持小幅 Top-1 变化的强断言"。

### 15.2 修正

新增 `tools/v6_24h/stats.py`：

| 函数 | 作用 |
|---|---|
| `cluster_ci(values, clusters)` | 单个臂的均值 CI，**按源图聚类重采样** |
| `paired_cluster_ci(a, b, clusters)` | `mean(a) − mean(b)` 的 CI，同一聚类同时携带两个臂的值 → 真正的**配对**区间 |

`eval200` 现在同时保存：
- `rows`：逐任务记录（task_id / src_stem / tgt_style / dT / dS / dino_c / lpips /
  target_margin / target_top1 / pred_class），配对统计的前提
- `dT_cluster_ci`：聚类 CI（`dT_ci` 保留原任务级版本以便对照）

### 15.3 验证（合成数据，强组内相关）

40 个源图 × 5 任务，源效应 σ=1.0、组内噪声 σ=0.05：

| 方法 | CI 宽度 |
|---|---:|
| 任务级（原实现） | 0.2145 |
| **源图聚类（修正后）** | **0.3736** |

**聚类 CI 比任务级宽 74%。** 这正是原实现低估的量级。

配对检验：A 与 B 相差常数 0.3 时，配对 CI 宽度为 0，且正确检出 `excludes_zero=True`；
A 与 B 完全相同时 diff = 0 且 CI 含 0。

**判定：Go/No-Go 的"ΔT 至少 +0.008"必须用配对聚类 CI 判断，
不能用两个臂各自的 CI 是否重叠来判断。**

### 15.4 影响与限制

- **影响**：此前所有以任务级 CI 声称的"显著"都需要用聚类 CI 重读。
  单臂均值本身不变，变的是**不确定性**。
- **限制**：A0 / N1 / N2 / N3 的 report.json 生成于 `rows` 加入之前，
  没有逐任务记录，无法回溯做配对统计。它们各自的聚合均值仍然有效。
  队列结束后可用约 2 分钟/臂重跑 eval 补齐 `rows`（不重训）。

---

## 16. 结论

**P0 通过，队列可以启动。** 三条对后续解读最重要的事实：

1. **风格分布项的梯度是重建项的 9.5 倍** → V6 主要由风格项驱动。
2. **B1 辅助编码器加载了但只贡献 0.048%** → V6 的有效性不来自辅助判别器。
3. **A0 在解耦重构后 step-1 loss 完全复现（1.3165）** → 后续所有单因素差值可信。

补充两条边界：
4. **零训练压缩在 fp32 上等价（7.4e-07）**，参数 −0.18%，但只是部署压缩。
5. **评测源图无泄漏（0 重叠），但原型与参考集来自训练池** → 绝对 ΔT 乐观，
   臂间比较有效。
