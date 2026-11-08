# V9 训练/评测开销分析与 OT 瓶颈定位

> 目的：为 infra 优化提供可复现的测量数据。所有数字来自 3060（RTX 3060 12 GB，WSL2，PyTorch 2.10+cu128，bf16 autocast）。
> 复现脚本：`sbflow/scripts/bench_components.py`、`sbflow/scripts/profile_ot.py`、
> `sbflow/scripts/profile_ot_sampler.py`、`sbflow/scripts/probe_batch.py`。
> 原始数据：`runs/blog_sbflow/bench/*.json`。

---

## 一、结论先行

1. **训练耗时 85% 在 Sinkhorn OT，不在网络。** 32.6M 网络的 fwd+bwd 只占 11%。
2. **OT 的代价是"固定延迟"，与 batch 无关**（batch 16/64/256 都是 ~104 ms/次）。所以它不是算力问题，是**每轮迭代的 kernel launch / Python 开销**问题。
3. **Sinkhorn 200 次迭代是 20 倍冗余**：真实数据在 `tol=1e-4` 时 5 次收敛、`tol=1e-6` 时 10 次收敛（行边缘误差 4e-8）。
4. **打开提前收敛后，32M 的 OT-on 训练从 0.442 s/update 降到 0.244 s/update（1.8×）**，数学结果不变。
5. **10 GB 显存在当前 latent 尺寸下不可达**：32×16×16 latent + 64 token，32M 模型在 batch 256 只吃 4.4 GB，Lite 在 batch 512 只吃 1.5 GB。
6. **评测慢在 DCAE 编解码 + PNG 落盘 + 模型加载**，方法本身的 ODE 只占 8%。

---

## 二、训练开销：分组件实测（100 次 update 平均）

| 配置 | 参数量 | mb×accum | 有效 batch | s/update | OT | fwd | bwd | opt | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32M, OT on, 200 iters | 32.6M | 16×4 | 64 | **0.442** | **374.9 ms (85.2%)** | 18.7 | 29.2 | 13.7 | 1064 MiB |
| 32M, OT on, **tol=1e-6** | 32.6M | 16×4 | 64 | **0.244** | 176.3 ms (72.7%) | 18.4 | 30.7 | 13.4 | 1064 MiB |
| 32M, OT off | 32.6M | 256×1 | 256 | 0.245 | 0.8 ms | 72.7 | 151.0 | 13.8 | 4384 MiB |
| 8M, OT off | 7.45M | 256×1 | 256 | **0.083** | 0.7 ms | 23.3 | 48.1 | 5.1 | 1573 MiB |
| 2M, OT off | 1.89M | 256×1 | 256 | **0.049** | 0.7 ms | 12.8 | 24.9 | 4.4 | 857 MiB |

读法：

- **OT 打开时，把网络从 32.6M 缩到 1.9M 只省 21%**（0.442 → 0.366 s/update，后者未列全但 OT 占 89.7%）。缩网络基本无用。
- **OT 关掉时，网络规模立刻成为主项**：32.6M 0.245 vs 7.45M 0.083 vs 1.89M 0.049 s/update。
- **仅加提前收敛（不改任何数学）就让 32M 的 OT 段从 375 ms 降到 176 ms。**

---

## 三、OT 为什么这么贵：逐层拆解

### 3.1 与 batch 无关 → 不是 FLOPs

`sbflow/scripts/profile_ot.py` 实测（reg=0.05，reps=20）：

| batch | 代价矩阵 (cdist) | Sinkhorn 200 iters | 行条件采样 | 合计 |
|---:|---:|---:|---:|---:|
| 16 | 0.073 ms | **52.09 ms** | **51.66 ms** | 103.8 ms |
| 64 | 0.245 ms | **54.67 ms** | **52.05 ms** | 107.0 ms |
| 256 | 0.612 ms | **54.00 ms** | **52.48 ms** | 107.1 ms |

batch 从 16 增到 256（代价矩阵元素数 ×256），总时间只从 103.8 变到 107.1 ms。
**说明这 107 ms 全是固定开销。**

### 3.2 拆开这 107 ms

注意上表 "行条件采样" 那一列是 `choose_targets()` 的整体耗时，**其中包含了它内部再跑一次的 Sinkhorn**。真正的采样器耗时单独测得：

| 操作 | batch=16 | batch=64 | batch=256 |
|---|---:|---:|---:|
| `torch.multinomial(cond, 1, replacement=True)` | **0.157 ms** | 0.158 ms | 0.161 ms |
| `torch.searchsorted(cumsum, u)`（逆 CDF） | 0.072 ms | 0.074 ms | 0.074 ms |
| Gumbel-max + argmax | 0.133 ms | 0.137 ms | 0.140 ms |

所以单次耦合的真实构成是：

- **Sinkhorn 200 次迭代 ≈ 52 ms**（即 **0.26 ms / 迭代**）
- 采样 ≈ 0.16 ms（可忽略）
- 代价矩阵 ≈ 0.07–0.6 ms（可忽略）

`0.26 ms/迭代` 对 16×16 到 256×256 的矩阵都是这个数，说明**每次迭代被 ~5–8 个 kernel 的启动延迟支配**（logsumexp 里含 max/sub/exp/sum/log），
而不是矩阵运算本身。这是典型的 **launch-bound / host-bound** 问题。

### 3.3 迭代次数是 20 倍冗余

`profile_ot_sampler.py` 在**真实缩略图**上测收敛（reg=0.05）：

| tol | 实际用掉迭代 | 行边缘误差 |
|---|---:|---:|
| 1e-4 | **5** | 3.5e-6 ~ 1.1e-5 |
| 1e-6 | **10** | 4.0e-9 ~ 4.8e-8 |
| 1e-8 | 10 ~ 200 | 4.0e-9 ~ 2.2e-8 |
| 200（固定） | 200 | 3.3e-9 ~ 2.2e-8 |

200 次迭代得到的精度和 10 次**完全同级**。原实现的 200 是纯浪费。

### 3.4 还有一处 ×accum 的重复

`train.py` 每个 microbatch 调一次 `pick_pair` → 每次 update 调 `accum` 次 Sinkhorn。
默认 accum=4，等于**同一 update 内解 4 次互不相关的小 OT 问题**。
若改成"每个 update 对完整有效 batch 解一次、再切分给各 microbatch"，
Sinkhorn 调用次数直接 ÷4（而且配对问题更大、更合理）。

---

## 四、已落地的优化（不改数学）

1. `sbflow/ot.py::sinkhorn(..., tol=, check_every=)`：新增可选提前收敛，`tol=0` 保持原行为。
   验证：`tol=1e-6` 时 300 → 110 次迭代，plan 最大差 **2.1e-7**。
2. `sbflow/train.py --ot-tol / --ot-iters`：把容差和迭代上限暴露为参数。
3. `sbflow/train.py --warmup`：可显式设置预热步数（原来固定 `min(1000, steps//10)`）。
4. 日志去同步：`loss_batch(..., want_info=)`，非日志步不再做 `.item()`。

实测收益：32M / OT-on 从 **0.442 → 0.244 s/update（1.8×）**。

---

## 五、仍待 infra 决策的点（按收益排序）

| # | 现状 | 建议方向 | 预期收益 |
|---|---|---|---|
| 1 | Sinkhorn 每迭代 ~0.26 ms，纯 launch 开销 | **CUDA Graph 捕获整个固定迭代数的 Sinkhorn 循环**（形状静态），或 `torch.compile` 迭代体 | 可能 10–50× |
| 2 | 每 update 调 accum 次 | 每 update 只解一次、对 `accum·B` 的耦合切分 | ÷4（accum=4） |
| 3 | 每次迭代 2 个 logsumexp 分两次 launch | 合并成单 kernel（同时更新 u 和 v 的 block-coordinate 版本） | ~2× |
| 4 | 采样用 `torch.multinomial` | 已测逆 CDF 快 2.2×（0.16 → 0.074 ms），但占比小，优先级低 | 可忽略 |
| 5 | `torch.randint` 取索引每 microbatch 2 次 | 预生成一个 epoch 级的索引池 | 小 |

**建议的最小验证顺序**：先做 #2（改代码，零风险，立刻 ×4），再做 #1（CUDA Graph，需要形状固定）。

---

## 六、显存：10 GB 目标为什么达不到

`probe_batch.py` 实测 V9-Lite（1.89M）纯 fwd/bwd：

| microbatch | 32 | 64 | 128 | 256 | 384 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| ms/microbatch | 34.7 | 35.0 | 34.9 | 41.1 | 56.0 | 70.9 |
| ms/sample | 1.08 | 0.55 | 0.27 | 0.16 | 0.15 | 0.14 |
| 峰值显存 (MiB) | 199 | 293 | 462 | 818 | 1172 | 1521 |

即使 batch=512 也只占 1.5 GB。原因是 latent 只有 `32×16×16`、DiT 只有 64 个 token，
每样本激活极小，显存几乎全是固定开销。

32M 模型在 batch 256 也只有 **4.4 GB**。

> 所以"把 batch 调到吃满 10 GB"在这套 latent 几何下没有意义：
> 要吃满 10 GB 需要 batch ≈ 1500–3000，而每域只有 1000 张训练图。
> 真正该做的是**提高 batch 到吞吐拐点（256 附近）后停止**，而不是追显存数字。

---

## 七、评测为什么慢（同样需要 infra 意见）

`eval_test600.py` 已加分段计时。单个 checkpoint（60 个任务，heun4 = 16 NFE）实测：

**generate 阶段：45.3 s**

| 阶段 | 耗时 | 占比 |
|---|---:|---:|
| DCAE encode | 9.4 s | 21% |
| **ODE 推理（16 NFE）** | **3.8 s** | **8%** |
| **DCAE decode（120 张）** | **18.5 s** | **41%** |
| **PNG 落盘（120 张）** | **11.5 s** | **25%** |
| 读图 | 2.1 s | 5% |

**score 阶段：模型加载 23.9 s + 计分 10.8 s**

| 阶段 | 耗时 |
|---|---:|
| EvaluatorModels 加载（CSD ViT-L + DINOv2 + CLIP + LPIPS） | **23.9 s** |
| 读生成图 | 3.5 s |
| CSD/DINO/CLIP 特征（120 张） | 4.1 s |
| 源图特征 | 1.7 s |
| LPIPS | 0.9 s |

**单 checkpoint 合计 ≈ 80 s。方法本身（ODE）只占 3.8 s / 80 s ≈ 5%。**

可优化点：

1. **`dcae_noop` 基线每个 checkpoint 都重算并重存一遍**，但它是 checkpoint 无关的常量。
   一次性算好缓存即可，直接省掉 60 张 decode + 60 张 PNG（约 15 s / run）。
2. **PNG 是浪费**：评测只需要像素和 CSD 特征。改存 `.npy` 或直接内存传递可省 11.5 s。
3. **EvaluatorModels 每进程加载 24 s**：把 score 与 generate 合进同一进程，或把特征提取常驻成服务，可摊掉。
4. DCAE decode 18.5 s：decode 生成图与 no-op 分开批、或用更小的解码分辨率（评测会 resize 到 224 给 CSD），可省一半以上。
5. 若只为模型选择，60 任务足够；跑满 10 个风格对（600 任务）时以上任一项都按 10× 放大。

---

## 八、容量决策：定 8M

按用户裁决，**模型容量定在 7.45M**（`width=256, depth=6, heads=4, patch=2`）。
依据（OT off、`mb=256×accum1`、2000 updates、seed 42、同一冻结 60 任务）：

| 模型 | 参数量 | 最佳 Target Top-1 | 出现在 | 训练耗时（2000 步） |
|---|---:|---:|---:|---:|
| V9-Lite | 1.89M | 15.00% | 1000–2000 | ~1.4 min |
| **V9-Mid（定档）** | **7.45M** | **23.33%** | **1000** | **~2.8 min** |
| V9 32M（mb16 旧配置） | 32.6M | 23.33% | 2000 | ~7.7 min |

**7.45M 在 1000 步就达到 32.6M 在 2000 步的水平**，训练时间约为其 1/3。
因此此前 32.6M 的选择缺乏依据，予以废弃。

> 注：32M 那一行来自 `mb16×accum4`（有效 batch 64）的旧 run，
> 与 7.45M 的 batch 256 不完全同口径；已另起一个 batch 匹配的 32M 对照
> （`Big32M_noOT`，`mb256×accum1`），评测被用户中止，未纳入本表。

---

## 九、复现命令

```bash
# 分组件训练开销
python -m sbflow.scripts.bench_components --width 384 --depth 12 --heads 6 \
  --microbatch 16 --accum 4 --ot-ratio 0.5 --ot-iters 200 --updates 100

# 同上但开提前收敛
python -m sbflow.scripts.bench_components --width 384 --depth 12 --heads 6 \
  --microbatch 16 --accum 4 --ot-ratio 0.5 --ot-tol 1e-6 --updates 100

# OT 逐层拆解
python -m sbflow.scripts.profile_ot --batches 16,64,256 --iters 50,100,200

# 采样器对比 + 真实数据收敛
python -m sbflow.scripts.profile_ot_sampler

# batch / 显存曲线
python -m sbflow.scripts.probe_batch --width 128 --depth 6 --heads 4 \
  --batches 32,64,128,256,384,512

# 定档 8M 训练（OT 关闭）
bash sbflow/scripts/train_mid_no_ot.sh Mid8M_noOT 2000 256 1
```
