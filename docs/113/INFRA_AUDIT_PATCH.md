# WEAVE 训练 / 评测 infra 审计与补丁草稿

> 只读审计，未运行任何 GPU 任务，未修改任何训练线正在使用的文件。
> 本文件为补丁**草稿**，应用前请逐条确认；所有 diff 的默认值都保持现有数值行为不变。
>
> 审计基准：`trainer.py` / `flow.py` / `output_supervision.py` / `features.py` / `utils/dataset.py` /
> `utils/run_evaluation.py` / `utils/compute_dino_metrics.py` / `run.py` / `runs/*.sh` / `config.json`。
> 目标机：RTX 3060 12G（sm_86，可用约 11G），torch 2.10.0+cu128，WSL2 Ubuntu-22.04。

---

## 0. 结论（先看这里）

1. **batch 64/96 对“纯 FM 主干”是可行的**；对“Route A 输出级监督（TAESD decode@512 + dinov2@384）”按现在
   的实现**不可行**，静态推算上限约 **16–20**。
2. **最大瓶颈不是主干，是 Route A 的“可微解码 + 教师前向”与主 loss 同图共存**。
   - 实测点：b1 = 5.26G，b14 = 9.59G ⇒ 边际 **≈0.33 GB/sample**，固定开销 **≈4.9G**。
   - 外推：`(11 − 4.93)/0.333 ≈ 18`。这与“主干只有 1.04M 参数”的直觉不矛盾——主干每样本只占约
     20–30MB，教师+解码每样本占约 200–300MB。
3. **`microbatch` 救不了 batch 上限**：`unroll_euler_decode` 先按 chunk 构图，最后 `torch.cat(imgs)` 再统一
   反传，所有 chunk 的已保存激活一直活到 backward 结束；同时 dinov2 是对**整个 batch** 做的。
   microbatch 只能压瞬时峰值，压不住 saved activations。**要真正提 batch，必须让 aux 分支的图在时间上
   与主图分开（分段 backward）或减少 aux 的样本数。**
4. **单点收益最大且风险最低的是“冻结特征离线缓存”**：`style_image` / `content_image` 的 dinov2 前向是
   **每样本常量**，现在每步重算两次 `no_grad` 前向（约占 dinov2 计算量的 2/5），离线缓存后
   **端到端约 1.4–1.7×，显存不变**。
5. `torch.compile` **不是当前主要杠杆**：Route A 打开时主干只占单步的 1–3%，编译主干的收益 < 5%，
   且与 `channels_last=true` 在代码里被显式互斥。建议延后。
6. 评测链本身已比较完善（批量/异步落盘/设备常驻都有开关），真正的浪费是 **`image_save_workers=1`、
   `save_summary_grid`、dinov2 用 fp32 且 batch 偏小**，以及评测脚本仍以“训练后串行 shell”方式跑。

---

## 1. 收益 / 风险排序表

| # | 改动 | 预期收益（量级） | 改动成本 | 风险 | 判定 |
|---|------|------------------|----------|------|------|
| 1 | 冻结特征离线缓存（content/style 的 dinov2 特征） | 端到端 **1.4–1.7×**；显存不变 | M（新脚本 + 数据接线 + 磁盘 22–45G） | 中低（需 parity 验证 bf16/分辨率一致） | **需要实测**（首选） |
| 2 | aux 分支子采样 K=8–16（每步随机抽） | batch 96 下吞吐 **≈4–6×**（按 0.122s/sample 线性缩） | S | 低-中（aux 梯度方差 ×√(B/K)） | **需要实测** |
| 3 | 分段 backward：先 `loss_fm.backward()`，再算 aux 并 `aux.backward()` | 让 batch 96 与 aux 的峰值**可相加变可取 max** | S–M | 低（梯度精确等价） | **立刻可做**（与 #2 一起才是 batch 96 的充分条件） |
| 4 | 解码器 / 教师 forward 的 grad checkpoint | aux 部分激活显存 **1.5–2.5×↓**；aux 计算 +20–40% | S | 中（重算，确定性无 dropout，可用） | **需要实测** |
| 5 | `batch_size=96` + `accumulation_steps` + `virtual_length_multiplier` 重算更新数 | 更新数/epoch 自由可调（解决 52 step / 208 update） | S（纯配置） | 低（GroupNorm 无 batch 统计，数值口径稳定） | **立刻可做** |
| 6 | DataLoader：`num_workers=2, pin_memory=true, persistent_workers` | 1–3% 墙钟（batch 越大越明显） | S（纯配置） | 低（顺序确定，已用预生成索引） | **立刻可做** |
| 7 | 每步 probe 标量节流（`probe_log_interval`） | 0.5–1.5 ms/step + 恢复 CPU/GPU 重叠 | S | 低（只影响日志粒度；默认 1 = 不变） | **立刻可做**（低优先） |
| 8 | 评测链批量化 + 落盘并发 | eval 750 图 **1.3–2×** | S | 低（必须跑 `batch_parity.sh`） | **需要实测** |
| 9 | `torch.compile`（主干） | Route A 开：**<5%**；Route A 关：**10–25%** | M | 中（冷启动 30s–8min、动态 batch 重编译、与 channels_last 互斥） | **不建议**（当前阶段）/ 关 aux 后**需要实测** |
| 10 | `channels_last` 对照实验 | ±3–8%（主干 conv/linear 混合，方向不确定） | S | 低 | **需要实测** |
| 11 | `TF32 / cudnn.benchmark / expandable_segments` | 已开启；只需固化到启动校验 | S | 低 | **立刻可做** |
| 12 | 启动即校验（空闲显存、batch、alloc_conf） | 防两条线互踩 | S | 低 | **立刻可做** |

---

## 2. 逐条审计

### 2.1 `torch.compile`

**现状**：代码里**已经有完整的开关**（`trainer.py` 第 222–234 行 + `_configure_compile`），
`training.torch_compile` / `training.torch_compile_mode` 已存在；`config.json` 里 `torch_compile=false`。
没有使用 `max-autotune`，也没有任何子模块级编译。
**并且** `channels_last=true` 与 `torch_compile=true` 被显式 `raise` 互斥（trainer.py:229-234），
因为“优化器状态保持 contiguous，而编译 kernel 可能假设 channels_last stride”。

**值得编译的模块**：
- 主干 `WEAVE`（4 个 residual block @ 64ch/32×32 + 风格 tokenizer）。张量在 B=96 下约 96×64×32×32，
  尺寸够大，Inductor 的 `GroupNorm+activation`、`conv1x1+add`、`linear+reshape` 融合有收益。
- **不建议**编译 loss 侧（DWT/`idwt2_haar` 里有 stride-2 切片 + `zeros` + 4 次赋值，Inductor 通常
  生成更差的代码，甚至 fallback）；也**不建议**编译 HF 的 `Dinov2Model`（冻结、SDPA 已融合、
  带 `interpolate_pos_encoding` 动态分支，编译收益/成本比很差）。

**模式建议**：`default`（`max-autotune-no-cudagraphs` 只适合长跑，见下）。
`dynamic=False` 可行：一个 run 内 `B_main`、`B_micro`、`t(B,)`、风格 token 数都是常量，
`drop_last=True` 保证最后一批不出现。**但** batch 扫描会触发按 B 重编译；扫描期用 `dynamic=None`
（默认 auto），定稿长跑再用 `dynamic=False`。

**必须注意的坑（静态确认过）**：
- `blocks.py` 的 `set_step()` 会在 forward 里读 `self._current_step` 这个 **Python int**。当前
  `gate_warmup_steps` 未配置（默认 0，短路后不会读该属性），所以暂时不会每步重编译；
  **一旦开启 gate warmup，Dynamo 会按 step 值重编译直到 `cache_size_limit`（默认 8）后静默回退 eager。**
  开 warmup 前必须把 step 传成 tensor 或移出编译区。
- `cfg_dropout_prob=0.0`，没有 `if mask.any()` 这类数据相关分支，图断裂风险较低。
- `last_debug` 是 forward 内写 dict 的 Python 侧副作用，`fullgraph=True` 会断，保持默认 `fullgraph=False`。

**冷启动成本估算**：主干约 200–400 个算子。
- `default`：首次编译 **30–60 s**（WSL2 上 Inductor 并行 worker，峰值内存约 2–4 GB 主机内存）。
- `max-autotune-no-cudagraphs`：每个 kernel 试 5–20 个 config，**3–8 min**；对 2000 步、约 1 h 的 run
  尚可接受，对 52 步的 run 不可接受。
- 缓存：`~/.cache/torchinductor`（默认在 WSL ext4），通常 100–300 MB。`TMPDIR` 在 WSL 常是 tmpfs，
  若把 `TORCHINDUCTOR_CACHE_DIR` 指到 `/mnt/i/...` 可跨容器复用，但 `/mnt/i` 是 9p 挂载，读写慢，
  建议只在「编译一次、多次 run」时这么做。代码里**已有** `torch_compile_cache_dir` 配置项可复用。

**结论**：Route A 打开时**不建议**；把 aux 关掉/降频后可以拿主干试 `default`，预期 10–25%。
“编译能省多少”的量级依据：主干单样本约 2 ms（≈1.1 GFLOP fwd，4 block/64ch/1024 token），
其中 launch 占比约 30–40%；融合后最多把这段降到 1.3–1.6 ms，而 Route A 单样本 122 ms。

---

### 2.2 AMP / 精度

**现状**：`use_amp=true`，`amp_dtype=bf16`，`GradScaler` **只在 fp16 时创建**（trainer.py:239-242）——正确，
bf16 不需要 scaler。

发现的问题（按影响排序）：
1. **`wavelet.py` 的 `dwt2_haar` 无条件 `x.float()`，并且 `orig_dtype` 是在 `x=x.float()` 之后取的**，
   所以返回值**永远是 fp32**，即“bf16 autocast 下 wavelet 域实际是 fp32”。这带来：每次 DWT/IDWT 都有
   bf16→fp32 的 cast 和 fp32 中间张量；`_euler_step` 的输出 `h` 也是 fp32，下一步进 conv 再 cast。
   由于 latent 只有 `B×4×64×64`（B=96 时约 6 MB），**显存影响可忽略，但每步多 4–6 个 cast kernel**。
   不建议改（改了会动数值路径），列入“不要动”。
2. **autocast 区域内的 loss 归约/统计是 fp32 的**（`flow.py` / `features.py` 里显式 `.float()`），这是对的，
   不要为了省显存把它们改 bf16。`softmax` 在 dinov2 内部由 SDPA 处理，温度没有额外缩放。
3. **反复 dtype 转换**：`_move_batch` 对 4D 张量做 `to(memory_format=channels_last)`（一次拷贝，12 MB/batch）；
   `unroll_euler_decode` 里 `img.float()`、`imgs.cat`；`features.py` 每次 `transpose(1,2).reshape().contiguous()`
   （见 2.6）。这些都不大，但 `_hf_forward` 的 **12 层全部**做 contiguous 拷贝是纯浪费（只需要 5 层）。
4. **dinov2 权重是 fp32**（`from_pretrained` 未指定 dtype，`features.py:177`）。前向在 autocast 下会按 bf16
   跑，权重每步被 cast；冻结模型可以一次性 `.to(dtype)`，但注意**这会让特征数值变化**，必须先做 parity。
   评测脚本 `compute_dino_metrics.py` 的 dinov2 是 **fp32 且无 autocast**（`load_dino` 只 `.to(device)`），
   推理没问题，但慢 2×、显存 2×。

---

### 2.3 每步同步与 CPU 开销（完整清单）

`trainer.py` 里所有会 sync 的点：

| 位置 | 频率 | 内容 | 估计开销 |
|------|------|------|----------|
| `trainer.py:1868` `_tokenizer_scalar_metrics()` | **每步** | 遍历 style tokenizer 的 `last_debug`，每个 key 一次 `.item()`（`trainer.py:1060-1095`） | ~15–40 次 stream sync |
| `trainer.py:1873` `_bridge_probe_stats()` | **每步** | 29 个固定 key + block 统计，一次 `.item()`（`trainer.py:1036/1044`） | ~30 次 stream sync |
| `trainer.py:1773` `torch.isfinite(loss).item()` | 仅 `numeric_debug` | NaN 守卫 | 1 次 sync（保留） |
| `trainer.py:1846-1853` | 每 `log_interval` / milestone | 平均指标转 float、grad_norm `.item()` | 可忽略 |
| `trainer.py:1644` `_avg()` | 每 `log_interval` | 全部 metric key 一次 `.item()` | 20 个 key × 每 20 步 |
| `trainer.py:1695-1697` `.tolist()` | 每步（**仅 Route B / pseudo cache 开启**） | `source_style_id`、`content_index` 从 GPU 取回 | 2 次 sync/step |
| `trainer.py:1148-1149` | 每步（仅 `numeric_debug`） | `.cpu().tolist()` 两个 style id | 2 次 sync |
| `trainer.py:813` `_capture_internal_probe_batch` | 每 epoch 首步 | `detach().cpu().clone()` | 1 次 sync/epoch |
| `flow.py` 的 `ll_partial_mode="wct"` 分支 | 每步（**默认 adain，不触发**） | `.float().cpu()` + `torch.linalg.eigh` 在 CPU | 会额外 1 次 sync + CPU 线性代数，**默认路径不触发** |

结论：每步约 **45–75 次 `.item()`**。这些调用发生在 backward 之后（GPU 队列已空），所以真实成本约
`60 × 15 µs ≈ 1 ms/step`，在 batch 96（≈11.7 s/step）下占比 <0.01%；但会**阻止 CPU 提前准备下一批**。
判定：**低优先，但值得节流**（见补丁 P6）。真正的大头不是 sync，而是 `_tokenizer_scalar_metrics` 每步
新建 GPU scalar（`content.new_tensor(...)`，`trainer.py:1871/1876`）造成的 allocator churn。

另外每步重算的 **DWT**：`model.forward` 里对 content/style latent 做 1 次 DWT，`flow.py` 里再对
target/content 做 2–3 次。张量只有 `B×4×64×64`，总计约 0.02 GFLOP —— **可忽略**，不属于瓶颈。
没有发现 Sobel/edge 算子。

---

### 2.4 数据管线

**现状**（`utils/dataset.py` + `config.json`）：
- latent 已经是 **per-style packed cache**，启动时一次性 `torch.load` 成 stack 放进 RAM 并 `share_memory()`；
  `__getitem__` 只做 index + 可选 hflip（**没有每样本 `torch.load`，没有每 epoch 重新解码**）。
  5000 张 4×64×64 fp32 ≈ 327 MB/路（content + target 两路约 650 MB）。
- **`num_workers=0`、`pin_memory=false`、`prefetch_factor=1`、`persistent_workers=false`**（schema 默认）。
  于是每步 `__getitem__` ×96 + `torch.stack` + 同步 H2D（非 pinned 时 `non_blocking=True` 实际仍同步）
  全在训练进程里串行执行。按 96×4×64×64×4B ≈ 6 MB / 12 MB（两路）估算，**约 3–8 ms/step**。
- 唯一的逐样本 `torch.load` 是 Route C 的 `_load_ref_tokens`：实测 data 351 s vs comp 168 s，**这才是
  数据侧的真正瓶颈**，和 `num_workers` 无关，需要改成“单文件 memmap 或 float16 全量驻留”。

**改造成 memmap 的收益**：对当前 Route A 路径 **≈0**（已经是 packed + RAM 常驻）。对 Route C 的 ref tokens：
5000×1.1 MB ≈ 5.6 GB，做成单个 `np.memmap` fp16 后可把 351 s 的 data 时间压到接近 0；若全量进 RAM 需 5.6 GB 主机内存，
要先确认机器内存。建议 memmap，不建议全量驻留。

**是否存在重复归一化 / augment**：没有。归一化在 packed 生成阶段做过；hflip 是预生成的布尔序列
（`_maybe_flip`），多 worker 下也确定，顺序不会变。

---

### 2.5 显存结构（估算，必须实测确认）

| 组成 | 占比（Route A，b14） | 说明 |
|------|----------------------|------|
| 主干参数/梯度/优化器 | < 40 MB | 1.04M 参数，`fused_adamw` 已开 |
| 主干激活 | ~20–30 MB/sample → b96 约 2–3 GB | 4 block @64ch/1024 token，self-attn 用 SDPA |
| 解码器（TAESD）@512，**带梯度** | **~150–300 MB/sample** | 512²×64ch 每张约 33.5 MB，3 个 up-stage 的 saved tensors |
| 教师 dinov2-B@384，带梯度 | ~40 MB（SDPA）– 210 MB（eager）/sample | 730 token × 12 block × 768；见下 |
| dinov2 权重 | 344 MB（fp32） | 冻结 |
| CUDA context + cuDNN workspace + allocator 缓存 | **≈4.9 GB 固定** | 由 b1=5.26 G 反推 |

**可以彻底离开训练图的教师调用**：
- `style_image` / `content_image` 的两次 `no_grad` dinov2 前向 —— 每样本常量，**可离线缓存**（P1）。
- 它们前面的两次 `decode_image`（`output_supervision.py` 中 `self.decoder.decode_image(...)`）同样是每样本常量，
  可一起缓存成 384 的 uint8 图像（约 2.2 GB）或直接缓存特征。

**梯度检查点覆盖**：
- 已覆盖：`unroll_euler_decode` 的 ODE 步（`torch.utils.checkpoint`，只包住很小的主干）。
- **未覆盖（遗漏的高激活层）**：TAESD 解码器（512 分辨率，最大头）、dinov2 的 12 个 block。
  这两处是 P4 的目标。

---

### 2.6 kernel / 内存布局

- `channels_last=true`（`config.json`），`_move_batch` 会把 4D 张量转成 channels_last（一次拷贝）。
  注意 **dinov2 / TAESD 不受影响**，所以收益只落在主干。
- `cudnn.benchmark=true`（trainer.py:225）。在 512 分辨率的解码器 + 会变化的 microbatch 下，首调用会重新
  autotune，可能选中 **大 workspace** 的算法——这部分会计入“固定 4.9 GB”的观感。建议在显存扫描时固定
  `microbatch` 并 `torch.cuda.empty_cache()` 后再读峰值。
- TF32：`allow_tf32` 已开（`set_float32_matmul_precision`）；fp32 matmul 走 TF32，符合预期。
- `expandable_segments`：**代码里没有设置**，只在 `runs/*.sh` 里通过 `PYTORCH_CUDA_ALLOC_CONF` 注入。
  直接用 `python run.py` 启动时不会生效 → 建议加启动校验（P7）。
- 不必要的拷贝 / cat：
  - `unroll_euler_decode` 最后的 `torch.cat(imgs)`：B=96 时多一份 3×512²×4B ≈ 288 MB 的常驻副本。
  - `features.py:219` 对 **12 层全部** 做 `transpose+reshape+contiguous`（只需要 5 层）：B=14 时约 190 MB 瞬时拷贝。
  - `blocks.py` 里 `dwt2_haar` 的 `torch.cat` 拼 3 个高频子带（很小，可忽略）。

---

### 2.7 评测链（`utils/run_evaluation.py` / `utils/compute_dino_metrics.py`）

- **batch 没有写死小值**：`generation_batch_size`、`metric_batch_size`、`vae_decode_batch_size`、
  `ref_feature_batch_size` 都是 CLI 参数；`config.json` 的 `full_eval_batch_size=16`，
  `runs/eval_one.sh` 已经传 32/32/16。**问题不在默认值，而在调用方**（见下）。
- **逐张落 PNG**：`save_generated_images` + `alias_png`，并且默认 `save_summary_grid=true` 会额外合成一张
  15k×15k 级的网格图。落盘走 `io_pool`，但 `image_save_workers=1` ⇒ 750 张 512² PNG 编码是**单线程**，
  约 45–60 s 的纯 CPU 时间。这是评测链最大的一笔浪费。
- **指标在哪算**：`compute_dino_metrics.py` 在 GPU 上提特征（fp32、无 autocast），然后 `generated_patches`
  搬到 CPU，**逐图** `P @ P.T`（729×729×384）和 `paper_style_distance`；750 张约 0.2–0.5 GFLOP，
  CPU 上几秒到十几秒，**不是瓶颈**。dinov2 默认 batch 8，`eval_one.sh` 已传 32。
- **能否 batch 32+ 且数值一致**：生成是确定性的（`torch.manual_seed(seed)` 全局设置，噪声总量与 batch 大小
  无关），但**必须验证**：不同 batch 下 cudnn 算法不同，512 解码会有 fp 级差异；`runs/batch_parity.sh`
  正是为此准备的，必须跑。指标是逐图计算，不受 batch 影响。
- `run_evaluation.py:3205` 每 batch 把 latent `.to("cpu", float16).contiguous()`，用于复用/落盘；这是必要的，
  但 750 × 4×64×64 fp16 ≈ 24 MB，忽略不计。

---

### 2.8 入口与编排

- `run.py` **不是** shell 硬串；训练→（可选）`_run_full_eval_for_checkpoint()` 是 `subprocess.run(..., check=True)`：
  **失败会抛出并终止训练，不会静默继续**（风险方向相反：评测坏了会连坐训练）。
  `full_eval_each_epoch=false`，`full_eval_defer_until_training_end` 默认把评测推到训练结束。
- **训练→eval→dino 的串行链在 `runs/*.sh`**（`eval_one.sh`、`p0_*.sh`、`sweep_queue.sh`），
  确实是先训完再一个接一个跑 eval，GPU 在 eval 阶段利用率低。
- **启动校验：没有**。`run.py:449` 只打印 device，不检查空闲显存、不检查 `PYTORCH_CUDA_ALLOC_CONF`、
  不检查是否已有别的进程占卡（两条线共用 3060 的现实风险）。
- **`s/step` 与 `peak_vram`：已有自动记录**。每 epoch 打印 `sec`, `sps`, `bstep`, `peak_alloc`, `peak_reserved`,
  并且 `torch.cuda.reset_peak_memory_stats()` 在 epoch 开始时重置。**注意区分 `allocated` 与 `reserved`**：
  OOM 判定要看 reserved，而“算法 workspace 是否被缓存”只看 reserved 会误判。

---

## 3. batch 64/96 可行性判断

**结论**：
- **Route A 关闭（纯 FM 目标）**：64/96 **可行**。静态估计主干 20–30 MB/sample，96 样本约 2–3 GB +
  约 1 GB 固定 + cuDNN workspace，落在 11 GB 内。`runs/mem_probe.sh` 里的 `flow_b180` 用例正是这个探针
  （结果在 `/mnt/i/.../mem2.jsonl`，本地读不到，必须上机看）。瓶颈会转移到 **CPU/数据**（`num_workers=0`）。
- **Route A 打开（现在的实现）**：64/96 **不可行**，上限约 **16–20**。原因见 §0-3：aux 图与主图同生共死，
  `microbatch` 只压瞬时峰值；dinov2 对整批做。
- **要同时拿到 batch 96 + Route A**，必须做的是 **P2 + P3（子采样 K≈8–12 + 分段 backward）**
  （可选叠加 P4 的 checkpoint）：
  - 峰值从 `fixed + B×0.33` 变成 `fixed + max(FM@96 ≈ 3 GB, aux@K ≈ 0.33K GB)`；
    K=10 时峰值约 `4.9 + 3.3 ≈ 8.2 GB`。
  - 代价：aux 项梯度方差 ×√(B/K) ≈ 3×（但均值无偏，λ 不用改）。
- **最可能的新瓶颈**：`fixed 4.9 GB` 这个“常数”到底由什么构成（cuDNN workspace / allocator 碎片 /
  两次 no_grad dinov2 的瞬时峰值）——这决定 batch 96 到底是 8 GB 还是 11 GB。**必须先拆解它**。
  其次是 CPU 侧（`num_workers=0` + 每步 45–75 次 `.item()`）。

---

## 4. 补丁草稿

> 所有补丁默认关闭/默认保持原行为，不影响正在跑的实验。
> 行号对应当前工作副本。

### P1（需要实测，首选）冻结特征离线缓存

目标：`style_image` / `content_image` 的 dinov2 特征只算一次；训练步里只保留 `img_out` 那一次带梯度的前向。

**新增脚本**：`tools/build_dino_feature_cache.py`（与 `tools/` 下现有脚本同风格）
- 输入：packed latent cache + `feature_res`（384）+ dinov2 local dir。
- 输出：`<data_root>/<style>/.dino_cache/feat_f384_{layer}.npy`（`float16`，shape `[N, 729, 768]`，按样本顺序）。
- **只缓存 `needed` 的并集层**（style 层 + content perceptual 层 + selfsim 层的 union，通常 3–4 层），
  一套缓存同时服务 content 与 target_style（两路都来自同一个 5000 张池）。
- 磁盘量级：`5000 × 4 层 × 729 × 768 × 2B ≈ 22 GB`；若 `allow_hflip` 需要额外的翻转版本 ⇒ 约 45 GB。
  若磁盘紧张，可只缓存 content 一路（约 11 GB），style 一路仍在线算，收益降到约 1.25×。
- 读侧用 `np.memmap` + 按 batch 的 `index_select`，避免一次性 22 GB 进 RAM。

**`utils/dataset.py`**：`__getitem__` 增加可选键
```python
# 在 __getitem__ 的 return dict 里（现有 "content"/"target_style" 之后）
if self._dino_cache is not None:
    out["dino_feat_index"] = idx          # int，喂给 trainer 侧缓存查找
```

**`output_supervision.py`（`forward`，第 348–392 行）** 增加可选的特征入参，跳过两次 no_grad 前向：
```diff
     def forward(
         self,
         model,
         *,
         content_latent: torch.Tensor,
-        style_image: torch.Tensor,
-        content_image: torch.Tensor,
+        style_image: torch.Tensor | None,
+        content_image: torch.Tensor | None,
         style_id=None,
         style_latent: torch.Tensor | None = None,
         style_text_tokens: torch.Tensor | None = None,
         reference_feats: torch.Tensor | None = None,
+        style_feats: Dict[str, torch.Tensor] | None = None,
+        content_feats: Dict[str, torch.Tensor] | None = None,
     ) -> Dict[str, torch.Tensor]:
         img_out = unroll_euler_decode(...)
         if self.feature_res and img_out.shape[-1] != self.feature_res:
             size = (self.feature_res, self.feature_res)
             img_out = F.interpolate(img_out, size=size, mode="bilinear", align_corners=False)
-            style_image = F.interpolate(style_image, size=size, mode="bilinear", align_corners=False)
-            content_image = F.interpolate(content_image, size=size, mode="bilinear", align_corners=False)
+            if style_image is not None:
+                style_image = F.interpolate(style_image, size=size, mode="bilinear", align_corners=False)
+            if content_image is not None:
+                content_image = F.interpolate(content_image, size=size, mode="bilinear", align_corners=False)
         ...
         feats_out = self.style_loss.extractor.forward_layers(img_out, needed)
-        with torch.no_grad():
-            feats_style = self.style_loss.extractor.forward_layers(
-                style_image.detach(), self.style_loss.layers,
-            )
-            feats_content = self.content_loss.extractor.forward_layers(
-                content_image.detach(),
-                self.content_loss.perceptual_layers + [self.content_loss.selfsim_layer],
-            )
+        if style_feats is None or content_feats is None:
+            with torch.no_grad():
+                feats_style = self.style_loss.extractor.forward_layers(
+                    style_image.detach(), self.style_loss.layers,
+                )
+                feats_content = self.content_loss.extractor.forward_layers(
+                    content_image.detach(),
+                    self.content_loss.perceptual_layers + [self.content_loss.selfsim_layer],
+                )
+        else:
+            feats_style, feats_content = style_feats, content_feats
```
**`trainer.py`（第 1729–1741 行）**：缓存命中时连解码都不做
```diff
-                    with torch.no_grad():
-                        _style_img = self._output_supervision.decoder.decode_image(target_style)
-                        _content_img = self._output_supervision.decoder.decode_image(content)
+                    if self._frozen_feats:
+                        _style_img = _content_img = None
+                    else:
+                        with torch.no_grad():
+                            _style_img = self._output_supervision.decoder.decode_image(target_style)
+                            _content_img = self._output_supervision.decoder.decode_image(content)
                     _sup_logs = self._output_supervision(
                         self.model,
                         content_latent=content,
                         style_image=_style_img,
                         content_image=_content_img,
+                        style_feats=self._frozen_feat_lookup(target_style_index, self.style_loss.layers),
+                        content_feats=self._frozen_feat_lookup(content_index, self.content_loss.layers),
```
`_frozen_feat_lookup(index, layers)` 返回 `{layer_name: [K, C, H, W]}`，直接喂给上面改过的 `forward`。

**风险**：缓存用 bf16 存、训练用 bf16 算，理论上逐位一致，但不同 batch 的 cudnn 算法可能带来
1e-3 级差异 → **必须用 `batch_parity.sh` 对比旧的 D5 指标**。若不允许任何漂移，就缓存 **uint8 的 384 图像**
（再走原前向），只省解码、不省 dinov2（收益降到约 1.1×）。

---

### P2（需要实测）aux 子采样

**`config_schema.py`（第 912–916 行附近，`OutputSupervisionConfig`/training 组）** 新增：
```python
    output_supervision_max_samples: int = 0    # 0=整批；>0=每步随机抽 N 个样本做输出级监督
```
**`trainer.py`（第 1728–1741 行，Route A 段）** 改为“只对抽样后的局部变量做 aux，不改写 `content`”：
```diff
                 if self._output_supervision is not None and (step_idx % self._output_sup_apply_every == 0):
+                    _aux_k = int(self.train_cfg.get("output_supervision_max_samples", 0) or 0)
+                    _c_aux, _t_aux, _sid_aux = content, target_style, target_style_id
+                    _tok_aux = batch.get("target_style_text_tokens")
+                    _ref_aux = batch.get("reference_feats")
+                    if 0 < _aux_k < content.shape[0]:
+                        _sel = torch.randperm(content.shape[0], device=content.device)[:_aux_k]
+                        _c_aux = content.index_select(0, _sel)
+                        _t_aux = target_style.index_select(0, _sel)
+                        if torch.is_tensor(target_style_id):
+                            _sid_aux = target_style_id.index_select(0, _sel)
+                        _tok_aux = _tok_aux.index_select(0, _sel) if torch.is_tensor(_tok_aux) else _tok_aux
+                        _ref_aux = _ref_aux.index_select(0, _sel) if torch.is_tensor(_ref_aux) else _ref_aux
                     with torch.no_grad():
-                        _style_img = self._output_supervision.decoder.decode_image(target_style)
-                        _content_img = self._output_supervision.decoder.decode_image(content)
+                        _style_img = self._output_supervision.decoder.decode_image(_t_aux)
+                        _content_img = self._output_supervision.decoder.decode_image(_c_aux)
                     _sup_logs = self._output_supervision(
                         self.model,
-                        content_latent=content,
+                        content_latent=_c_aux,
                         style_image=_style_img,
                         content_image=_content_img,
-                        style_id=target_style_id,
-                        style_latent=batch.get("target_style_latent") if torch.is_tensor(batch.get("target_style_latent")) else target_style,
-                        style_text_tokens=batch.get("target_style_text_tokens"),
-                        reference_feats=batch.get("reference_feats"),
+                        style_id=_sid_aux,
+                        style_latent=_t_aux,
+                        style_text_tokens=_tok_aux,
+                        reference_feats=_ref_aux,
                     )
```
关键点：**不要覆盖 `content` / `target_style` / `target_style_id`**，Route C（第 1751–1766 行）与
`self.model.last_reference_embedding` 仍要用整批。
说明：`mean over K` 是 `mean over B` 的无偏估计，**λ 不需要重标定**；只有梯度方差变大（约 √(B/K)）。

---

### P3（立刻可做）分段 backward

**`trainer.py`（第 1724–1793 行）**：把 aux 的 forward/backward 从“与主 loss 同一次 backward”改成两次。
真实代码里主 loss 是 `loss = loss_dict["loss"]`（第 1724 行），aux 附加项是 `_sup_total`（第 1742–1746 行）。

最小改法（1）：在第 1742 行的 `_sup_total` 计算之后插入：
```diff
                     _sup_total = (
                         self._output_sup_lambda_style * _sup_logs["loss_style_out"]
                         + self._output_sup_lambda_content * _sup_logs["loss_content_out"]
                     )
+                    if self._split_aux_backward:
+                        # 主损失（FM）先反传并释放主图，再用同一份累积梯度反传 aux
+                        if self.scaler is not None:
+                            self.scaler.scale(loss / self.accumulation_steps).backward()
+                            self.scaler.scale(_sup_total / self.accumulation_steps).backward()
+                        else:
+                            (loss / self.accumulation_steps).backward()
+                            (_sup_total / self.accumulation_steps).backward()
+                        self._sup_applied_this_step = True
                     loss = loss + _sup_total
                     loss_dict["loss"] = loss
+                    loss_dict["loss_output_sup"] = _sup_total.detach()
-                    loss_dict["loss_output_sup"] = _sup_total
```
最小改法（2）：在第 1788–1793 行的主 backward 处跳过已经做过的反传：
```diff
             t0 = time.perf_counter()
-            if self.scaler is not None:
-                self.scaler.scale(loss / self.accumulation_steps).backward()
-            else:
-                (loss / self.accumulation_steps).backward()
+            if not (self._split_aux_backward and self._sup_applied_this_step):
+                if self.scaler is not None:
+                    self.scaler.scale(loss / self.accumulation_steps).backward()
+                else:
+                    (loss / self.accumulation_steps).backward()
```
（`_sup_applied_this_step` 在 Route A 分支里置 True，Step 开头置 False。）

如果把上面 4 行抽成 `self._backward(x)` 会更干净；`scaler.step()` / `unscale_` / `clip` 仍只在
原有位置调用一次（连续两次 scaled `backward()` 共用同一个 scale，是合法的）。
**梯度精确等价**（两项相加的导数等于导数相加），唯一变化是日志里 `loss` 已包含已反传的项，
因此 `loss.detach().item()` 的结果不变（数值相同，只是图已释放）。
注意：M 级改动，需要确认 `zero_grad` 的时机、`accumulation_steps` 的边界计数、以及 `numeric_debug` 里
`loss.detach()` 的调用都还在图释放之前/之后正确工作。

---

### P4（需要实测）解码器 / 教师 grad checkpoint

**`output_supervision.py`（`unroll_euler_decode`，第 143–210 行）** 在 decode 处：
```diff
-                img = self.decoder.decode_image(h)
+                if self.decoder_checkpoint and torch.is_grad_enabled():
+                    img = torch.utils.checkpoint.checkpoint(
+                        self.decoder.decode_image, h, use_reentrant=False
+                    )
+                else:
+                    img = self.decoder.decode_image(h)
```
**`__init__`（第 311–346 行）** 增加 `self.decoder_checkpoint = bool(decoder_checkpoint)` 与
`self.extractor_checkpoint = bool(extractor_checkpoint)`；extractor 侧同理包住
`self.style_loss.extractor.forward_layers`（返回 dict，必须 `use_reentrant=False`）。
预期：aux 激活显存 1.5–2.5×↓，aux 计算 +20–40%。

---

### P5（立刻可做，纯配置）batch / 更新数 / DataLoader

不改代码，只在跑批配置里加（`_cfg_routeA_tmp.json` 那类临时配置，**不要动 `config.json`**）：
```json
{
  "training": {
    "batch_size": 96,
    "accumulation_steps": 2,
    "probe_log_interval": 5
  },
  "data": {
    "num_workers": 2,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 2
  }
}
```
更新数公式：`updates_per_epoch = floor(5000 × virtual_length_multiplier / (batch_size × accumulation_steps))`。
要达到 P0 系列同样的 2000 updates/epoch（batch 14、vlm 5.6）：`vlm ≈ 2000 × 96 × 2 / 5000 = 76.8`——数据重复太大；
现实起点建议 `vlm=15, accum=2` ⇒ 781 step/epoch、**1 epoch ≈ 1.5 万样本**、updates ≈ 781/epoch，
4 epoch ≈ 3124 updates（对比现在 208）。**这是解决“52 步就结束”的正解，和 batch 大小无关。**

---

### P6（低优先）probe 标量节流

**`config_schema.py`（第 962 行 `log_interval` 之后）**：
```python
    probe_log_interval: int = 1   # 1=每步（现状）；N=每 N 步采样一次并复用
```
**`trainer.py`（第 1868–1877 行）**：
```diff
-            token_scalar_metrics = self._tokenizer_scalar_metrics()
+            _probe_n = max(1, int(self.train_cfg.get("probe_log_interval", 1) or 1))
+            _do_probe = _probe_n == 1 or step_idx % _probe_n == 0
+            token_scalar_metrics = self._tokenizer_scalar_metrics() if _do_probe else {}
             if token_scalar_metrics:
                 ...
-            bridge_probe_stats = self._bridge_probe_stats()
+            bridge_probe_stats = self._bridge_probe_stats() if _do_probe else {}
```
默认 1 ⇒ 与原行为完全一致。

---

### P7（立刻可做）启动校验

**`run.py`（第 449–451 行，`device` 之后）** 插入：
```diff
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     logger.info("Device: %s", device)
     logger.info("Seed: %d", seed)
+
+    if device.type == "cuda":
+        _free, _total = torch.cuda.mem_get_info()
+        _used = (_total - _free) / 1024 ** 3
+        logger.info("GPU memory before init: free=%.2fG / total=%.2fG (used=%.2fG)",
+                    _free / 1024 ** 3, _total / 1024 ** 3, _used)
+        if _free < 9.5 * 1024 ** 3:
+            logger.warning("Only %.2fG free; another run may be using this GPU. "
+                           "Peak VRAM numbers will be polluted.", _free / 1024 ** 3)
+        if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
+            logger.warning("PYTORCH_CUDA_ALLOC_CONF is unset; expandable_segments will not be enabled.")
```
（`import os` 已存在；不改变任何训练行为。）

---

### P8（需要实测）评测链

**`runs/eval_one.sh`（第 21–28 行的参数块）**：
```diff
-  --image_save_workers 1 \
+  --image_save_workers 4 \
   --generation_batch_size 32 \
   --metric_batch_size 32 \
-  --ref_feature_batch_size 16 \
+  --ref_feature_batch_size 32 \
+  --no-save_summary_grid \
```
`--no-save_summary_grid` 只在“不看网格图”的批量 run 上加；正式出图/附录的 run 保留。
若只看指标不看图，再加 `--no-save_generated_images --no-alias_png`（需确认 ArtFID/KID 未启用）。
另：`run.py` 的 `full_eval_lpips_chunk_size` 默认 4，显存富余时可提到 16（`run.py` 里传给评测子进程）。

`utils/compute_dino_metrics.py`：把 `--batch_size` 从 8 提到 64（纯推理、`inference_mode`），
**但保持 fp32 与 `dino_model_name` 不变**——改成 bf16 会让 DINO-S 指标产生 1e-3 级漂移，不能和既有主表混用。

---

### P9（不建议，先放着）torch.compile

开关已存在，**不需要新代码**；只在实验配置里：
```json
"training": { "torch_compile": true, "torch_compile_mode": "default", "channels_last": false }
```
注意必须同时 `channels_last=false`（trainer.py:229-234 会 raise）。
若确实要保留 channels_last，需要改 `trainer.py:229-234` 的互斥逻辑并单独验证优化器状态 stride；
**不建议**在没有收益证据前动它。扫描 batch 时用 `dynamic=None`，定稿长跑再 `dynamic=False`。

---

## 5. 必须上机实测的点（静态无法判断）

1. **固定开销 4.9 GB 的构成**：分别测 `flow-only @b1/b14/b24` 与 `RouteA @b1/b14` 的
   `peak_allocated` / `peak_reserved`，每次配置前 `empty_cache()`；确认是 allocator 缓存、
   cuDNN workspace 还是两次 no_grad dinov2 的瞬时峰值。
2. **单步耗时分解**：`main fwd+bwd` / `decode@512` / `dinov2 img_out` / `dinov2 style+content` /
   `data load`。`fractions` 决定 P1 和 P2 的真实收益（上面 0.122 s/sample 的拆分是估算）。
3. **dinov2 实际 attention 实现**：`model.config._attn_implementation` 是 `sdpa` 还是 `eager`，
   以及 `output_hidden_states=True` 是否真的保留了 13 份 hidden state。eager ⇒ 每样本多约 150 MB，
   收益巨大（只需 `attn_implementation="sdpa"`）。
4. **`microbatch` 是否真的降峰值**：`microbatch=1/4/8/14` 四档在**同一 batch** 下测峰值；
   若峰值基本不变，就证明“chunk 图未释放”，P3 是唯一解。
5. **评测 batch 32/48 vs 2 的数值一致性**：`runs/batch_parity.sh`（生成像素 diff + D5 指标 diff）。
6. **P1 缓存特征与在线特征的差异**：同图 bf16 缓存 vs 在线 bf16 前向，`max|Δfeat|` 与最终 D5 指标差。
7. **`num_workers=2` 的实际收益**：对比 epoch 日志里的 `data=` 与 `comp=`；若 `data` 已经很小，说明瓶颈全在 GPU。
8. **`channels_last` 开关对照**：同一 batch、同一步数，比较 `bstep` 与 `peak_reserved`。
9. **dinov2-B 在 3060 上的实测 TFLOPS**：用于校准“compile 能省多少”和“P4 重算代价”的量级。
10. **WSL 上 Inductor 的冷启动与内存**：`TORCHINDUCTOR_CACHE_DIR`、`TMPDIR`、编译峰值主机内存；
    这些在 3060/WSL2 上必须实测，静态无法给准数。

---

## 6. 建议的执行顺序（不改动在跑的训练线）

1. 先在**旁路目录**用 P5 的配置跑 `batch 96 / accum 2 / vlm 15` 的 **flow-only** 短跑（100 step），
   同时读 §5-1 的显存分解 → 确认纯主干的 batch 上限。
2. 用 §5-4 确认 microbatch 无效后，再上 **P2 + P3**（aux K≈10 + 分段 backward），保持 `feature_res=384`。
3. P1（离线特征缓存）最后做，因为它需要生成 5–11 GB 缓存并做 parity；但对**长跑总时间**收益最大。
4. P8 只改 `runs/eval_one.sh` 的副本（例如 `eval_one_fast.sh`），不要覆盖正在被队列使用的脚本。

---

## 7. 最终补丁包（可应用）

### 7.1 文件

| 文件 | 状态 | 作用 |
|------|------|------|
| `docs/113/patches/infra_20261104.patch` | 新增 | 统一 diff，**以此为准**（本文 §4 的早期草稿已过时） |
| `tools/apply_infra_patch.py` | 新增 | 精确 old→new 替换：`--check` / `--apply` / `--emit-diff` |
| `frozen_feature_cache.py` | 新增 | P1 运行时 memmap 查找（不可用时自动降级在线路径） |
| `tools/build_dino_feature_cache.py` | 新增 | P1 离线构建（decoder + extractor 各跑一遍） |
| `runs/eval_one_fast.sh` | 新增 | P8 评测链，`eval_one.sh` 保持字节不变 |

### 7.2 应用方式（二选一）

```bash
# 方式 A：git apply（推荐显式关闭 autocrlf，否则会把 run.py 全文件 LF→CRLF）
git -c core.autocrlf=false apply --check docs/113/patches/infra_20261104.patch
git -c core.autocrlf=false apply          docs/113/patches/infra_20261104.patch

# 方式 B：替换执行器（保留各文件原有的 CRLF/LF，逐字节一致）
python tools/apply_infra_patch.py --check     # 19 个锚点必须各命中 1 次
python tools/apply_infra_patch.py --apply
```

注意：`trainer.py` / `config_schema.py` / `output_supervision.py` / `utils/dataset.py` 在本工作树里是 **CRLF**，
`run.py` 是 **LF**。补丁 diff 按各自原始字节生成，因此 `core.autocrlf=true` 会让 `run.py` 变成整文件改动
（内容等价但 diff 噪声极大）。方式 B 不受该设置影响。

**锚点快照（重要）**：补丁锚点是按 2026-11-04 当前工作树生成的。`trainer.py` 的 probe 段已被另一条线
改写成 `probe_due` 结构，且该文件当前 **`probe_interval` 与 `probe_accum` 都没有定义**（一旦执行到
probe 段就会 `NameError`，与本次补丁无关，是对方在写的半成品）。P6 只补上
`self.probe_log_interval` 与 `probe_interval = self.probe_log_interval`；`probe_accum` 属于对方在写的代码，
不纳入补丁。若他们的改动继续推进，先跑 `--check`，锚点失配就重新生成，不要硬套。

### 7.3 本机已验证（未启动训练）

| 检查 | 命令 | 结果 |
|------|------|------|
| 锚点唯一性 | `python tools/apply_infra_patch.py --check` | OK，19/19 |
| 干净临时树上应用 | `git -c core.autocrlf=false apply --check`（空仓库副本） | rc=0 |
| 语法 | `python -m py_compile`（5 个被改文件） | rc=0 |
| P1 缓存查找 | 合成 memmap（2 风格 × 2/3 图 × flip 行） | `rows` 计算、越界/未知风格返回 `None`、`feature_res` 不匹配自动禁用，全部通过 |
| P1 数值等价 | `OutputSupervision.forward(style_feats=..., content_feats=...)` vs 在线路径（TinyConv） | `loss_style_out` / `loss_content_out` **\|Δ\| = 0** |
| P3 梯度等价 | `(fm+aux).backward()` vs `fm.backward(); aux.backward()` | `max\|Δgrad\| = 0` |

默认配置（`config.json` 原样）的行为：

- 新增开关全为 `False`/`0`/`""`，P1/P2/P3 分支不进入；
- Route A 块的调用点与参数不变，只是搬进了 `_output_sup_step()`（内含等价的 autocast 上下文）；
- 主 backward **没有任何改动**（§4 草稿里那版 `_sup_backward_done` 守卫已废弃）；
- 除 P7 的两条 warning 外无新增日志/同步点。

---

## 8. 应用顺序、收益与风险

按“改动风险从低到高、收益从确定到不确定”排序：

| 顺序 | 项 | 配置键 | 预期收益 | 主要风险 |
|------|----|--------|----------|----------|
| 0 | 补丁落地 | — | — | 训练线文件若正在被写，先停队列再 `--check` |
| 1 | **P7 启动校验** | 无（run.py） | 0 性能；避免“显存被别的进程占着”污染峰值与 OOM 归因 | 仅两条 warning，无行为变化 |
| 2 | **P6 probe 节流** | `probe_log_interval=4` | probe 已按新结构存在，补丁把它接到配置项；每步省掉 `_tokenizer_scalar_metrics` / `_bridge_probe_stats` 的 Python+小 kernel 开销（约占总步时的 1–3%，需实测） | 指标变成每 4 步采样一次；日志对照时注意 |
| 3 | **P8 评测链** | `runs/eval_one_fast.sh` | 评测 wall 约 **1.5–2.5×**（PNG 4 线程 + 关 grid + dino 32→64） | 丢 `summary_grid.png`；dino batch 64 需约 2–3 GB，评测机空闲时要看 P7 |
| 4 | **P4 extractor checkpoint** | `output_supervision_extractor_checkpoint=true` | aux 教师激活大约减半（是 P1 之外唯一能压 `img_out` 那一路的手段） | aux 前向 **+20–35%** 时间；只在 aux 段生效 |
| 5 | **P2 aux 子采样** | `output_supervision_max_samples=16` | aux 显存/时间按 `K/B` 缩小；b96 时 K=16 只有 1/6 | aux 梯度方差 ↑ `sqrt(B/K)`（风格分布量级，K≥8 一般可接受） |
| 6 | **P3 分段 backward** | `output_supervision_split_backward=true` | 峰值从 `主图 + aux 图` 变为 `max(主图, aux 图)`，实测前估计省 0.3–1.0 GB | 依赖 allocator 归还块；必须配 `expandable_segments:True`；两次 `backward` 共用同一 scaler（语义正确） |
| 7 | **P1 特征缓存** | `output_supervision_frozen_feats_dir=...` | aux 段消掉 **1 次 decode@512 + 2 次 dinov2 前向**（约占 aux 时间 40–55%、aux 激活 30–45%） | 需先建缓存（22 GB / 带 flip 45 GB）；fp16 缓存与 bf16 在线的 `max\|Δ\|` 必须实测；数据池变动要重建 |

P1 打开后仍会保留 **content 侧一次 decode**：`w_edge > 0` 的 Sobel 边缘项确实需要像素。
若某次实验把 `output_supervision_w_edge` 设为 0，则可以连 content decode 一起省（本补丁已允许
`content_image=None`，只是默认配置用不到）。

### 8.1 P9：`torch.compile` 保持关闭

- 现状 `torch_compile=false`，且 `channels_last=true` 与 compile 在 `trainer.py` 里显式互斥。
- 静态估算 compile 收益 `< 5%`：主干 1.04 M 参数、32×32 latent，单 step 主图计算约 2 ms，占 122 ms/sample
  的 1.6%；能编译的部分本来就不是瓶颈，而 dinov2 是冻结的 HF 模型（`no_grad` 段），Inductor 收益更小。
- 结论：**不开**。若未来主图占比上升（>15%）再考虑，且必须先解决与 `channels_last` 的互斥和 WSL 冷启动
  （`TORCHINDUCTOR_CACHE_DIR`、编译期主机内存）。

---

## 9. 待 GPU 线执行的验证清单

以下 6 项静态无法判定，全部在**旁路配置**上执行，不碰正在跑的训练线。

### 9.1 4.9 G 固定开销的 allocated/reserved 构成

```bash
# 用现有探针脚本，分别跑 flow-only 与 RouteA，各 batch 档位之间先 empty_cache
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash runs/mem_probe.sh
# 同时在 trainer 里临时打印（不要提交）：
#   torch.cuda.memory_allocated() / memory_reserved() / max_memory_allocated() / max_memory_reserved()
python -c "import torch; print(torch.cuda.memory_summary())"   # 取一步结束后的快照
```

期望：`reserved - allocated` 稳定在 1.0–2.0 GB（allocator 池 + cuDNN workspace + CUDA context）；
若 `b1` 的 `peak_allocated` ≈ 4–5 GB 而其中参数/优化器只占 ~20 MB，则固定开销主要来自 context/workspace，
`expandable_segments` 应能把 `peak_reserved` 压低 0.5–1.5 GB。**若压不动，b96 一票否决。**

### 9.2 单步耗时分解

```bash
bash runs/routeA_scan_1b.sh     # 读日志里的 comp=/data=/bstep=/sps
```

期望：`aux(dinov2 3 次前向 + 2 次 decode) ≥ 60%` 的 `comp`；`data < 10%`。
若 dinov2 占比不到 50%，说明 P1/P2 的收益上限比本文估计的要低，先做 9.3。

### 9.3 dinov2 是 sdpa 还是 eager

```bash
python -c "
from features import build_feature_extractor
e = build_feature_extractor('dinov2', device='cpu', local_dir='<weights/dinov2-base>')
print(e.model.config._attn_implementation)
print(type(e.model.encoder.layer[0].attention.attention).__name__)
"
```

期望 `sdpa` + `Dinov2SdpaSelfAttention`。若为 `eager`，仅加 `attn_implementation="sdpa"` 就能省
约 15–25% 的教师时间与 100–150 MB/样本激活，优先级高于 P4。

### 9.4 `microbatch=1/4/8/14` 峰值是否真的不变

```bash
for MB in 1 4 8 14; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OUTPUT_SUPERVISION_MICROBATCH=$MB bash runs/mem_probe.sh
done
```

期望：四档 `peak_allocated` 差值 `< 0.05 GB`（证明 `torch.cat` 后统一 backward、chunk 图不释放）。
若确实不变，**microbatch 应从配置里删掉**，用 P3 替代。

### 9.5 评测 batch 32 vs 2 的 parity

```bash
EVAL_BATCH=2  bash runs/eval_one.sh  …/ckpt.pt /tmp/eval_b2
EVAL_BATCH=32 bash runs/eval_one.sh  …/ckpt.pt /tmp/eval_b32
python -c "
import json
a=json.load(open('/tmp/eval_b2/results_split.json')); b=json.load(open('/tmp/eval_b32/results_split.json'))
d=max(abs(a[k]-b[k]) for k in a if isinstance(a[k],(int,float)))
print('max|delta| =', d)
"
```

期望：生成像素 `max|Δ| ≤ 1/255`（fp32/bf16 的 ODE 累加误差），D5 指标 `max|Δ| < 1e-4`。
若超差，P8 只保留 `image_save_workers=4`，batch 维持现状。

### 9.6 缓存特征 vs 在线特征 `max|Δ|`

```bash
# 先只在 16 个样本上建缓存，验证通过再全量
python tools/build_dino_feature_cache.py --config config.json --out /tmp/feat_probe \
  --limit_per_style 4 --batch_size 8 --amp_dtype bf16 --flip_rows --overwrite
# 对比脚本（旁路，读取缓存 index.json 后与在线 forward_layers 对齐比较）
python -c "
import torch, json
from frozen_feature_cache import FrozenFeatureCache
c = FrozenFeatureCache('/tmp/feat_probe', feature_res=384, layers=['block6'])
print('enabled', c.enabled)
# 取 rows 后与 e.forward_layers(img, ['block6']) 比 max|Δ|
"
```

期望：`--amp_dtype bf16` 时 `max|Δ| ≤ 2e-3`；`--amp_dtype none`（fp32 缓存）时 `≤ 1e-5`。
超过 1e-2 说明缓存构建的 resize/归一化与训练路径不一致（先查 `feature_res` 与 `interpolate` 的 `align_corners`）。

---

## 10. “b96 + 监督分支同时开”的可行性判断

**最省事的组合确实是 P1 + P2 + P3 一起上**（三者都不改模型结构、都不碰主图前向，只动 aux 段与反传顺序）。
但**结论是：方向正确，b96 仍不保证能进 11 G；b64 才是现实目标。**

用现有实测（b1 = 5.26 G、b14 = 9.59 G ⇒ 固定 ≈ 4.9 G、边际 ≈ 0.33 GB/sample）拆解：

| 组合 | 峰值估算（`expandable_segments` 生效） | 判断 |
|------|----------------------------------------|------|
| 现状 b14 + 监督 | 9.59 G（实测） | 可跑 |
| b32 + 监督（无补丁） | 4.9 + 32×0.33 ≈ 15.5 G | **OOM** |
| b96 + 监督 + P1（仅缓存） | 4.9 + 96×0.20 ≈ 24 G | **OOM** |
| b96 + P1+P2(K=16)+P3 | 4.9 + 96×0.08 + 16×0.20 ≈ 15.7 G | **OOM（大概率）** |
| b64 + P1+P2(K=8)+P3 | 4.9 + 64×0.08 + 8×0.20 ≈ 11.6 G | **边缘，需实测** |
| b96 只有主图（监督全关） | 4.9 + 96×0.06 ≈ 10.7 G | 大概率可行 |

关键前提（任一条不成立，上表的数字就要下修）：

1. `0.33 GB/sample` 里 **教师占比 ≥ 60%**（即主图边际 ≤ 0.13）。P1 之后边际才会落到 0.06–0.08 量级；
   这个“P1 后的边际”是唯一决定性的未知量，验收命令：9.1 + 9.2，且 P1 开启时取 b8/b16 两档斜率。
2. P2 只压 aux 那一路的**样本数**，不压主图；所以 b96 时 aux 仍以 `K × 单样本教师激活` 计（与 B 无关）。
3. P3 只在 allocator 真把 FM 图归还时才有收益；`expandable_segments` 必开，否则峰值仍是两图之和。
4. 缓存 dtype 用 fp16、`feature_res=384` 必须与训练一致；dual-target lerp 行会整批回退在线路径（不报错，但变慢）。

**结论**：

- 若目标是“b96 + 监督”，P1+P2+P3 是**必要条件**，但**不是充分条件**：即使三者全开，主图在 b96 下自己就要
  ~10.7 G，叠加任何 aux 都会越过 11 G。要真正 b96+监督，需要再动主图（P0 类结构改动：扩散块 grad checkpoint /
  减少 residual / 降低 `feature_res`），**不在本补丁范围内**。
- 现实路径：**b64 + P1 + P2(K=8) + P3**，并用 9.1/9.2 拿到的“P1 后边际”复核；若边际 > 0.09，退到 b48。
- 不确定的地方已标注：主图边际、教师占比、allocator 归还行为都需要 9.1–9.4 的实测数字，本文给的是区间估计而非承诺。

---

## 11. 一句话总结

补丁包默认全关、默认数值逐位不变（已用合成用例验证 P1 损失与 P3 梯度 `|Δ|=0`），落地顺序为
**P7 → P6 → P8 → P4 → P2 → P3 → P1**；P9 保持关闭；b96+监督在现有结构下大概率仍 OOM，
把目标定在 b64 并把 9.1/9.2 的实测边际作为最终决策依据。

