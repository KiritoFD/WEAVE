# WEAVE V9-LD · RTX 4070 Laptop / 6 GiB 可用显存 / ETA=12h

**版本：2026-11-14。性质：工程实施与预注册实验计划，不是已跑通的程序。**

## 0. 结论与变更边界

**原则：保留 LD0–LD5 科学消融；把“单机完整模型链”改为“冻结 N2C latent 缓存 → 低显存局部优化 → 离线 VAE 解码 → 单独指标计算”。** 3060 用于 StyleGallery 复现，不能占用。4070 Laptop 物理显存 8GB、用户报告**实际可用约 6GB**；设备可能因桌面、显示和其他 CUDA 进程动态波动。**以 `torch.cuda.mem_get_info()` 实测 free 为准**，不把“4070 算力较好”转换成未经测量的速度预测。

- 实验总预算：**单卡 4070 Laptop 累计 GPU/任务 wall-clock 12 小时**；前 5h45m 做诊断和开发，**后 6h15m 不被训练侵占，专门留给正式生成、解码、评测和交付**。
- 冻结：B4、N2C (`step_04000.pt`)、R1、VAE/评测权重；不改旧结果、旧任务顺序或训练集划分。
- 算法：从 N2C 的 `z_N2` 出发，实验局部作用范围、真实局部画派能量下降方向、内容梯度投影、逐样本硬预算和回溯线搜索。**无额外大型模型**。
- LD5 小门控训练是条件分支，**不是必跑项**。若 LD1–LD4 无正结果或测速预算不足，不训练 LD5。
- 本次**不会复刻 StyleGallery**；StyleGallery 独立在 3060 复现。只借鉴动态局部匹配；SMS 借鉴实际频带内容约束，不加载 SD LoRA；AD 借鉴目标局部特征方向，不加载扩散 UNet。

## 1. 实验基线与口径

| 工作点 | Cross600 DINO-S ↑ | DINO-C ↑ | LPIPS ↓ | Target Top1 ↑ | G ↑ | 来源 |
|---|---:|---:|---:|---:|---:|---|
| N2C.e | 0.435755 | 0.782994 | 0.496922 | 0.328333 | -0.017893 | 3060 正式管线 |
| F2.s1 | 0.433092 | 0.754548 | 0.512673 | 0.291667 | -0.019359 | 4070 复建 |
| F2.e | 0.434422 | 0.738995 | 0.527628 | 0.306667 | -0.017893 | 4070 复建 |
| F2.s4 | 0.437128 | 0.707789 | 0.558992 | 0.308333 | -0.013820 | 4070 复建 |

**严格要求**：不同机器的 VAE kernel 和缓存重建可能影响小数点后三位。本地 `F2_LOCAL_EVAL.md` 已报告复建 VAE 解码 MAE 约 8.6/255，**不可声称是可忽略的“浮点舍入”**。从 3060 搬运官方 `runs/v8/N2C.e/gen/batch_XXXX.pt` 中 `z_src,z0,z_final`，本地统一解码 `LD0` 和 `LD1–LD5`，再比较本地成对结果。历史 3060 N2C 的数值只作外部核验锚点。

**原有接口可复用：** `src/infer/run.py` 的每批 `.pt` 含 `ids,z_src,z0,z_final`；`exp/v8/guidance.py` 含 `SPDHead,target_margin,project_rms`；训练画作池和 `exp/v8/style_dist.py` 提供真实风格数据来源；`src/eval` 提供独立指标。优先拷贝**750 个任务的 N2C latent**（数据量很小），不得重新运行昂贵 B4/N2C 生成来填满每个臂。若确实没有完整缓存，设单独缓存构建阶段，并把耗时从 12h 总预算扣除。

校验项：每批任务 ID 与 `eval_protocol/tasks_750.json` 顺序完全一致；所有样本和参考库按源图/来源路径做 SHA 检查；`z_final` 与同一任务的 `z0,z_src` 形状相同；150 个源图/750 任务计数正确。**不可拿历史 F2 在 4070 的差异替代 LD0 本机重评分。**

## 2. 6 GiB 内存架构：三进程、零大模型共存

| 阶段 | GPU 上加载什么 | 明确不加载什么 | 建议 batch | 降级顺序 |
|---|---|---|---:|---|
| A. 构建风格 Patch Bank | 初版可完全 CPU；必要时仅临时张量 | B4、N2C、R1、VAE、DINO/CSD | CPU 256–2048 patches/chunk | 全部转 CPU |
| B. LD1–LD4 latent 编辑 | **R1 + N2C SPDHead + 小型局部特征/原型**；`z` 为唯一可微输入 | B4、VAE、DINO/CSD、CLIP、LPIPS | **从 B=4 起测**，尝试 8/16；OOM 退至 2/1 | B=4→2→1；删候选缓存 |
| C. 输出 VAE 解码 | SD1.5 VAE FP16/bf16（选择已验证数值稳定的精度） | R1、SPD、生成器、指标模型 | B=2 起测；OOM→1 | `vae.enable_slicing()`，必要时小块处理 |
| D. 质量评测 | **单个**评测网络，例如 DINOv2 / CSD / CLIP / LPIPS，逐模型加载 | R1、生成器、VAE、其他评分模型 | 先 B=2，调到 4/8 | 对全量输出使用同一 pipeline |
| E. LD5 门控训练 | R1 + SPDHead + 风格原型 + 轻量 CNN；**不保留二阶计算图** | B4、VAE、评测网络 | **microbatch=2–4**，必要时 1 | AMP、截断一阶反传、累积 |

**安全线**：若启动时可用显存约 6.0 GiB，PyTorch 模型进程峰值 `max_memory_reserved()` 尽量 **≤4.8–5.0 GiB**，留约 1 GiB 应对系统波动、CUDA context、碎片与短峰值。该阈值是工程安全余量，不是模型已测占用。首阶段测 `torch.cuda.mem_get_info`、`max_memory_allocated`、`max_memory_reserved`、nvidia-smi 进程总使用量；不要只看张量 allocated。若实际 free <5 GiB，强制 B=1，并择时停止不必要桌面 GPU 进程；不可偷偷减图像分辨率而仍称 512 全量。

**关键 GPU 内存规则**：
1. GPU **只常驻一份** R1；`critic.parameters()`、SPD 权重、门控训练以外的所有参数 `.requires_grad_(False)`；`autograd.grad(loss,z)`，不能对冻结模型权重 `.backward()`。
2. 近邻匹配的索引 `detach`，**距离函数不能 detach**。局部 Haar 描述符采用 fp32 归一化/距离计算；UNet 本轮不参与前向。
3. 每步 `g_R`、`g_local`、`g_content` **分别求梯度、立即 detach**，不同时保留三套计算图；回溯候选全部 `torch.no_grad()`；完成一个 batch 立刻落盘 CPU。
4. 若显存吃紧，绝不同时持有 `style_bank [全部训练 patches]` 和所有候选 `z_cand`；只把 `[5,128,D]` 原型置 GPU，每次候选只存 score/best，保留单个 best latent。
5. 不调用 `torch.cuda.empty_cache()` 试图修复真实 OOM；应减少 batch、释放引用/重启对应阶段的独立进程。解码、评测每阶段使用新 Python 进程，避免缓存和 attention hooks 残留。

### 2.1 低显存下训练 batch/updates 重新配置

| 项 | 3060 旧设计 | 4070 Laptop 6GiB 设计 |
|---|---|---|
| 零训练推理 `batch` | 常见 8 | **从 4 探测**：允许 1/2/4/8/16，选满足显存上限的最大高吞吐配置；各臂统一 |
| LD5 `microbatch` | 未定/较大 | **默认 4；OOM→2→1**，优先一阶截断 |
| LD5 `effective batch` | 原计划训练 1000–2000 updates，未限定 | **32**（微批 4 → 累积 8；微批 2 → 累积 16） |
| 优化器 step | 依赖原 V8 代码 | **明确写真正的梯度累积循环**；每个微批 loss 乘 `n_micro/32` 再反传，累积完成后只 `optimizer.step()` 一次 |
| LD5 最长更新 | 1000–2000 | **软目标 300–800 updates；1h15m 硬停**，根据 P0 采样实测重算，不保证固定更新数 |
| 原型 K | 每风格 128 | 仍为 **128**；约 5×128×32×4B=80 KiB，不是 VRAM 瓶颈 |
| 局部单元 | 8×8 latent patch →64 个区域 | 不变；逐样本 64 个 Patch，与 batch 几乎线性增长 |
| 推理 GD | K=1/2/4 | **优先 K=2，保留 K=1 低延迟备选；K=4 只做少量诊断，不进默认全量** |

**累积实现检查**：必须保证 optimizer 每收到累计 32 个**独立训练样本**才更新一次，不能仅改 `MICROBATCH` 环境变量导致学习率/有效样本数变化；如果最后一个微批不足 32，就除以真实累积样本数。AMP 时使用 `GradScaler`（fp16），clip_grad_norm 在 `unscale_` 后执行。验证 micro=4/accum=8 和 micro=2/accum=16 在固定 32 样本上的 loss/参数更新近似一致；运行顺序造成的浮点差需记录。

## 3. 核心方法与上一版保持一致，但更节省 GPU

### 3.1 动态局部画派能量（训练画作先验）

输入 latent 维度 `(B,4,64,64)`；拆成不重叠 `8×8` latent patches，共 64 个局部单元，采用 Haar 的 LL 平均/方差 + 高频三方向 std + 少量方向/色度统计，维数 D≈24–32。**所有 Patch 使用统计量而不是原始视觉内容复制**。训练池每类提取、全局归一化后，CPU 聚类为 K=128 原型/风格、记录对应训练图路径和 seed 以审计。内存中的全部真实训练样本无需驻留 GPU。

局部原型从目标画派**任意位置**动态匹配，与之前失败的 C2 固定 (x,y) 原型不同。每一步最近邻 top-4 stop-grad，能量保留可微：

\[
E_{t,i}(z)=-\tau\log\frac{1}{4}\sum_{p\in NN_4(f_i(z),\mathcal P_t)}\exp\left(-\frac{\|f_i(z)-p\|_2^2}{\tau}\right)
\]

\[
L_{loc}(z,t)=\operatorname{mean}_i\left[E_{t,i}+\beta\operatorname{softplus}(\mu+E_{t,i}-E_{\neg t,i})\right]
\]

只有在**训练池留出真实画作**上确认 5×5 能量具有目标选择性，并且 16 个输出 latent 的小步长有限差分实测下降后，才允许该方向参与完整 LD3/LD4。若完全无目标识别性，立即停止原型分支，防止再得到“风格统计有效优化但实际画派错误”的伪结论。

### 3.2 三种编辑方向和局部选区

以缓存的 `z0`、`z_N2` 为输入，起点 `z=z_N2`：

\[
g_R=P_{N2}\nabla_z m_{R1}(z,t),\qquad g_L=-\nabla_z L_{loc}(z,t),\qquad h=\nabla_z C_{LL}(z,z_0)
\]

- `C_LL`：二级 Haar 低频结构变化（必要时加本地可微自相似度），不在内循环使用 DINO、CSD、CLIP。
- 三方向：`R1 only`、`Local only`、`0.5 R1+0.5 Local`；先 RMS 归一，避免风格 loss 尺度不同。
- 将 `z` 分成 64 个 8×8 格，按预计局部风格下降 / 内容损伤风险选 top25% 区域（备选 50%）；对 mask 轻微平滑，再执行一次正交投影，确保不会因为平滑重新引入一阶内容风险。
- **GPU 省时路径**：先给每方向计算一阶局部收益/风险，仅保留前 2 个方向进行精确回溯；回溯系数先 `{0.5,1.0}`（每步最多 4 次前向局部能量/R1 评分），仅在 dev 证明显著增益且实际吞吐允许时开放完整 3×3 候选。

### 3.3 逐样本硬预算的 Test-Time Gradient Descent

与 N2 对照保持相同参考：`z0=B4 output`，不是从 `z_N2` 重新计算预算。不得把 `z_N2` 的位移当成“零额外代价”。

\[
\mathrm{cost}_{tot}(z)=\frac{\operatorname{mean}(z-z_0)^2}{\max(\operatorname{mean}(z_0-z_s)^2,10^{-8})}
\]

**主候选 eps_total=0.10、备选 0.12、0.15**；若起始 N2C 已经超出该样本预算，原样返回，标注 `already_over_budget`。空间结构额外约束的阈值须根据**独立开发集**中 N2C 的 `C_LL` 分布固定；不要为了做出目标分数在 test750 扫阈值。

每次优化 K∈{1,2}，主 K=2；候选 `z'=project_rms(z+eta*d, z0, rho)` 必须**同时**满足 `cost_tot<=eps`、结构阈值，以及代理目标改善。`argmin` 的代理包含真实局部能量与 R1 的适度辅助项；无可行候选则回滚上一 latent；不能继续用旧 `eta*` 的“最大可行步长”策略。候选选择器**不得读取独立评测指标**。

**同风格 Same150**：若 `source_style_id == target_style_id`，默认不追加局部优化，返回原 `z_N2`；必须显式记录 identity short-circuit 及其影响，不能将它当成无条件方法提升。

## 4. 6 臂消融矩阵与数据集

| Arm | 修改 | 主要反证 |
|---|---|---|
| LD0 | N2C 原始 latent，**在 4070 同管线重评分** | 本地锚点、解码差异 |
| LD1 | N2C + 额外 R1 更新，K=2、同硬预算 | 多算两步本身是否提升 |
| LD2 | LD1 + 局部 mask，仍用 R1 方向 | 仅局部性是否有效 |
| LD3 | N2C + 局部目标真风格梯度，**全域**更新 | 仅真实风格方向是否有效 |
| LD4 | 局部 mask + R1 / Local / 混合方向选择 + 实际回溯 | 组合是否超过单因子 |
| LD5（可选） | LD4 的方向混合与 mask 用 `<100K` 参数 CNN 学习 | 学习门控是否超过手工选区 |

**对照必须一致**：LD1–LD4 使用同一 K、输入 `z_N2`、hard eps、dev 测试任务、相同选择数据；不得让某个臂仅因为较大的搜索预算而胜出后不报告运算开销。LD4 多方向的额外计算计入真实推理速度。`LD5` 只在 LD4 已有正收益且值得学习时训练。

开发数据必须按源图排重：优先使用 **V6 独立 val200**，通过 source SHA/文件路径核对与 Test750 无重叠。历史报告发现 val200 泛化乐观，故按目标画派分桶并做额外 holdout 交叉检查。若无法找到独立开发集，允许在 Test750 按源图另切开发/最终评估，但必须**重新命名实验协议**、明示不再等同原 Cross600/Same150 盲测，不能用同一批数据反复搜索再称之为冻结 test750。

## 5. 12h 重算：先测速，再动态排队

**预算总计 12h，不允许 P0–P4 占用最后 6h15m 的正式生成窗口。** CPU 路径的原型聚类、磁盘准备和报告可与 GPU 小实验并行，不纳入 GPU 占用但要记录 wall-clock。

| 时间窗 | 用时 | 内容与真实交付 | 硬退出条件 |
|---|---:|---|---|
| `T+0:00–0:30` | **0.50h** | P0 检查可用显存、迁移 N2C 750 latent 缓存、任务 ID/模型签名校验；3 个 warmup + 16 样本 profile（R1/Local/候选/解码/指标分别计时） | 若缓存或独立评分不一致，优先修环境；**不开始训练** |
| `T+0:30–1:15` | **0.75h** | P1 CPU 建库 K128、训练画作留出 5×5 能量矩阵、16 样本有限差分/梯度方向符号测试 | local gradient 无效→只做 R1 局部化诊断，不训练门控 |
| `T+1:15–3:30` | **2.25h** | P2 LD1/2/3/4 先各跑 dev50（约 200 task outputs）；筛选至多 2 臂跑额外 dev100；同管线快速指标和内容风险分析 | 按**实测每图 wall 秒**控制样本数，LD2/LD3 消融优先于大规模搜参 |
| `T+3:30–4:45` | **1.25h** | P3 **有条件**训 LD5；micro=4→2→1，effective batch32，200/400/800 updates 动态截止（不是硬承诺） | LD4 若无明显收益或预计完不成 P5，则**跳过 LD5**，改为修方法/补消融 |
| `T+4:45–5:45` | **1.00h** | P4 独立 dev 只调最有效臂 K=1/2、区域比例 0.25/0.5、eps .10/.12（.15 只做必要对照）；冻一个最优配置 | 只选**一个**正式候选 + LD0，冻结所有超参和 reference bank |
| `T+5:45–12:00` | **6.25h** | P5 正式候选 Test750（750 tasks），LD0 本地同管线对照重评分、五类/同风格/速度/显存/CI、图表与报告 | **最后 30min 强制停止新生成**、汇总已有产物；失败报告不得伪装全量完成 |

### 5.1 时间可行性公式（不得拍脑袋估算）

必须在 P0 用 16 张 **真实的最重 LD4 配置**，warmup 3 张后计算平均每图秒数 `t_gen`，另外单独实测 16 张解码 `t_dec` 和各模型评测合计 `t_eval`。P5 保守占用 6h15m=22500 秒，其中预留汇总 1800 秒，装载/落盘和失败重试保守再留 1800 秒，则：

\[
\boxed{750\,(t_{gen}+t_{dec}+t_{eval}) \leq 22500-1800-1800=18900\ \mathrm s}
\]

即**合计不超过 25.2 s/image**。若 LD0 已缓存且两臂均在相同本机解码/评测管线，可复用 LD0 原输出指标；正式候选新生成只做 750 张，不能把 N2C 整套模型再运行一次。

- 预测 `≤12 s/img`：可用完整 `K=2, dirs=3, etas=3` 若经 dev 确认有效，另留充分余量。
- `12–22 s/img`：优先用 `K=2, dirs≤2, etas≤2`；尽可能保留 750 全量。
- `22–25.2 s/img`：以 dev 上已验证的 `K=1` 或轻量候选组合为正式配置，并明确其与较重消融不同；P5 不再增加额外实验臂。
- `>25.2 s/img`：12h 内不保证完成 750；必须在 P4 前下调**计算量但不暗改分辨率/指标定义**，若仍不达标则按已完成任务数报告 incomplete，不能假报完成。

**训练预算同样按实测计算**：`t_update = (32/micro)*t_microstep + t_optimizer`；若 LD5 的 800 updates 会超过 1.25h，自动减少 updates 或直接跳过。梯度累积只减少峰值显存，**不减少所需样本总量和 FLOPs**，在小 microbatch 下可能反而降低吞吐。

### 5.2 microbatch 探测与计时记录

探测只使用 **同一个分支、固定 16 张**，分别计每项峰值显存与 img/s：

```text
GPU free_now, total, other_process_usage
batch 1 / 2 / 4 / 8 / 16:
  peak_allocated, peak_reserved, OOM/NaN?, forward/backward_ms, img_per_s
  LD1 step, LD3 local gradient, LD4 candidate search
train LD5 micro 1 / 2 / 4:
  peak_reserved, microstep_ms, optimizerstep_ms, accumsteps, effective_batch
separate decode/eval:
  decoder ms/image, each metric ms/image, peak_reserved
```

12h 预算运行时每隔 10 分钟更新 `remaining_gpu_sec` 和 `projected_finish_sec`，若预计完成时间超过剩余时间，按优先级**先砍 LD5，再减开发扫描，最后采用 dev 已验证的低成本 GD 配置**。绝不通过对测试集结果调超参实现“冲刺”。

## 6. 工程结构与实施步骤

**已有模块**（复用）：
- `src/infer/run.py`：task750 latent 输出 `z_src,z0,z_final`；
- `exp/v8/guidance.py`：`SPDHead,target_margin,project_rms`；
- `exp/v8/energy.py`：旧 R1 支持能量及负结果对照；
- `exp/v8/style_dist.py`：V6 训练风格库来源；
- `src/eval`：DINO/CSD/LPIPS/CLIP/G；
- `src/utils/vae.py`：本地 FP16 VAE fallback，需核对数值协议。

**计划新增**（仓库中当前不保证存在；下述命令必须实现后才能运行）：

```text
exp/v9_local/
    cache_n2c.py             # 从 runs/v8/N2C.e/gen/batch_*.pt 导入只读 z_src,z0,z_N2
    patch_bank.py            # 训练池 CPU streaming; 5*128*D prototypes; leakage audit
    local_style.py           # 8x8 patch Haar features; dynamic differentiable target energy
    direction.py             # R1/SPD + local gradients; content gradient; region scores
    solver.py                # 2-way shortlist, projected 2-eta line search, hard cost
    run.py                   # LD0-LD4; batch auto-probe; atomic resume; no VAE inside
    gate.py / train_gate.py  # LD5 optional; true grad accumulation, truncated first-order
    decode_and_eval.py      # sequential decoder then per-network metrics; reuse WEAVE evaluator
    tests/                   # grad signs, cached equivalence, budget, mask, task id, OOM
sbflow/scripts/run_v9_local_4070_eta12.sh
runs/v9_local_4070/
    cache/ bank/ dev/ test/ logs/ profile/ eval/
docs/v9_local/
    RESULTS_4070.md results.csv pareto.svg failure_register.md
```

**建议入口（待实现，并非已经能复制运行）**：

```bash
python -m exp.v9_local.cache_n2c --src runs/v8/N2C.e/gen --out runs/v9_local_4070/cache
python -m exp.v9_local.patch_bank --source train --prototypes 128 --device cpu
python -m exp.v9_local.run --arm LD4 --split dev50 --batch 4 --max-memory-gib 5.0
python -m exp.v9_local.run --arm LD4 --split test750 --batch 4 --resume
python -m exp.v9_local.decode_and_eval --split test750 --arm LD4 --batch 2
```

V8 结果路径可能在另一台机器上：**必须先复制原始 `.pt` 或其完整缓存到 4070；不可在本机凭文件名虚构缓存**。同时保存 ckpt SHA、task manifest SHA、style bank SHA、seed、micro/accum/precision、peak VRAM、每张实际 end-to-end inference time。

### 6.1 本地 4070 的 CUDA/WSL 注意项

1. 使用独立 Python 进程，避免浏览器与训练软件争抢显存；先检查 `nvidia-smi` 和 `torch.cuda.mem_get_info()`，记录动态空闲情况。
2. 仅在 **方向梯度** 上启用 `torch.enable_grad()`；不要用外层 `torch.inference_mode()` 包住整个 LD 推理再试图开启 autograd（inference tensor 会有梯度兼容问题）。
3. `autograd.grad(..., create_graph=False, retain_graph=False)`；若两个损失需要同一 graph，重建便宜的 Haar 特征而不是长期 retain 大图；局部最近邻 stop-grad，保留局部距离对 z 的梯度。
4. GPU VAE 不与 R1 同驻；metrics 必须逐网络独立执行。本地 6GB 若 CSD ViT-L 单独仍超限，使用 CPU 分批评分，但照实记录耗时与设备，不能改评测模型。
5. 正式推理计时采用**同卡热启动**的实际端到端 image/latent pipeline；从 3060 复制预计算 `z_N2` 可用于算法消融，但仅计“LD 增量耗时”时必须明确该口径，不能将它标成整条 WEAVE 的推理时间。要与外部 Baseline 比拼完整时延，需另外同卡实测 `source encoding + B4 + N2C + LD + decode`，不能把两个机器实测片段简单相加。

## 7. 判停、安全性与统计

1. 正负方向有限差分必须通过：`L_loc(z-η grad L_loc)<L_loc(z)` 小步长、每画派真实留出图的目标能量低于错类；不满足就暂停 LD3/LD4。
2. `mask=0` 或硬预算无可行候选必须**逐样本精确返回 z_N2**；`already_over_budget` 原样返回并计数。含平滑 mask 的方向要重新投影内容梯度，不能只做一次。
3. LD1–LD4 dev 只选符合硬内容预算且本地局部风格代理下降的样本；拒绝样本不删除、不得只对成功子集汇报平均。
4. 只留必要缓存，计算中间特征不跨批累积到 GPU；当 peak_reserved>5.0GiB/OOM，缩小 batch，必要时重启阶段进程，不偷偷关闭内容约束以省内存。
5. Same150 显示短路数量和保真指标；Cross600 分五个目标风格、20 个 source→target 方向统计 Top-1、G、DINO-S、DINO-C、LPIPS，给 paired bootstrap 95% CI。阈值在开发集冻结。
6. **最低接受门槛（Cross600）**：DINO-S≥0.442、DINO-C≥0.780、LPIPS≤0.50、Top-1≥0.325、G≥-0.016，且 Same150 不明显退化。**强突破**：DINO-S≥0.445、DINO-C≥0.79、Top-1≥0.35，至少 4/5 目标风格受益；仍为目标，不是承诺。
7. 任何从 test750 中挑参数得到的分数不能宣称未见测试；若 12h 超时或缺失任务，所有均值附真实 `n_success/750`，不伪报完整。

## 8. 结果→下一阶段决策

| 结果 | 判读 | 下一步 |
|---|---|---|
| LD2>LD1，LD3≈LD1 | 空间选区有价值，真实特征方向无增量 | 改画派原型/局部语义匹配 |
| LD3>LD1，LD2≈LD1 | 真实风格梯度有效 | 继续改负样本/原型与权重；不强训 gate |
| LD4>LD2 且 >LD3 | 选择**正确局部方向**是可发表的机制 | 进一步研究 LD5 低开销门控；突出效率 |
| LD5>LD4 | 学习式局部选择有增量 | 继续改轻量局部风格表示 |
| 均≤LD1 或 DINO-C 大幅恶化 | 该局部特征/代理失败，**不是**证明 V6 网络容量不足 | 转向真正可训练的多尺度空间残差生成器 |

**本轮最后一句话：4070 6GB 不能照搬 3060 的“大 batch + 一边生成一边解码评测”。利用既有 N2C latent 缓存，将推理优化压缩为“小 R1 + Haar Patch Bank + SPD”，让 microbatch 和梯度累积只出现在条件训练的 LD5，才能以 12 小时可靠检验“局部性 × 正确方向 × 测试时梯度下降”这一核心假设。**
