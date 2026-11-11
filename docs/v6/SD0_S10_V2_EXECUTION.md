# SD0-S10 V2：执行方案与 P0 基线

**日期**：2026-11-11
**状态**：全部组件实现并通过自测；串行链 `run_sd0_s10.sh` 已拉起，P1 Warm 训练中。
**目标**：完全冻结官方 `sd-vae-ft-ema` Encoder，在**原版未缩放 latent** 上训练 10.658M S10 Decoder，48h 预算，交付通用高保真 SD1.5 兼容 Decoder。

---

## 1. 固定边界（实测确认）

| 项 | 值 |
|---|---|
| Encoder | `/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema`，`config.json` sha256 `92d3dfb7...` |
| Encoder / quant_conv | 全程冻结，`requires_grad_(False)`，哈希前后一致（单测 PASS） |
| Teacher | 官方 49.49M Decoder，`eval()` + 无梯度，输出 detach |
| Student | `widths_forward=[64,128,192,256]`，**10,657,923** 参数 |
| latent | `z = E_SD0(x).latent_dist.mean`，**未缩放**；`scaling_factor=0.18215` 仅用于外部生成流水线 |
| `post_quant_conv` | 属 Decoder 接口，随 Decoder 训练；`StudentDecoder` 单测确认**只应用一次** |
| 分辨率 | 512×512，LPIPS@512 |

### SD0 与 A0 latent 关系（实测 16 张）
```
cos(SD0, A0) = 0.9956   RMS ratio A0/SD0 = 0.9960
```
比方案引用的 0.992/0.998 更接近 → **支持 Warm-A0 是有价值初始化**，不能预设 Cold 更正确。

### 剪枝索引的重要修正
最初尝试把 A0 的 `channel_index` 数字直接套到官方 SD0 Decoder 上，得到灾难性结果（LPIPS 0.808）。核实发现：
- `s10/decoder.pt`（decoder_build_s10.py 的 Taylor 剪枝）与 `s10_p1_l1/decoder.pt`（decoder_train.py 内重剪）的索引**重叠仅 39–74%**，是两次独立剪枝
- Warm（`s10_p5_lowlr`）继承自后者

**最终 Cold 构造**（`build_cold_sd0.py`）：在**官方 SD0 Decoder 上重新计算 Taylor 重要性**（用 SD0 latent + SD0 pqc 校准），再 `prune_decoder` 继承权重 → `cold_init.pt`。这才是「按原 S10 方法剪枝」的正确含义。

Cold 零步 LPIPS 0.824 曾疑似 bug，但对照测出 **A0 自己的未训练 Taylor 剪枝在同一批 SD0 latent 上是 0.848** —— 确认这是「结构化剪枝后未经恢复」的正常状态，正是 P1 要追平的对象，不是构造错误。

---

## 2. 数据（P0 完成）

用官方 SD0 Encoder **重新编码**（旧 A0 缓存作废，未用 RMS 缩放转换）：

| 划分 | 数量 | 文件 |
|---|---:|---|
| train（`in_train=True` 且非 val） | 78,661 | `runs/sd0_s10/full/latent_train_fp16.bin` (2.40 GiB) |
| holdout（冻结 val） | 2,048 | `latent_holdout_fp16.bin` |
| test256（独立，与 train/holdout 不重叠） | 256 | `latent_test256_fp16.bin` |

`split.json` 校验记录：
```
image_id_disjoint: true
bad_rows: 三组 zero=0 nonfinite=0（全扫，非抽样）
reencode_cos_min: 0.999950 (128 张随机重编码)
encoder_determinism_max_abs: 0.0
```
注：缓存首次运行在重编码校验处因 CPU/GPU 设备不一致崩溃；已修并加 `--verify-only` 复用已编码文件，未重复 31 分钟编码。

---

## 3. P0 零步基线（同协议 SD0，不套用 A0 历史值）

| 指标 | official SD0 | W (Warm-A0) | W-PQC | C (Cold-SD0) |
|---|---:|---:|---:|---:|
| L1 | 0.07152 | 0.07162 | 0.07165 | 0.50437 |
| PSNR | 24.72 | 24.89 | 24.89 | 10.76 |
| SSIM | 0.7539 | 0.7577 | 0.7576 | 0.2267 |
| **LPIPS@512** | **0.07836** | 0.09340 | 0.09348 | 0.82449 |
| CSD Keep | **0.96093** | 0.87593 | 0.87611 | 0.17905 |
| HF corr L0 | 0.42465 | 0.42538 | 0.42519 | −0.01623 |
| **HF 能量比** | 0.98562 | **0.69415** | 0.69553 | 0.27688 |
| 颜色误差 | 0.00205 | 0.00517 | 0.00515 | 0.29366 |

**结论**：
1. **W-PQC 与 W 几乎相同**（LPIPS 差 0.00008）→ 输入映射差异可忽略，按方案删除该候选，不安排完整训练
2. **HF 能量比 0.694 确认系统性幅值衰减存在** → 方案允许在 C0 开启弱能量正则 λ_e；但仍以逐系数 Haar 监督为主
3. Cold 的 0.824 是未恢复剪枝状态，P1 让 W/C 各跑 800 步同配置对照后再选起点
4. 这是**同协议 SD0 基线**，后续验收不得套用 A0-S10 历史的 LPIPS 0.0926 / CSD 0.8535

---

## 4. 组件与测试

| 文件 | 作用 |
|---|---|
| `sd0_cache.py` | SD0 latent 缓存 + 全套完整性校验，含 `--verify-only` |
| `sd0_losses.py` | 两级 Haar DWT（逐系数 L1，FP32）、固定子带尺度、弱能量正则、三类扰动（多尺度相关高斯/真实插值/后验式）、响应蒸馏 `SmoothL1((Δy_S−Δy_T)/(a_δ+ε))`、off-manifold 教师一致性、`prune_with_indices` |
| `build_cold_sd0.py` | 在官方 SD0 上 Taylor 剪枝生成 `cold_init.pt` |
| `p0_baseline.py` | 三候选零步审计 → `p0_baseline.json` |
| `train_sd0_s10.py` | P1/C0/C1 统一训练循环，梯度标定、断点续训、best/last 双存 |
| `eval_sd0_s10.py` | 四层评测中的干净重建、扰动/Jacobian（区分 J^T v 与完整 Loss 输入梯度）、Infra |
| `test_sd0_s10.py` | 单测：Encoder 哈希不变、单次 pqc、wavelet 零值/形状/梯度、δ=0 响应≈0、教师无梯度、学生输入梯度有限非零 |
| `run_sd0_s10.sh` | 48h 串行总控，单 GPU 队列、阶段可跳过、Gate 失败保存负面报告 |

**自测全部 PASS**（CPU/GPU）；Gradient-free 与梯度标定数值：`hf_grad_ratio≈11–13 → λ_h≈0.011–0.014`（对应 15% 目标）、C1 `resp_grad_ratio≈0.86 → λ_r≈0.17`、`λ_o=0.075`。

### 修掉的显存/工程问题
- bs=8 在 P1 阶段 OOM（21.88 GB）→ 默认 bs=4（effective batch 仍 64，accum=16）
- C1 teacher 两次前向 + 16 个 accum 学生图叠加 → response 分支移到 accum 循环**之外**、2 张独立小批、教师输出立即释放；bs=4 实测 13.8 img/s 无 OOM
- P1/C0 阶段不加载 teacher（C1 才需要），省 49.49M 常驻
- CUDA generator 不能用于 `torch.randint` → 改用 Python RNG 选扰动类型/幅度
- trainer 输出名 `{stage}_{init}_*` 与 Gate 读取名不一致 → 总控统一 `cp` 成标准名

---

## 5. 训练阶段（预算上限，可提前收敛/因 Gate 失败停止）

| 阶段 | 步数 | 主要内容 |
|---|---:|---|
| P1 | W/C 各 800 | 同数据/Loss/优化器/评测对照，选初始化；不引入 Jacobian |
| P2 C0 | 3000 | Charbonnier + LPIPS@512(0.5) + λ_h·Haar 逐系数（λ_e 默认 0）+ 可选弱能量 |
| P3 C1 | 2500 | 加响应蒸馏，扰动三类（50/30/20）、两档幅度（小 0.02/0.05，中 0.10/0.20）、25% 更新进双分支 |
| P4 | 可选 | 仅当 C0/C1 仍有明显风格保持差距，单一冻结感知网络小权重消融 |
| P5 | 验收 | 四层评测 + 生成器替换 |

**Gate 设计**：P1 起点 Gate（等后续成本比较）、C0 HF Gate（LPIPS 不恶化 + HF 改善 + 无锐化伪影 + CSD/颜色保持）、C1 响应 Gate（响应误差降 + 梯度一致性改善 + 干净重建保持 + 大扰动无伪影）。Gate 失败保存 `NEGATIVE_REPORT.json`，不自动宣称成功。

---

## 6. 本轮补齐的组件（训练进行中并行完成，未打断串行链）

训练在跑时改动 `run_sd0_s10.sh` 会破坏 bash 的字节偏移读取，因此新增组件一律用**独立文件**，由 `run_p5_final.sh` 串起；`guard_p5.sh` 在训练链退出后自动拉起它。

| 文件 | 状态 |
|---|---|
| `eval_generator_swap_full.py` | **第三层完整实现**（不再是骨架）。固定官方 SD0 Encoder、只换 Decoder；显式 `×sf → generator(scaled) → ÷sf → 未缩放 Decoder` 转换（生成器在缩放空间训练，Decoder 必须吃未缩放 latent）；五生成器 Oracle1/B4/B3/ScoreDiff N12/N24；CSD Top-1+Margin、DINO-S、CLIP-S、Style Affinity、DINO-C、LPIPS、**CLIP 目标风格选择性**（`Δsim(target)−max_other Δsim`，全局共同抬升归零）单独报告、五目标各自变化；按源图聚类配对 Bootstrap（保留重复簇）。已编译通过、**离线依赖全部核实**，但**未实跑**（等 C0/C1 checkpoint）。 |

**第三层离线依赖核实（4090 无外网）**：
- `CLIPModel` / `CLIPProcessor`（`openai/clip-vit-base-patch32`）必须指定 `cache_dir=/root/Workspace/WEAVE_VAE/eval_cache/hf/hub`，实测加载成功（151,277,313 参数）
- `facebook/dinov2-base` 在**默认** `~/.cache/huggingface`，指定 `eval_cache` 反而报 `LocalEntryNotFoundError` → 已改为不指定 cache_dir，实测 OK
- `canonicalizer_stats.pt` 由 `vae_swap_full.py` 在 `SDVAE = sd-vae-ft-ema`（即官方 SD0）空间上构建，与本轮 latent 空间一致（存的是 `ll/lh/hl`  wavelet 统计，键为 `ll_mean/ll_std/...`）
- `CLIPProcessor` 返回 `BatchFeature`，已修为取 `["pixel_values"][0]` 再 stack，否则 `torch.stack` 会失败
- 生成器权重、R1 critic、ScoreDiff、750 任务协议、csd_proto 全部在位
- **仓库内部 API 契约逐一核对（纯 CPU/静态）**：`ConditionalScoreNet(base_dim=64)` → `cond_dim=128`，与 checkpoint 的 `style_emb(5,128)` / `style_mlp.0(128,128)` 吻合（Q2a 是 `base_dim=16`→`(5,32)`）；`C3UNet(base_dim=16, mode="direct", inject_16=True, style_keys=False)` 与 `vae_swap_full.py` 一致；`edit_with_score_difference(model, z_src, src_ids, tgt_ids, num_steps, step_size, struct_anchor_weight, seed)` 参数名与仓库自身调用一致；`canon(z, mode="c1_gentle", augment=False)` 签名匹配；`csd_model.feats_pils(model, pils, batch, device)` 签名匹配
- **空间契约确认**：`eval_src_latents.pt` RMS 0.707（缩放空间）；`vae_swap_full.py` 用 `src_z = z*sf` 送生成器、`raw = zgen/sf` 送 Decoder，与本脚本 `z_scaled = zs*sf` → `z_unscaled = zgen/sf` 完全一致。所有变体吃同一批未缩放 SD0 latent。
| `export_sd0_s10.py` | Diffusers 兼容独立包装器导出 + 验证。用 `cold_init.pt` 实测：standalone wrapper 与训练包装器输出 **max\|diff\| = 0.000e+00**。导出含 `config.json`（Encoder 哈希、IO/latent scaling 约定、结构参数、数据信息）与无仓库依赖的 `decoder.py`。 |
| `test_sd0_s10_extra.py` | 补齐方案要求的单测：BF16/FP32 一致性（相对差 0.02328）、断点恢复（optimizer 动量/scheduler/下一步权重完全一致，diff 0）、数据覆盖审计（三组计数/无重叠/行对齐）、Validation 不共享训练图。**全部 PASS**。 |
| `test_c1_graph.py` | 纯 CPU 验证 C1 隔离响应分支图拓扑（不占 GPU、不打断串行链）：两次 backward 不冲突、梯度累加、教师无梯度、连续两 update 干净、`corr_gauss` 真实 4×64×64 形状/RMS 正确。**全部 PASS**，消除「隔离版未经实测」风险。 |
| `run_p5_final.sh` | P5 补跑总控：main+extra 单测 → 四层验收 → 第三层生成器替换 → 导出 → **Gate 事后审计** → 汇总 `P5_FINAL_REPORT.json`。不修改正在运行的总控。 |
| `audit_gates.py` | 补齐方案要求但运行中总控 Gate 未覆盖的判据：C0 的 **CSD Keep / 颜色不明显恶化**（从 `c0_metrics.jsonl` 读 colour，Gate 文件本身没有该字段）、C1 的**梯度方向一致性改善 / 大扰动无新增伪影**（从 `eval_p5_final.json` 的 `perturb.medium` 与 `input_grad.jtv_cos_median` 读）。「无锐化伪影」需人工看视觉网格，脚本如实标 `None` 而不臆断。 |

### 仍需用户决策/后续执行
1. **TensorRT 实测**未做（Infra 目前只测 PyTorch forward / forward+input-grad）。
2. **P4 感知特征校准**是可选阶段，总控与 P5 脚本均未编排（需人工依据 C0/C1 的 CSD Keep 差距决定是否执行）。
3. `eval_generator_swap_full.py` 与 `run_p5_final.sh` 尚未实跑；`guard_p5.sh` 已在运行并将在训练链正常结束后自动执行 P5（GPU 空闲时）。若需提前人工触发，可手动 `bash tools/vae_suite/run_p5_final.sh`。
4. 旧的 `eval_generator_swap.py`（骨架）保留作接口参考，以 `eval_generator_swap_full.py` 为准。
5. ~~C1 隔离版响应分支尚未在 GPU 上实测~~ → **已用 `test_c1_graph.py`（纯 CPU，不占 GPU）验证图拓扑全部通过**：重建 backward 后响应分支建全新图、`l_extra.backward()` 不报 "backward a second time"、梯度正确累加而非覆盖、教师无梯度、学生梯度有限非零、δ=0 响应≈0、**连续两个完整 update 都干净**、`corr_gauss` 在真实 4×64×64 上形状与 RMS 比（0.0500）正确。同时给 `make_perturbation` 加了 `C==4` 断言（生产 latent 本就是 4 通道，防止未来误用非 4 通道输入时 `posterior` 分支静默广播错）。该测试已加入 `run_p5_final.sh`。**注意**：这验证的是图管理与数值逻辑，C1 在 512px 真实显存占用仍需 C1 启动第 4 步（首次 `upd%4==0`）实跑确认不 OOM。
6. `p1_W_best.pt`（旧进程产物）的 meta 缺 `encoder_hash` 字段；C0/C1 新进程写入的 checkpoint 均带该字段，最终交付（best_recon/response/overall 来自 C0/C1）不受影响。
7. **`eval_sd0_s10.py` 的 P5 实跑前静态修复**（此前未 GPU 实跑，本轮 CPU 验证）：① `infra()` 被 `@torch.no_grad()` 装饰但 `forward+grad` 分支要 `.backward()`，必然崩溃 → 去掉装饰器、按模式选 `no_grad`/`enable_grad`、显存统计兼容 CPU，CPU mock 实跑两模式通过（2.21ms / 0.38ms）；② `perturb_metrics` 的 student 前向未包 `no_grad`，750 任务循环会累积 autograd 图泄漏显存 → 与 teacher 一起包进 `no_grad`（input-gradient 段自行建图）；③ `old_s10` 保留 A0 训练的 pqc（历史对照的刻意混淆变量），c0/c1 用 SD0 pqc，meta 里已记 `pqc_source`。
8. **C0 Gate 交接已用真实数据预演**：`c0_W_metrics.jsonl`（upd 250/500/750）带全部 Gate 依赖字段（`lpips/hf_corr_L0/hf_l1/csd/colour` + `train_*`），真实三条预演 `passed=True`（LPIPS −0.00199、HF corr +0.00570）。总控 `INIT=W` → `c0_W_metrics.jsonl` → `cp` 成 `c0_metrics.jsonl` → Gate 读该文件，路径闭环。
9. **C1 Gate 的 `resp` 是累计均值语义**：trainer 的 `stage_sum/stage_cnt` 从 upd1 累加到每个 VALID 点，不是最后 250 步的分段均值；因此 `last<=first+0.01` 偏保守，真实下降也可能被早期高值拖住而判失败。`audit_gates.py` 已补 `resp_trajectory` + `resp_is_cumulative_mean=True` 显式标注，避免把 Gate 失败误读为「响应变差」；按方案 Gate 失败时保留 C0 并写负面报告，不会丢权重。
10. **整链现已能自主跑到 P5 完成**：新增 `guard_p5.sh`（`flock` 单例 + `pgrep` 锚定 `^bash tools/vae_suite/run_sd0_s10.sh`，用 `setsid nohup` 拉起，自身不占 GPU）。它在训练链退出后检查 `SERIAL PIPELINE COMPLETE` 或 `FINAL_REPORT.json`，成立才自动执行 `run_p5_final.sh`；若链异常结束则打印日志尾部并 `exit 3`，**不对半训练权重做评测**。已实测守护存活、锚定匹配只命中真正的 chain（不误匹配守护自身）、flock 锁生效。
11. `run_p5_final.sh` 原先各步 `|| echo WARN` 会把失败吞掉仍返回 0。已改为 `mark_fail/mark_ok` 累积，最终写进 `P5_FINAL_REPORT.json` 的 `failed_steps` 与 `status`（`complete`/`incomplete`），并打印 `P5 INCOMPLETE — failed steps:...`；权重始终保留。失败累积逻辑已在 bash 子 shell 验证（`a c` 正确收集，OK 项不误入）。
12. **第三层「DINO-S 目标选择性」修正为真实按风格度量**：原 `dino_selectivity` 只是「生成图对源图 DINO 相似度 − 前 32 张源图 DINO 相似度均值」，与全局相似度高度重合，无法区分「目标风格真涨」与「所有风格一起涨」。已改用冻结测试集里的 5 个单位范数 CLIP 原型（512 维，`vae_style_reference_subset.pt` 的 `clip_prototypes`），定义 `clip_target_selectivity = Δsim(target) − max_other Δsim(other)`，其中 Δsim 是生成图相对**源图 round-trip** 的原型相似度变化。CPU 数值验证：目标独涨 → 正、全局同涨 → 0、错误风格涨更多 → 负。该指标进入 `per_target` 五风格分解。顺带删掉了读 `vae_style_reference_subset.pt` 却从未使用的死代码，并修掉一处会导致 `KeyError` 的残留 `acc["dino_selectivity"]`。`task_id`（750 个全唯一）、`csd_proto` 五风格键、`clip_prototypes` 五键均已用真实数据核对。
13. **交付契约实证核对**：真实 `c0_W_best.pt` 的 meta 含 `encoder` / `encoder_hash`（config + safetensors sha256）/ `widths_forward` / `latent_convention`（未缩放）/ `needs_post_quant_conv` / `stage` / `upd` / `init`，且 `state_dict`(140 keys) + `post_quant_conv` + `opt` + `sched` 齐全。导出包装器的 `config.json` 完整覆盖方案要求的全部字段：`params=10,657,923`、结构参数、Encoder 标识+哈希、IO 数值范围（`latent (B,4,64,64)` / `RGB (B,3,512,512)≈[-1,1]`）、latent scaling 约定（`scaling_factor_reference=0.18215`，仅外部流水线用）、训练数据信息（78,661/2,048/256）、pqc 说明。`export_sd0_s10.py` 的 `verify()` 已改为 CPU 可用，并用**真实 128MB 训练 checkpoint** 纯 CPU 导出验证：standalone 包装器与训练包装器 **max\|diff\| = 0.000e+00**。两个中期导出目录已清理，P5 会用最终 c0/c1 权重重新导出。同时确认 `build_decoder` 与 standalone 都把 `widths_forward` 原样传给 `block_out_channels`（diffusers `Decoder` 内部自行 reverse），无顺序问题。
14. **部分失败不再丢交付（plan: 评测脚本失败不丢权重 / Gate 失败保存负面报告）**：原 `guard_p5.sh` 在链无 `SERIAL PIPELINE COMPLETE` 时直接 `exit 3` 拒跑 P5 —— 若 C1 阶段 OOM 导致链异常退出，即使 C0 已完整训练、`c0_best.pt` 存在，也会零交付。已修：链异常结束但有 `c0_best.pt` 时写 `NEGATIVE_REPORT.json`（`CHAIN_INCOMPLETE_C1_OR_LATER`）、标 `.chain_status=2`，仍拉起**保守 C0-only P5**（`run_p5_final.sh` 的 MODELS 本就按文件存在自动构造）；若连 `c0_best.pt` 都没有但有 `c0_W_best.pt`（链死于 C0 中途）则自动 `cp` 恢复；两者皆无才 `exit 3`。`P5_FINAL_REPORT.json` 会记录 `chain_status=2` 与 `status=degraded_chain_incomplete_c0_delivered` 及 `NEGATIVE_REPORT.json`。**两条分支已在隔离沙箱 `/tmp` 实测**（有 C0 → 写负面报告+跑 P5+`.chain_status=2`；无 C0 → `EXIT=3` 且 P5 未运行），未触碰真实产物。旧 guard 已 kill 并用新脚本重启（PID 689377），chain 未受影响。

---

## 7. 当前状态

```
[SERIAL] 0 compile + selftests      ALL PASS
[SERIAL] 1 SD0 cache                verified (78661 / 2048 / 256)
[SERIAL] 2a Cold init               cold_init.pt
[SERIAL] 2b P0 baseline            p0_baseline.json
[SERIAL] 3_p1_W                    DONE 56.4min  LPIPS 0.09060 / CSD 0.85726
[SERIAL] 3_p1_C                    DONE 56.6min  LPIPS 0.12819 / CSD 0.79743
[SERIAL] 4 P1 Gate                 chose W ("Cold 持续落后则用 Warm")
[SERIAL] 5_c0                      running 1/3000, 14.4 img/s (~3.7h)
                                   init-from p1_W_best.pt, teacher=None
```

**P1 起点对照（同数据/Loss/优化器/评测，各 800 步）**：

| | W (Warm-A0) | C (Cold-SD0) |
|---|---:|---:|
| LPIPS@512 | **0.09060** | 0.12819 |
| CSD Keep | **0.85726** | 0.79743 |
| HF corr L0 | **0.45455** | 0.39597 |
| PSNR | **25.25** | 24.40 |
| SSIM | **0.76792** | 0.73732 |
| 颜色误差 | **0.00393** | 0.00804 |

Cold 从零步 0.824 快速恢复到 0.128，但**在 800 步等成本下仍落后 +0.038 LPIPS 且趋稳** → Gate 选 W，符合方案「Cold 持续落后但改善趋势明显可延长一次；若无追赶迹象直接用 Warm」。Warm 携带历史训练，这是「等追加成本」比较而非严格总训练成本比较（方案已注明）。

**两个关键交接已实测通过**：
1. P1_W 结束 → 产物归一化 `p1_W_best.pt`/`p1_W_metrics.jsonl`/`p1_W_summary.json` → 自动启动 P1_C
2. P1_C 结束 → P1 Gate 读三份标准名文件选 W → 自动启动 C0（`--init-from p1_W_best.pt`，`teacher=None` 路径无 KeyError，实测 14.4 img/s）

后续由 `run_sd0_s10.sh` 自主推进：C0 → HF Gate → C1 → 响应 Gate → final eval → 导出。每阶段 Gate 结果写入 `runs/sd0_s10/*_gate.json`，最终汇入 `FINAL_REPORT.json`。训练链结束后 `guard_p5.sh` 自动执行 `run_p5_final.sh`，补齐第三层评测、完整单测、独立包装器导出与 `P5_FINAL_REPORT.json`（含 `status` 与 `failed_steps`）。

**本轮修复（未打断串行链）**：
1. C1 响应 Gate 需要读 metrics jsonl 里的 `resp`/`resp_cos` 字段，但原 trainer 的 VALID 记录只有重建指标 → Gate 必然读到默认值 9 而误判失败。已给 trainer 加 `stage_sum` 跨 update 累加器，把训练侧均值以 `train_*` 与 Gate 友好别名写入 jsonl。P1_W 是旧进程（Gate 不读 resp，不受影响）；P1_C/C0/C1 是新进程，使用修复版。三个 Gate 表达式已在模拟数据上干跑验证，且 P1_C 的真实 `p1_cold_C_metrics.jsonl` 已确认带 `train_charb / train_hf / train_lpips` 等字段，`hf_l1 / hf_corr_L0 / lpips / csd` 齐全。
2. **C1 teacher 输入契约错误（会毁掉响应蒸馏）**：原 `teacher = official.decoder`，但 `response_loss` 传的是未缩放 raw latent `z`。官方 Decoder 的正确输入是 `pqc(z)`（`AutoencoderKL.decode` 的路径），student 的 `StudentDecoder` 内部已含 pqc → teacher 与 student 看到的输入不同，Δy_T 是错的。已改为 `StudentDecoder(pqc_sd0, official.decoder)`，与 student 同接口。`eval_sd0_s10.py` 的 teacher 本来就是 `official.decoder(official.post_quant_conv(z))`，无需改。P1_C 是旧进程（无 response 分支，不受影响）；只有 C1 新进程会用到修复版 teacher。

**并发 smoke 的教训**：曾在 P1_W 训练时另起 P1_C smoke，两进程抢 24 GB 显存导致 OOM（非 P1_C 本身问题）。已清理残留并确认串行链未受影响。后续所有验证改为本地干跑或等 GPU 空闲，严格执行单 GPU 串行队列。
