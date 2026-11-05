# V4 风格突破 · 阶段报告（2026-11-05）

> 写作时间 2026-11-05 12:50（Asia/Shanghai）。T0 = 02:50，14h 窗口至 16:50。
> 本地写、git 提交后再同步远程，符合「远程代码只读」约定。

## 0. 一页结论（300 字内）

这轮没有拿到"强、可靠、可控"的目标风格。**E0（eps）已证伪**：6000 步、注入修复后 `style01_rel_diff` 0.0098–0.0106（B 臂 0.0137，反而更低），`style0_vs_null` 0.0052（B 0.00688）。**换成 v-target（E1/E2）先把去噪空间的风格可分辨性做出来了**：正式审计首次出现有效桶 `[6,7]`（t≈750–999），相对增益 +8.4%/+21.9%，但图像空间没转化——DINO-S/DINO-C 反而低于 E0。**扩容不是解药，而且这次有了直接证据**：8.45M→14.17M（E4）+68% 参数、8000 步，val 只从 0.4958 降到 0.4945，DINO-S 只 +0.016、LPIPS 反而差 0.005；加深（E5，12.77M）更差（val 0.5011）。**引导是通的，但幅度受限**：把引导放进真正有信号的 t∈[750,999] 后，w 0→20 让 DINO-S 单调 0.4172→0.4463（+7%），代价 LPIPS 0.606→0.631；目视到 w20 才出现可辨的色调差异，结构仍然缺失。**瓶颈判断**：模型能编码风格（分类器可读、审计显著），但不能在像素上生成对应结构；train/val 双双卡在 v-loss≈0.50（target_std 0.90，R²≈0.40），属于"优化/目标未收敛"，不是单纯容量不足。E6（20k 步长训）在跑，是当前唯一未关闭的机制假设。

---

## 1. 起点：B 臂与审计根因

**B 臂读数**（`runs/v4/arms/B/train.jsonl` 末行 `event=done`）：`step=5500, best_step=4000, best_val=0.15265, elapsed_s=3559.6`。config（`runs/v4/arms/B/config.json`）：eps 目标隐含（无 `prediction_type` 字段，trainer 默认 eps）、`cond_mode` sum、`film_init_std=0.02`、`batch_size=256`、`lr=2.8e-4`、`cfg_dropout=0.0`。

> **"19 小时"未能核实**：本机 B 臂产物只有 3559.6 s（≈59 min）/5500 步。若指另一台/另一轮旧 B 臂，其日志不在本仓库，本报告不作数字引用。

**审计根因**（`runs/v4/style_breakthrough/audit_B_best.json`）：
- `verdict.pass = false`，`effective_vs_wrong_buckets = []`、`effective_vs_null_buckets = []`。
- `cond_loss.buckets[*].rel_null_pct` 最高仅 **+1.76%（bucket 6）**，胜率 0.71；bucket 7 为 **−0.05%**（负增益）。判据要求 `ΔL>0 ∧ CI95下界>0 ∧ 胜率≥0.75 ∧ 所有 rep 均值>0 ∧ 相对≥0.5%`。
- `grad[*]`：`style_embed` 0.00097/0.00124/0.00090/0.00042/0.00009/0.00010/0.00005/0.00004（桶 0→7），`out_conv` 0.0224/0.0191/0.0115/0.0107/0.0089/0.0071/0.0041/0.0039。
  → `out_conv / style_embed` 实测 **12.8×–103.8×**（桶 4 为 98.9×，桶 7 为 103.8×）。用户口径的"约 120 倍"**未被本产物复现**，量级（百倍）成立、具体倍数不成立。
- `film[*]`：桶 0 的 `stage0.0.film1` `delta_over_h=0.453`（time）、`style_specific_rel=0.150`，`style_film_proj` 梯度为 **0.0**、`zero_grad_params=1/148` —— 与"FiLM 主要在编码 t、style 分支梯度近乎为零"一致。

**根因判断**：在 eps 目标下，条件路径（style_embed→style_proj）梯度比主干输出小两个量级，条件信息没有进入可用的生成方向；且风格可分辨性只出现在高噪声桶，低噪声桶增益 <1%，低于噪声底。

---

## 2. E0 证伪了什么

配置 `v4/arms/E0.json`：eps、sum FiLM（`film_init_std 0.02`）、6000 步、batch 256、lr 1e-4、`early_stop_after 6000 / patience 6`。
产物：`runs/v4/style_breakthrough/E0/{best.pt(step 5500), step_02000/04000/06000.pt}`。

**速度/资源**（`E0/train.jsonl`）：s/step 中位 **0.528**、墙钟 **3212.5 s**、峰值自报 **8.65 GiB**（`nvidia-smi` 实测进程约 11.4/12 GiB）、有效 batch **256**、null_frac **0.098**。

**代理曲线**（每 250 步，`E0/train.jsonl` 的 `event=val`）：
| step | val_loss | val_x0_mse | style01_rel_diff | style0_vs_null |
|---|---|---|---|---|
| 250 | 0.58710 | 650.864 | 0.00438 | 0.00395 |
| 500 | 0.20967 | 67.738 | 0.00350 | 0.00318 |
| 1000 | 0.17225 | 10.496 | 0.00619 | 0.00395 |
| 2000 | 0.16290 | 3.776 | 0.00763 | 0.00467 |
| 3000 | 0.15969 | 2.421 | 0.00768 | 0.00451 |
| 4000 | 0.15803 | 1.903 | 0.00920 | 0.00487 |
| 5000 | 0.15719 | 1.666 | 0.00898 | 0.00472 |
| **5500 (best)** | 0.15696 | 1.603 | 0.00976 | 0.00519 |
| 6000 | 0.15681 | 1.559 | 0.00988 | 0.00535 |

（另有 5750 的 `style01_rel_diff=0.01061` 为全程最高，但 5500/5750/6000 三点在 0.0098–0.0106 间抖动，不存在单调趋势。）

**平台起点 ≈3000 步**：3000→6000 `val_loss` 只再降 **1.8%**；`style01_rel_diff` 从 0.00768 到 0.00988，绝对值始终 <0.011，且 4000→5000 还回落过一次。`val_x0_mse` 已从 650 收敛到 1.56（前 500 步的下降几乎全部来自这里，不是风格）。

**同口径对照**：B-best `style01_rel_diff` 0.0137 / `style0_vs_null` 0.00688 → E0 **−29% / −25%**，即"eps + 修复注入"比不修还差。

**为什么"更久/更大"不是解药**：见 §4——E4（+68% 参数、8000 步）在 eps→v 的口径下也只把 val 从 0.4958 动到 0.4945；E0 这条路的问题不在步数，在"条件信息进不了生成方向"。

**E0 四项指标**（`smallset/E0_best`，s0.45、75 对、20 步 DDIM，`gen.json#content_lpips` + `refpool.json`）：

| ckpt | DINO-S↑ | DINO-C↑ | LPIPS↓ | CLIP-S↑ |
|---|---|---|---|---|
| step_02000 | 0.39422 | 0.35752 | 0.61594 | 0.70403 |
| step_04000 | 0.39835 | 0.36989 | 0.60367 | 0.70836 |
| **best(5500)** | 0.40109 | 0.37338 | 0.60080 | 0.70873 |
| step_06000 | 0.40149 | 0.37372 | 0.60096 | 0.70871 |

（同池对照组：identity DINO-S 0.53742 / DINO-C 1.0 / CLIP-S 0.63901；VAE roundtrip DINO-S_s 0.5341 / DINO-C 0.95284 / LPIPS 0.24918 / CLIP-S 0.80564。）

> 注意：**identity 的 DINO-S(0.537) 高于任何生成结果（0.385–0.417）**，说明本池 DINO-S 主要由通用 DINO 相似度驱动、不能单独证明"风格转移"。CLIP-S 上生成 0.708–0.717 > identity 0.639，是唯一站得住的正面信号；代价是 DINO-C 从 1.0 掉到 0.33–0.38、LPIPS 0.60–0.71；而 VAE roundtrip 清晰（LPIPS 0.249）证明解码器无罪。

---

## 3. v-target 臂与分类器：当前状态

**E1（v、sum FiLM）**：`E1/train.jsonl` 末行 `done`：`step=3750, best_step=2750, best_val=0.51557, elapsed_s=2024.4`。曲线关键点：
- `val_loss` 0.6779@250 → 0.5156@2750 → 0.5132@3750。
- `style01_rel_diff` 峰值 **0.1782@2500**，随后降到 0.1499@2750、0.1389@3750；`style0_vs_null` 峰值 0.1146@2500 → 0.0952@3750。
- `target_std` 在 `train.jsonl` 的 `event=train` 字段里约 **0.90–0.92**（v-target 已归一，符合 v-pred 设计）。
→ **初判**：v-target 确实把"条件可分辨性"从噪声底抬出来了（代理量约为 E0 的 14–18 倍），但代理量在 2500 步后与 `val_loss` 脱钩并回落；离判据（正式审计 pass=true）仍差：审计只拿到 `effective_vs_null_buckets=[6,7]`，`effective_vs_wrong_buckets=[]`、`pass=false`。

**E2（v、split FiLM、`style_film_dim=64`、`style_film_init_std=0.05`）**：`runs/v4/style_breakthrough/E2/train.jsonl` → 4 000 步未早停，best 3250 = **0.5143**，s/step 0.576，峰值 9.53 GiB；`style01_rel_diff=0.0882`、`style0_vs_null=0.1031`。**比 E1 低**（0.088 vs 0.178），即"强逐层注入"在本题上更差。

**分类器（`v4/style_classifier.py`）**：main `best val_top1=0.6559@1500`（final 0.6378）；alt `best 0.665@2000`（final 0.6353）。**swap 语义评估**（独立 clf_alt 判定，`evalclf_*`）：无引导 top1 **0.200（等于随机，输出被判为源风格）** → w1.0 0.2375 → w2.0 0.325（E1）/0.3125（E2），margin 仍为负（−0.24）。**未完成**：尚未在 E4/E5/E6 上重跑 swap（见 §9）。

**新增扩容臂（本轮为回答"扩容收益"而做）**：
| 臂 | 结构 | 步数 | s/step | 峰值 | best val | style01(末) | DINO-S / DINO-C / LPIPS / CLIP-S |
|---|---|---|---|---|---|---|---|
| E4 | 宽 96/192/256/320，1 block，14.17M | 8000 | 0.616 | 8.1 GiB | 0.4958@6000 | 0.1976 | 0.41723 / 0.37735 / 0.6056 / 0.70875 |
| E5 | 原宽，2 blocks，12.77M | 8000 | 0.392 | 6.18 GiB | 0.5011@6500 | 0.1857 | 0.41606 / 0.36541 / 0.6097 / 0.71378 |

**E6（长训臂，运行中）**：`v4/arms/E6.json`（E4 结构、20000 步、`val_every 500`），11:30:55 起，0.62 s/step，ETA ≈ 15:00。截至 12:25 已到 step 4300，`val_loss` 0.5042@3000（E4 同 step 0.5050）。

---

## 4. 规模扩张的证据（含"缺乏直接证据"的说明）

**我们真跑过的扩容对照（本轮新增，直接证据）**：
- 8.45M（E0/E1/E2）→ 14.17M（E4）：**+68% 参数、步数 6000→8000**，`val_loss` 0.15696（eps 口径）与 0.4958（v 口径）不可直接比；同口径只有 E1/E2→E4：0.5132/0.5143 → **0.4958**（−3%），DINO-S 0.38523/— → **0.41723**（+0.032），LPIPS 0.7026 → **0.6056**（改善），CLIP 0.6363 → **0.7088**。
- 加深（E5，12.77M，2 blocks/stage）：`val_loss` **0.5011**、DINO-S **0.41606** → **比加宽更差**。
- 结论：**"扩容收益不大"有直接证据，但它不是零**——图像空间指标确有改善（主要来自 v-target，而非参数），去噪空间的风格可分辨性（审计桶 6/7 增益 11.8%）与 E1 完全同水位（E1 11.79%、E2 12.96%、E4 11.83%）。

**prior 9250 步 vs arms 4000 步（不可直接比较）**：
- `runs/v4/prior/train.jsonl`：`best_step=9250, best_val=0.13098, elapsed_s=1202.6`；config `batch_size=16, grad_accum=2` → 有效 batch **32**，9250 步 = **296k 样本**。
- `runs/v4/arms/B`：4000 步、有效 batch **256** → **1.02M 样本**（是 prior 的 3.4 倍）。
→ **"9250 步 > 4000 步"这个比较被 batch 混淆**，不能作为"多训更久更好"的证据；反过来，样本数更多、val 更高（0.153 vs 0.131），说明 prior 的 lr/batch 配置本身更"拟合"。**此条标"不确定"**。

**`runs/v4/prior/sweep_10k/table.json`（历史模型，不同口径）**：
| 配置 | DINO-S | DINO-C | CLIP-S | LPIPS |
|---|---|---|---|---|
| identity | 0.5013 | **0.9006** | 0.7087 | 0.2766 |
| s0.05 | **0.5185** | 0.8512 | 0.7212 | 0.3219 |
| s0.15 | 0.5032 | 0.6606 | 0.7377 | 0.4430 |
| s0.30 | 0.5031 | 0.4869 | 0.7391 | 0.5634 |
| s0.45 | 0.5107 | 0.4117 | 0.7454 | 0.6454 |
| s0.60 | 0.5302 | 0.3743 | 0.7522 | 0.7114 |

→ 事实是 **DINO-C 从 0.90 崩到 0.37**，而 **DINO-S 在 0.503–0.530 的窄带里非单调波动**（s0.05 最高 0.5185，s0.15/0.30 反而低于 identity）。所以"strength 升 → DINO-S 升"这个读法**不成立**；DINO-S 在这个池里区分力不足，不能当作风格达成的主证据。

**24M 级别对照：从未跑过**。本仓库无对应产物，标"未核实"。

---

## 5. 对计划的含义

| 判断 | 状态 | 依据 |
|---|---|---|
| eps 目标 + 修复注入 → 产生条件响应 | **已证伪** | E0 6000 步：style01 0.0098 vs B 0.0137；审计 pass=false、无有效桶 |
| 靠"训更久/更大"解决 | **已证伪（直接证据）** | E4 +68% 参数 +33% 步数：val −0.3%、DINO-S +0.016；E5 加深更差 |
| DINO-S 作为风格达成的主指标 | **证伪（度量层面）** | 本池 identity DINO-S 0.537 > 所有生成 0.385–0.417；历史池 identity 0.5013 ≈ 生成 0.5107 |
| v-target 让风格可分辨 | **成立（仅去噪空间）** | 审计桶 6/7：E1 +8.4%/+21.9%、E4 +11.8%/+22.4%，胜率 0.82 |
| 分类器引导能把可分辨性转成像素 | **未验证（方向部分成立）** | w 0→20 单调：DINO-S 0.4172→0.4463、CLIP 0.7088→0.7165、LPIPS 0.6056→0.6311；目视 w20 才见色调差异 |
| 强逐层条件（split，`style_film_init_std .05`） | **证伪（更差）** | E2 style01 0.0882 < E1 0.1782 |
| 长训（20k）能否突破 washed-out | **完全未测（正在跑）** | E6 11:30 起，ETA 15:00 |
| 采样步数/强度不足导致高频丢失 | **证伪** | 50 步 vs 20 步：DINO-S 0.4172→0.4224、LPIPS 0.6056→0.5991，量级不变 |

**下一轮应该围绕什么重新设计**：不是再加参数/步数，而是**先修"生成保真度"**。证据是同一模型下 VAE roundtrip 清晰（LPIPS 0.249、CLIP-S 0.806），而任何扩散输出都糊（LPIPS 0.60–0.71）；同时 train/val 都停在 v-loss≈0.50（target_std 0.90，R²≈0.40）。建议的下一步（按性价比）：
1. **目标/优化诊断**：固定 batch、扫 lr（2e-4/4e-4）与 t 采样分布，看 train_loss 能否压到 0.35 以下；若压不下去，说明不是容量而是目标/调度/EMA 的问题。
2. **蒸馏式重建自检**：用真实 latent 在 t=500 单步去噪，看 `x0` 预测的 std 是否系统性塌缩（若是，属于回归到均值，需换 loss 或加大噪声端权重）。
3. 分类器引导改成**只在 t∈[750,999] 生效**并用独立分类器判定（本轮已把 `--guide-t-min` 落地，提交 `f43b0d8` 系列），把"可分辨性→像素"这条链补齐后再评估。

---

## 6. 附加核实：内部早停是否真的生效

**6.1 触发记录（历史真实触发）**
- 盲跑在线案例：`docs/reproduction/internal_dynamics_early_stop.csv` 第 4 行 `epoch=4, gate_delta=0.0037097484, shared_ll_hf_grad_ratio=0.1634, stop_requested=1`；`docs/reproduction/internal_dynamics_early_stop.md#L30-L47` 记录"monitor selected epoch 4 and terminated training there"。
- 鲁棒性曲线（`docs/reproduction/internal_dynamics_robustness_*.jsonl`，字段 `internal_probe_stop_requested`）：
  - `_b2.jsonl` 第 4 行 = 1.0（epoch 4 触发，文件仅 4 行）
  - `_b8_relative.jsonl` 第 4 行 = 1.0（epoch 4，文件 4 行）
  - `_seed123.jsonl` 第 3 行 = 1.0（epoch 3，文件 3 行）
  - `_seed7.jsonl` / `_b8.jsonl`：15 行，**全 0**；与 `internal_dynamics_early_stop.md#L70-L73` 一致——绝对阈值 `rho=1` 对 seed 7 / batch 8 不可达，故改用相对收缩规则 `gate_delta>0 ∧ rho_epoch/rho_prev≤0.65`。
→ **不是"从未触发"**：至少 3 次真实触发（e42 盲跑、b2、b8_relative），并在 seed123 上提前到 epoch 3。

**6.2 触发是否选对 ckpt**
- `docs/713/SUBMISSION_HANDOFF_2026-07-15.md#L147-L151`：epoch 4 DINO-S **0.4915** / CLIP 0.7126 / LPIPS 0.2596 / DINO-C 0.8103；epoch 6 DINO-S **0.4878** / CLIP 0.7101 / LPIPS **0.2563** / DINO-C **0.8215**。
- 3 seed 复核（`internal_dynamics_early_stop.md#L76-L80`、`seede7_eval.json`）：seed7 ep4 0.4910/0.7140/0.2668/0.8076；seed42 ep4 0.4915/0.7126/0.2596/0.8103；seed123 ep3 0.4862/0.7144/0.2552/0.8040。
→ **选对了"风格"**（ep4 的 DINO-S/CLIP-S 高于 ep6），代价是内容指标略差（DINO-C 0.810 vs 0.822、LPIPS 0.2596 vs 0.2563）。**epoch 5、7–15 的逐 epoch 数值未在本仓库找到，标"未核实"**；"ep4 是 DINO-S 最大"目前只有文档断言 + ep6 对照支撑。

**6.3 省了多少**
- 计划 15 epoch，停在第 4 → **省 11/15 = 73% 的训练**。
- 真实墙钟/步数：原 run 落在 `I:\Github\Latent_Style\WEAVE\runs\submission\hf_oriented_internal_early_stop\`，该路径在本机**已不存在**（`/mnt/i/Github/Latent_Style/WEAVE` 下无 `runs`），故**具体分钟数未核实**。

**6.4 能否搬到 v4**
- 判据依赖 `loss_fm_spectral_ll` / `loss_fm_hf_total` 与 LH/HL 目标-HF **gate**（`gate_delta`、`rho`）。v4 的 `V4UNet` 没有 LF/HF 路由分解，也没有 per-route gate → **这三个量在 v4 上无定义，当前不可计算**。
- 文档自己限定了范围：`internal_dynamics_early_stop.md#L86-L90`——"validated for the oriented-HF architecture and should not yet be presented as an architecture-independent theorem… 换路由设计必须显式定义 gate 与参数组，并先在一条完整曲线上复盘"。
→ **是有意不搬，不是遗漏**。v4 现在的早停是另一套机制：`train.py` 的相对改善判据（`improved = val_loss < best*(1-0.005)`，`early_stop_after` + `patience`）；E1 在 3750 停止（`E1/train.jsonl` 末行 `event=done, step=3750`）即该机制触发，与内部动力学无关。

**6.5 一句话结论**：内部早停是**"在 oriented-HF 架构上真实触发、并被 3 个 seed 复盘验证过的机制"**，不是"写了从未生效的代码"；但它**在 v4 上完全未移植（有意，且当前无对应可算量）**，所以对 14h 这轮的四个 v 臂没有任何作用——v4 的早停只由 val-loss patience 决定。

---

## 7. 时间账（T0 02:50，窗口至 16:50）

| 阶段 | 实际时刻 | 耗时 | 偏差/说明 |
|---|---|---|---|
| E0 训练 | 02:50:10 → 03:43:42 | 3212.5 s | 计划内 |
| **空档** | 03:43:42 → 05:15:49 | **92 min** | **GPU 空转**（首轮评测队列 06:54–07:00 又空转 6 min，另有 11:07–11:16 的 9 min，见 §9） |
| E1 训练 | 05:15:49 → 05:49:33 | 2024.4 s | 早停 3750/4000 |
| E2 训练 | → 06:28:12 | ~2300 s | 4000 步未早停 |
| clf main / alt | 07:0x → 08:0x | — | main best 0.6559@1500；alt 0.665@2000 |
| E0/E1 四项指标 + 引导扫描 | ~07:20 → 09:35 | ~2.2 h | 含 75 对 × 多次配置 |
| E4 训练 | 08:17:02 → 09:38 | 4943.2 s | 与评测串行，无重叠 |
| E4 评测（4 ckpt + s75） | 09:40 → 10:05 | ~25 min | |
| 50 步采样实验 | 10:04 → 10:10 | ~6 min | |
| E5 训练 | 10:15:07 → 11:07 | 3132 s | |
| E5 评测 + w8/w20 | 11:16:52 → 11:35 | ~18 min | 首次脚本 11:06:55 失败（9 min 空转） |
| **E6 训练（进行中）** | 11:30:55 → ~15:00 | ~12400 s（估） | 20000 步 |

累计空转 **≈107 min**（92 + 6 + 9），占总窗口约 12.8%。**当前进度**：14h 中已用约 10h（02:50→12:50），剩 4h；E6 占 2.5h，剩 1.5h 用于 E6 评测与最终报告。

---

## 8. 产物与 ckpt 清单

路径前缀 `runs/v4/style_breakthrough/`（远程 `/mnt/i/Latent_Style/WEAVE/`）。

| 臂 | ckpt 文件 | 步数 | 关键指标 |
|---|---|---|---|
| E0 (eps) | `E0/best.pt`, `step_02000/04000/06000.pt`, `last.pt` | 6000 | best 5500, val 0.15696, style01 0.0098 |
| E1 (v,sum) | `E1/best.pt`, `step_02000.pt`, `last.pt` | 3750 | best 2750, val 0.51557, style01 0.1782@2500 |
| E2 (v,split) | `E2/best.pt`, `step_02000/04000.pt`, `last.pt` | 4000 | best 3250, val 0.5143, style01 0.0882 |
| E4 (宽 14.17M) | `E4/best.pt`, `step_02000/04000/06000.pt`, `last.pt` | 8000 | best 6000, val 0.4958, style01 0.1976 |
| E5 (深 12.77M) | `E5/best.pt`, `step_02000/04000/06000/08000.pt`, `last.pt` | 8000 | best 6500, val 0.5011, style01 0.1857 |
| E6 (长训) | 运行中 | 20000（ETA 15:00） | 截至 4300 步 val≈0.504 |

评测产物：`smallset/E0_{best,step_02000,step_04000,step_06000}`、`smallset/E4_{best,step_02000,step_04000,step_06000}`、`smallset/E4_best_s75`、`smallset/E5_best`、`smallset_g/*`（w1/2/4/8/20、t999 变体）、`sweep_10k`（历史）。审计 json：`audit_{B,E0,E1,E2,E4}_best.json`。拼图：`montage_w_sweep.png`、`montage_E4_guide_t999.png`。

---

## 9. 未完成 / 失败项（逐条留痕）

1. **GPU 空转 92 min**（03:43:42→05:15:49）+ 首轮评测队列 6 min（06:54:35–07:00:20）+ 本轮 9 min（11:07–11:16，脚本 `set -u` 在函数内引用未传参立即退出，未生成文件，已改写成显式命令重启）。合计 ≈107 min。
2. **`1e33fb8` 一度推送失败**（`Recv failure: Connection was reset` / `send-pack: unexpected disconnect`），后改用 stdin 同步到 3060；截至本报告本地与 `origin/main` 已一致（HEAD 已推）。
3. **两次既有 pytest 失败**：本轮**未复跑 pytest**，具体用例名**未核实**（用户所述，不应在本报告中写成已定位）。
4. **E1 无 `step_04000.pt`**（3750 早停），E4 无 `step_08000.pt`（8000 是末步，仅 `last.pt`）。
5. **未跑 24M 级对照**，未跑真正的"同 batch 只变步数"的受控长训对照（E6 是唯一近似：同结构 20000 步）。
6. **swap 语义评估尚未覆盖 E4/E5/E6**（只到 E1/E2）。
7. **B 臂"19 小时"说法**在本机产物中无法复现（3559.6 s / 5500 步），标"未核实"。
8. **内部早停 epoch 5–15 的逐 epoch 指标**未在本仓库找到，标"未核实"。

---

## 10. 复现命令（均为项目根相对路径）

```bash
# 训练（示例：E4）
python3 -m v4.train --config v4/arms/E4.json

# 显存/速度
python3 -m v4.train --config v4/arms/E4.json --bench --bench-mbs 64,96,128,160

# 四项指标（生成 + 参考池）
python3 -m v4.gen_smallset --ckpt runs/v4/style_breakthrough/E4/best.pt \
  --out runs/v4/style_breakthrough/smallset/E4_best \
  --src-per-style 3 --steps 20 --strength 0.45 --cfg 1.0 --seed 0 --lpips --lpips-net alex
python3 tools/12h/v4_refpool_eval.py runs/v4/style_breakthrough/smallset/E4_best \
  --n-per-style 10 --indices runs/v4/arms/B/indices.pt

# 引导（只在高噪声段生效）
python3 -m v4.gen_smallset --ckpt runs/v4/style_breakthrough/E4/best.pt \
  --out runs/v4/style_breakthrough/smallset_g/E4_w2_hi \
  --src-per-style 3 --steps 20 --strength 0.45 --seed 0 --lpips \
  --guide-clf runs/v4/style_breakthrough/clf_main.pt --guide-scale 2.0 \
  --guide-t-min 750 --guide-t-max 999

# 条件审计
python3 -m v4.audit_cond --ckpt runs/v4/style_breakthrough/E4/best.pt \
  --out runs/v4/style_breakthrough/audit_E4_best.json
```
