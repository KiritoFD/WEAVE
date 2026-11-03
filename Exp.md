# WEAVE 实验流水账

> 每个 run 结束立刻追加一行；无效 run 也记录并标原因。
> ~~指标口径：600 跨风格为主表 + 150 同风格补充；参考图策略 `first_image_of_target_style_test_dir`。~~ **旧口径（2026-11-04 已废弃）**：主表回到论文协议 750 all-pairs，600/150 只作拆分附表；详见下方"当前状态"。
> **SOTA 判定（2026-11-04 起，写死）**：600 DINO-S ≥ 基线 +0.02 且 CLIP-S ≥ +0.01 才算有风格收益；约束 DINO-C ≥ 0.7839、LPIPS ≤ 0.3153；必须按目标风格拆分（Minimalism 最弱）+ 报 150 同风格；每条路线 ≥3 档超参 × 2 seed；每次全量 750 评测 + gate + registry，`results.json` 必须带 protocol/counts/backbone/per_style。
> ~~目标变更：不再复现论文 0.4916，唯一比较对象是本仓库基线。~~ **已被推翻（2026-11-04）**：评测协议回到论文原表（750 all-pairs + dinov2-small + 论文 CLIP 主干），该任务已移交评测线。
> **早停标注规则（2026-11-04 起）**：`internal_probe_enabled=true` / `internal_early_stop_enabled=true` 已设为默认；#1–#31 为历史未武装（`early_stop_armed=false`），#32 起每张表/每条记录必须标注 `armed / fired / fired_epoch / stop_reason`。判据：`epoch>=3` 且 `rho/rho_prev<=0.65` 且 `gate_delta>0`，且 `global_step>=min_steps`（cap/曲线设 2000）。

## 当前状态（2026-11-04，训练线停手交底）

**一句话：内部 infra 已打稳，实验只跑到基线重建，A1–A4 / B 路线未开始；训练线由用户叫停，仓库只做整理，未 commit。**

### 1. 做到哪一步

| 类别 | 状态 | 说明 |
|---|---|---|
| 阻塞 bug（probe `NameError`） | **已修** | `trainer.py` 补 `probe_interval`（读配置）/`probe_accum`；#34 冒烟 200 步跨 3 个 probe 点、`SMOKE_RC=0` |
| infra 补丁 P7/P6 | **已合入** | 启动校验 + probe 日志节流；#36 before/after 0.515/0.481 → 0.513/0.480，loss 逐位相同 |
| infra 补丁 P1/P2/P3/P4 | **未合入（降为可选）** | 用户明确：纯内部监督路径用不上，保留为外部监督分支的可选开关，不进默认路径 |
| P8 评测链 | **未做** | `image_save_workers` / `save_summary_grid` / dinov2 eval batch 三处优化没有落地 |
| 数据模式（`train_sample_limit` / `pairing_seed` / `pairing_mode=fixed`） | **已实现** | A1 使用；默认关闭，现有 run 行为不变 |
| 分层抽样 + strict-reference 评测 | **已实现** | `--subset stratified --n 250`、`--ref_pool strict`；`results.json` 带 `subset`/`ref_policy`，#35 冒烟通过 |
| `tools/diagnose_saturation.py` | **已写，未实跑** | A2/A3/A4 三件套（速度余弦、‖v‖/‖u‖、分时间桶误差、TF/ODE 轨迹对比）；只有 CPU 冒烟级别验证，未产出分析结论 |
| identity 监督（`source_style_id` / `identity_prob` / `pairing_mode=same_style`） | **未做** | 上一轮授权的第 4 项被"停手"覆盖 |
| A1–A4 | **未开始** | A1 起了 3 个臂，被用户叫停，全部标 `aborted` |
| B 集成 / 18k latent 编码 | **未开始** | B 相关模块（`b1_style_operator.py` 等）仍在仓库根，未合并进主路径 |
| 论文协议 750 复现 | **已移交评测线** | 本条线不做；见下方"未完成/移交" |

### 2. run 有效性（registry 共 37 行）

| 状态 | run | 说明 |
|---|---|---|
| 有效 | #8 `eval_base_ep10`（0.3977）、#32 `baseline_nohh600`（0.3958）、#33 `b160_retest` | 均已进 registry；#33 只做吞吐/显存，无评测 |
| 有效（路线对照，均未超过基线） | b_/c_/d_ 系列（#14–#29） | 2000 步预算下 C/D 相对 Style-ID 对照无跨风格增益 |
| 有效（infra/诊断） | #7 `dino_cache_5k`、#19–#22 显存 ramp、#25 `routeD_vae_probe`、#30/#31 AdaIN on/off | 只用于 infra 与机制判定 |
| 作废 | #10–#13（round1 lr 平台期）+ `_invalid_lrbug_*` | registry 已加 `status: "aborted"`，不删行 |
| 作废 | `es_fire_big`（b180 OOM，peak 11.6G） | `status: "aborted"` |
| 作废（用户叫停） | `a1_32`（2000/2000 步但无评测）、`a1_128`（0 步）、`a1_512`（4 步） | registry `status: "aborted"`，`stop_reason: aborted_by_user` |
| 仅记录在 Exp.md、未进 registry | #34 `probe_smoke`、#35 `subset_smoke`、#36 `p6_bench_i1/i25`、`vram_slope`、`eval_parity` | infra 冒烟，不是实验结论 |

### 3. batch / 显存 / 吞吐现状

- **内部监督主干预默认档：batch=160**。5000 样本 → **31 步/epoch**，0.45–0.50 s/step，≈320–360 samples/s；peak 10.38 GB（guard）/10.02 GB reserved/9.28 GB alloc，无 OOM。
- 每样本边际显存 **0.0628 GB/sample**（b64/96/128 = 4.47/6.55/8.51 GB 实测，固定开销 ≈0.45 GB）；按此外推 b160 ≈ 10.5 GB，与实测吻合。
- 上限：**b176 未实测（作为理论上限写入 runbook）**；**b180/b192 已 OOM，不要再试**。
- batch 档位与 lr 重定规则、步数预算公式（`samples/batch × epochs = 总更新数`）见 [docs/NEXTGEN_RUNBOOK.md](file:///g:/GitHub/Latent_Style/WEAVE/docs/NEXTGEN_RUNBOOK.md)。

### 4. 早停是否可用

- 代码可用，#34 已实证能过 probe 点；`vram_guard.sh` 的"僵尸进程被当成低显存"误判已修（判活看 `ps` 状态）。
- **判据从未在 from-scratch 上真正 fired**：rho 结构性下限约 0.2，0.65 阈值 + `min_steps` 组合在 15 epoch 内一次都没触发；反而在 epoch3 因 `gate_delta` 恒零路径以 `reason=gate` 提前停（#34）。→ **可信度待验证**，非诊断 run 必须显式 `internal_early_stop_min_steps>=2000`，诊断 run 直接关早停。

### 5. 已知 bug / 风险

- ~~`probe_interval` / `probe_accum` NameError~~ **已修**（#34 验证）。
- 评测 batch 影响生成：`EVAL_BATCH=2` vs `32` 像素 max|Δ|=37/255（`eval_parity` 结论 `PARITY_FAIL`）→ **评测必须固定 `EVAL_BATCH=32`**，不要拿不同 batch 的结果横比。
- 6 个历史评测脚本在 `runs/` 顶层曾用单线程 PNG 编码 + 上万级 summary grid；评测链提速（P8）未做，全量 750 评测约 5–6 分钟。
- 计划基线（0.4918 / 1,037,087 参数 / 750 all-pairs）与本仓库基线（0.3977 / 873,680 / 600 跨风格）是**两套口径**，不得横比；且按 2026-11-04 最新决定，600/base 口径的数字一律标"旧口径（已废弃）"，主表待论文协议 750。

### 6. 未完成 / 移交清单

| 项 | 状态 | 备注 |
|---|---|---|
| 论文协议 750 复现（dinov2-small + 论文 CLIP 主干 + 750 all-pairs + 论文参考图策略 + 8 步 ODE + AdaIN 2.0） | **已移交评测线** | 本条线不做；canonical ckpt `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` 已丢失，`exp/710_baseline_weave_paper_reval` 为 reuse-only（无 ckpt） |
| A1 饱和诊断（32/128/512 样本过拟合） | 被叫停 | 只有 `a1_32` 跑完 2000 步，无评测；脚本在 `archives/20261104/run-scripts/a1_overfit.sh` |
| A2/A3/A4 诊断（速度余弦、‖v‖/‖u‖、分时间桶误差、TF/ODE 对比） | 未跑 | `tools/diagnose_saturation.py` 可用性仅 CPU 冒烟级 |
| B 集成 | 未开始 | `b1_style_operator.py` / `b2_destyle.py` / `b_pseudo_targets.py` / `b_training_diagnostics.py` 仍在仓库根，未接主路径 |
| 18k latent 编码 | 未开始 | — |
| P8 评测链提速 | 未做 | workers=4、关 summary grid、dinov2 eval batch=64 |
| P1 冻结特征缓存 | 未合入 | 开关实现已在 `frozen_feature_cache.py`，默认不启用 |

## 当前最优配置

| 路线 | run id | 配置要点 | 600 DINO-S | 600 CLIP-S | 600 DINO-C | 600 LPIPS | 150 DINO-S |
|---|---|---|---|---|---|---|---|
| baseline | base_ep10 | 论文模型 epoch_0010，L/14 + 统一 split | 0.3977 | 0.6425 | 0.7939 | 0.3053 | 0.8263 |
| C | c_ref_s0 | reference_hybrid + InfoNCE 0.1，batch 14，2000 步 | 0.3947 | 0.6426 | 0.8211 | 0.2740 | 0.8395 |

训练吞吐定档：**内部监督主干预 batch=160**（#33 复测：5000 样本 → 31 步/epoch，0.45–0.50 s/step，peak 10.38GB guard / 10.02GB reserved；b176 为上限，b192 已 OOM 不再尝试）。Route A 外部监督分支另计：**batch=14 / microbatch=14**（峰值 9.59GB、wall ≈8 img/s）；Route C 不带输出监督时 wall ≈197 img/s（Style-ID 臂），参考 token 装载会把 wall 压到 ≈50 img/s。

## Run 记录

| # | run id | 路线 | 关键配置 | seed | 步数/epoch | batch | 峰值显存 | 耗时 | 600 DINO-S | 600 CLIP-S | 600 DINO-C | 600 LPIPS | 150 DINO-S | 有效 | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | routeA_smoke_20261103 | A | λs=1.0 λc=0.5 steps=2 apply_every=1 | 0 | 2 epoch × 250 步 (vlm=0.05) | 1 | 6.15GB | 213s | — | — | — | — | — | 打通 | 机制可跑，loss 3.32→3.23，flow 1.09→1.11；未评估 |
| 2 | ramp_b1 | 吞吐 | Route A 全开，12 步 | 0 | 12 步 | 1 | 6.16GB | 34s | — | — | — | — | — | 有效 | 稳态 0.40s/step，sps≈2.5（长 epoch 实测） |
| 3 | ramp_b2 | 吞吐 | Route A 全开，12 步 | 0 | 12 步 | 2 | 12.21GB | 66s | — | — | — | — | — | 临界 | 显存几乎打满，sps 0.68 无提升 → 不用 |
| 4 | ramp_b4 | 吞吐 | Route A 全开，12 步 | 0 | — | 4 | — | 34s | — | — | — | — | — | **无效：OOM** | 12GB 上限下 batch 4 不可用 |
| 5 | ramp_b8 | 吞吐 | 未执行 | — | — | 8 | — | — | — | — | — | — | — | 未跑 | b4 OOM 后终止 |
| 6 | ramp2_b2 | 吞吐 | 显存释放后（空闲 844MiB）复测 | 0 | 12 步 | 2 | >11GB | 356s | — | — | — | — | — | **无效：OOM** | 仅空出 11GB 仍不足，b2 在 output_supervision 的 VGG 前向 OOM → 训练固定 batch=1 |
| 7 | dino_cache_5k | 依赖 | dinov2-base 224/last_hidden_state | — | — | 64 | — | ~60min | — | — | — | — | — | 有效 | content_5k 5000 样本 + refs_test 150，manifest 齐全 |
| 8 | eval_base_ep10 | 评测 | L/14 CLIP + 统一 split，全 750 张 | — | — | 2 | ~3GB | 350s | 0.3977 | 0.6425 | 0.7939 | 0.3053 | 0.8263 | 有效 | **新口径基线**（600+150 计数校验通过） |
| 9 | routeA_smoke_ep12 | A | λs=1.0 λc=0.5 steps=2 every=1（仅 500 步） | 0 | 2 epoch × 250 步 | 1 | 6.15GB | 213s | 0.3938 | 0.6461 | 0.8258 | 0.2722 | 0.8526 | 有效（机制生效） | 风格基本持平（DINO-S −0.004），LPIPS 反而更低 → 500 步不足，仅作链路验证，不进结论 |
| 10 | a_ls005_lc01_s2_e1_s0 | A | λs=0.05 λc=0.1 s2 e1 lr=1e-5 | 0 | 1000 步 | 1 | 6.15GB | 416s | 0.3912 | 0.6454 | 0.8359 | 0.2667 | 0.8572 | **无效**：总 loss 未下降（1.664→1.722） | lr 已到 1e-5 下限 → 平台期，机制生效但无有效更新，round1 作废 |
| 11 | a_ls01_lc01_s2_e1_s0 | A | λs=0.1 λc=0.1 s2 e1 lr=1e-5 | 0 | 1000 步 | 1 | 6.15GB | 414s | 0.3926 | 0.6454 | 0.8342 | 0.2670 | 0.8558 | **无效**：同上 | 风格未升（DINO-S −0.005），内容略升 → 结论同上 |
| 12 | a_ls03_lc01_s2_e1_s0 | A | λs=0.3 λc=0.1 s2 e1 lr=1e-5 | 0 | 1000 步 | 1 | 6.15GB | 419s | 0.3961 | 0.6443 | 0.8249 | 0.2672 | 0.8508 | **无效**：总 loss 未下降（1.678→1.691） | round1 三档全部作废：lr 已到 1e-5 下限，机制生效但无有效更新；round1b 以 lr=5e-5 / 2000 步重跑 |
| 13 | _invalid_lrbug_b_ls03_lc01_s2_e1_s0 | A | λs=0.3 λc=0.1 s2 e1 配置 lr=5e-5 | 0 | 中断于 epoch 11 | 1 | 6.15GB | — | — | — | — | — | — | **无效：中断** | 配了 lr=5e-5 但日志实际 `lr=1.00e-05`——resume 恢复了旧 ckpt 的 scheduler/optimizer 状态，覆盖新 lr。修复：`training.resume_optimizer=false`；队列已重启 |
| 14 | b_ls03_lc01_s2_e1_s0 | A | λs=0.3 λc=0.1 s2 e1 **lr=5e-5** | 0 | 2000 步 | 1 | 6.15GB | 800s | 0.3992 | 0.6480 | 0.8071 | 0.2655 | 0.8399 | **有效** | 首个 lr 真正生效的 Route A run：DINO-S +0.0015（未达 +0.03 阈值），DINO-C +0.013、LPIPS −0.040 明显改善；风格增益不足 |
| 15 | b_ls01_lc01_s2_e1_s0 | A | λs=0.1 λc=0.1 s2 e1 lr=5e-5 | 0 | 2000 步 | 1 | 6.15GB | 797s | 0.3913 | 0.6455 | 0.8319 | 0.2689 | 0.8536 | **有效** | λs 降到 0.1 后 600 跨风格退化（−0.0064），但 150 同风格 +0.027；说明 600 跨风格更吃风格监督强度 |
| 16 | b_ls03_lc03_s2_e1_s0 | A | λs=0.3 λc=0.3 s2 e1 lr=5e-5 | 0 | 2000 步 | 1 | 6.15GB | 771s | 0.3853 | 0.6459 | 0.8467 | 0.2729 | 0.8608 | **有效** | λc 翻三倍后 600 跨风格 DINO-S −0.0124（内容更强）；150 同风格 +0.034。跨风格与内容权重呈反向取舍，SWD 风格项压不过 FM |
| 17 | b_ls03_lc01_s4_e1_s0 | A | λs=0.3 λc=0.1 **s4** e1 lr=5e-5 | 0 | 2000 步 | 1 | 6.15GB | 1019s | 0.3929 | 0.6430 | 0.8318 | 0.2720 | 0.8476 | **有效** | 展开 4 步没有带来风格增益（0.3929 < 0.3992），吞吐降到 1.97 sps（bstep 0.507s）；多展开一倍代价换不到收益 |
| 18 | b_ls03_lc01_s2_e2_s0 | A | λs=0.3 λc=0.1 s2 **e2** lr=5e-5 | 0 | 2000 步 | 1 | 6.15GB | 453s | 0.3951 | 0.6453 | 0.8241 | 0.2681 | 0.8467 | **有效** | apply_every=2 吞吐翻倍（4.41 sps，bstep 0.227s），风格 0.3951 略低于 e1 的 0.3992；性价比最优 |
| 19 | memopt_ramp1 | 吞吐 | DINOv2@384 + bf16 + descriptor 分块，mb=1 | 0 | 12 步/档 | 1/2/4/6 | 5.26/5.64/6.40/7.16GB | 27–50s | — | — | — | — | — | 有效 | 新特征分支峰值远低于 VGG（b1 5.26GB）；mb=1 串行展开，wall 吞吐 1.22→3.0 img/s |
| 20 | memopt_ramp2 | 吞吐 | 同上，mb=batch 并行展开 | 0 | 20 步/档 | 6/8 | 7.27/8.07GB | 24–28s | — | — | — | — | — | 有效 | 并行 microbatch 提速：b6 3.02→4.94、b8 3.90→5.75 img/s（wall）；峰值 7.2–8.1GB |
| 21 | memopt_ramp3 | 吞吐 | 同上，高 batch | 0 | 20 步/档 | 10/12/14 | 8.87/9.67/9.60GB | 38–47s | — | — | — | — | — | 有效 | b10/12/14 = 5.32/5.75/6.02 img/s（wall），b14 峰值 9.6GB 仍在 10GB 预算内 |
| 22 | memopt_stability_b14 | 吞吐 | 最终选档 b14 mb14，100 步实测 | 0 | 100 步 | 14 | 9.59GB | 170.6s | — | — | — | — | — | 有效 | wall 吞吐 8.21 img/s（日志 `sps` 即 samples/s，含 data 时间）、1.706s/step、lr=5e-5 正确、GPU 利用率曾达 92.6%；相对旧 batch1 的 2.5 img/s 约 3.3×（此前"115 img/s / 46×"是把 sps 误乘 batch，已更正） |
| 23 | c_ref_s0 | C | reference_hybrid + InfoNCE 0.1，lr=5e-5，2000 步 | 0 | 2000 步 | 14 | 0.89GB | 560s | 0.3947 | 0.6426 | 0.8211 | 0.2740 | 0.8395 | **有效**（C 闸：nce 1.03–2.75、grad 非零、loss 下降） | 比 Style-ID 对照高 +0.0011 DINO-S / +0.0021 DINO-C，但仍低于基线 0.3977；data 351s vs comp 168s → 参考 token 装载是瓶颈 |
| 24 | c_styleid_s0 | C（对照） | style_id，nce=0，其余同 ref 臂 | 0 | 2000 步 | 14 | 0.84GB | 142s | 0.3936 | 0.6418 | 0.8219 | 0.2719 | 0.8406 | **有效**（CTRL 闸：机制按设计关闭、grad 非零、loss 下降） | 对照臂；比 ref 臂快 3.9×（196.98 vs 49.98 img/s）→ C 的吞吐损失来自 reference token 数据管线，不是模型本身 |
| 25 | routeD_vae_probe | D | sd-vae-ft-ema@512 重建上限，VGG relu2_2/relu3_3 | — | 150 图 | — | — | 90s | — | — | — | — | — | 有效 | HF 能量比中位数 0.996、edge IoU 0.74、SWD 0.119/0.206 → verdict `latent_ok`：潜空间能承载笔触细节，纹理模块（Route D）值得开 |
| 26 | c_ref_s1 | C | 同 c_ref_s0，seed 1 | 1 | 2000 步 | 14 | 0.89GB | 551s | 0.3938 | 0.6420 | 0.8260 | 0.2673 | 0.8427 | **有效**（C 闸：nce 非零、grad 非零、loss 下降） | seed 1 复现方向一致：仍低于基线 0.3977，与 Style-ID 对照差 +0.0002（DINO-S），说明 reference 分支的跨风格增益≈0 |
| 27 | c_styleid_s1 | C（对照） | style_id，seed 1 | 1 | 2000 步 | 14 | 0.84GB | 140s | 0.3948 | 0.6422 | 0.8235 | 0.2743 | 0.8439 | **有效**（CTRL 闸：机制按设计关闭、grad 非零、loss 下降） | 两 seed 平均：ref 0.3943 vs control 0.3942（差 +0.0001，噪声级）→ **Route C 参考分支在当前 2000 步预算下没有跨风格收益**，低于基线 0.3977 |
| 28 | d_tex_on_s0 | D | 纹理模块 hidden=32，seed 0，2000 步 | 0 | 2000 步 | 14 | 0.84GB | 161s | 0.3935 | 0.6421 | 0.8213 | 0.2728 | 0.8406 | **有效**（D 闸：纹理模块 ON、`texture_refiner.out.weight` norm 3.76、grad 非零、loss 下降） | 相对 Style-ID 对照 −0.0001 DINO-S；纹理模块在 2000 步预算下无可测增益 |
| 29 | d_tex_on_s1 | D | 同上，seed 1 | 1 | 2000 步 | 14 | 0.84GB | 161s | 0.3950 | 0.6418 | 0.8226 | 0.2748 | 0.8432 | **有效**（D 闸） | 两 seed 平均 0.3943 vs 对照 0.3942 → 同 C：无增益；逐风格看 Minimalism 未改善 |
| 30 | p0_adain_on | 诊断 | base_ep10 + `endpoint_adain_scale=1.0`（每步统计匹配） | — | 750 图 | 2 | ~3GB | 292s | 0.3980 | 0.6425 | 0.7941 | 0.3053 | 0.8264 | 有效 | 复现基线（#8 为 0.3977，差 0.0003 为评测噪声） |
| 31 | p0_adain_off | 诊断 | 同 ckpt，`endpoint_adain_scale=0.0`（纯网络输出） | — | 750 图 | 2 | ~3GB | 296s | 0.3977 | 0.6425 | 0.7940 | 0.3053 | 0.8264 | 有效 | **AdaIN-off 与 on 的差：DINO-S +0.0003 / CLIP-S 0.0000 / DINO-C +0.0001 / LPIPS 0.0000**；750/750 张图像哈希全不同（mean\|Δ\|=0.28/255），说明匹配确实执行但对感知指标贡献≈0 → 用户推论"其余都是推理期外挂"不成立，0.3977 基本是网络自身输出；后续 ACD 结论天然满足"关掉 AdaIN 仍成立"，无需再复核 |
| 32 | baseline_nohh600 | **计划内固定基线** | `experiments/architecture/hf_oriented_nohh.json`：b96、lr 2e-4、600 步（12 ep × 52）、`enable_hh_head=false` | 0 | 600 步 | 96 | 6.6GB | 198s | 0.3958 | 0.6414 | 0.7907 | 0.3060 | 0.8240 | 有效（early stop 未武装） | 计划内 600 步基线；与 #8（0.3977，论文 ep10 ckpt）差 0.0019，属不同初始化/预算的噪声量级，见"口径登记" |
| 33 | b160_retest | 吞吐受控复测 | #32 配置，b160、496 步（16 ep × 31）、lr=5e-5·√(160/14) | 0 | 496 步 | 160 | 10.38GB(guard)/10.02 reserved/9.28 alloc | 248s | — | — | — | — | — | 有效 | 0.45–0.50 s/step，无 OOM；之前那次 531MiB 掉零未复现；`vram_guard` 僵尸修复后未误杀 → **b160 定为标准档**，b176 为上限，b192 不再尝试 |
| 34 | probe_smoke | infra 冒烟 | 默认 `internal_probe_enabled` + `internal_early_stop_enabled`，200 步跨 3 个 probe 点 | 0 | 200 步 | 96 | 6.72GB | 73s | — | — | — | — | — | 有效 | 修掉 `probe_interval/probe_accum` NameError：3 个 probe 点全部走通，`SMOKE_RC=0`；early stop 在 epoch3 以 `reason=gate` 触发（gate 恒零路径），**不是 rho 0.65 触发** |
| 35 | subset_smoke | 评测协议冒烟 | `--subset stratified --n 10 --ref_pool strict` | — | 50 请求（40 跨 + 10 同） | 8 | — | ~3min | 0.3802 | 0.6282 | 0.7776 | 0.2978 | 0.8088 | 有效（仅协议验证） | 10 张内容图（2/风格）+ 每风格 1 张保留参考图；`eval_protocol.json` 落盘，`results.json` 带 `subset`/`ref_policy`；n=10 不用于任何结论 |
| 36 | p6_bench_i1 / i25 | infra 冒烟 | b160、100 步，仅 `probe_log_interval` 不同（1=逐步行、25=节流） | 0 | 100 步 | 160 | 9.48GB | 62s / 61s | — | — | — | — | — | 有效 | unthrottled bstep 0.515/0.481 → throttled 0.513/0.480：节流省约 1–2 ms/step，loss 逐位相同（1.3257 / 1.3189）→ 与审计结论一致，同步点不是瓶颈 |

## 内部早停记录

| run | armed | fired | fired_epoch | stop_reason | rho(首→末) | gate(首→末) |
|---|---|---|---|---|---|---|
| #1–#31（历史） | false | false | — | early stop disabled | — | — |
| cap32_es / cap128_es（待跑） | true | 待定 | 待定 | 待定 | 待定 | 待定 |
| curve（待跑） | true（min_steps=2000） | 待定 | 待定 | 待定 | 待定 | 待定 |

数据源：每个 run 的 `ckpt/internal_dynamics_summary.json`；registry 字段 `early_stop_armed / fired / fired_epoch / rho_series / gate_mean_series / stop_reason`。

## 备注

- 2026-11-03 历史基线（旧 CLIP base-patch32、旧产出复用）：600 跨 DINO-S 0.4080 / CLIP-S 0.7007 / DINO-C 0.7667 / LPIPS 0.3412；150 同 DINO-S 0.8264 / CLIP-S 0.8467 / DINO-C 0.8240 / LPIPS 0.2568。仅作参考，正式对比须用 L/14 同管线重评。

## 口径登记（2026-11-04）

### 两套基线口径，不得横比

| 口径 | 来源 | 参数量 | 评测集 | 600 DINO-S | 600 CLIP-S | 600 DINO-C | 600 LPIPS | 150 DINO-S |
|---|---|---|---|---|---|---|---|---|
| 计划基线 | `docs/reproduction/hf_oriented_nohh_result.md` | 1,037,087（旧代码含死 HH 权重） | 750 all-pairs | 0.4918 | — | — | — | — |
| 本仓库基线 A | #8 eval_base_ep10（论文 ep10 ckpt） | 873,680（无 fusion 头、无 HH） | 600 跨 + 150 同 | 0.3977 | 0.6425 | 0.7939 | 0.3053 | 0.8263 |
| 本仓库基线 B | #32 baseline_nohh600（计划内固定基线） | 982,618（fusion 头、HH 关闭） | 600 跨 + 150 同 | 0.3958 | 0.6414 | 0.7907 | 0.3060 | 0.8240 |

每张表必须标注用的是"计划基线 0.4918 / 750 all-pairs"还是"本仓库基线 0.3958–0.3977 / 600 跨风格"。两套口径的参数量、评测集、ckpt 来源都不同，**任何跨口径比较均无效**。

### 参数量差异（1.037M vs 874k）

- 873,680 = 主干预（无 `target_latent_hf_subband_fusion` 头、无 HH 头）。
- 982,618 = 主干预 + target-HF fusion 头（lh/hl 两个 delta 头，约 109k），HH 头关闭；当前代码已不再构建 54,469 个死 HH 权重。
- 1,037,087（计划报告）= 982,618 + HH 专用权重 54,469 + HH 头 2,436；旧代码即使 `enable_hh_head=false` 也会建这些参数但不参与前向（"死权重"）。
- 结论：差异主体是 **fusion 头 ~109k + 旧代码死 HH 权重 54.5k**，不是"仅 HH 头 2.4k"；功能上 982,618 与计划 1,037,087 等价。

### 早停阈值可信度

- 判据：`epoch>=3` 且 `rho/rho_prev <= 0.65` 且 `gate_delta > 0`（`rho = 共享主干 LL 梯度范数 ÷ HF 梯度范数`）。
- rho 结构性下限约 0.2；35% 单轮下跌在 15 epoch 内从未触发 → **0.65 阈值从未在 from-scratch 上触发过，可信度待验证**。
- 非诊断 run 保持 0.65 + `min_steps=2000`；A1 诊断 run 关闭早停。
- #34 probe_smoke 的 `reason=gate` 是 gate 恒零的启动校验路径，与 rho 阈值无关：默认配置（`internal_early_stop_enabled=true`、`min_steps=0`）在 epoch 3 就会因此停下。**非诊断 run 必须显式设 `internal_early_stop_min_steps>=2000`**，否则等价于早停失效；A1/A2/A3/A4 诊断 run 直接关闭早停。
