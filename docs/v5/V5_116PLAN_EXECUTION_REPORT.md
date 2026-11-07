# WEAVE V5 / 116plan 24h 实验执行报告

> 自动生成于 2026-11-07 10:48:14；证据均来自 `runs/v5/**/*.json` 与完整 750 评测产物。

## 0. 决策树实际路径

```text
P0 条件/数据冻结
  -> P1 Latent Oracle O1/O2/O3 (弱/强/无约束)
     -> G1: Oracle success=False => Branch_B_Supervision_Fix
  -> B1 训练 Latent Style Encoder (15,545 张真实 D5 latent) 并重测 Oracle
     -> B1 success=False => Route_C_Real_Art_Distribution_Flow_Prior
  -> C2 纯噪声生成先验 credible=False (margin=+0.0089)
  -> C2b/C2c 低/中噪声补救（`partially_noised_real_art_tau0.40`）credible=True margin=+0.0594
  -> C3 200 任务扫描 best=t0.75_g1 gate_passed=False
     config={'ckpt': 'runs/v5/route_c/flow_lowmid.pt', 'tau0': 0.75, 'guide': True, 'seed': 42}
     Style-ID 正确/错误 margin=+0.0014 selective=False
  -> C4 600+150 双 seed 全量评测 => 最终裁决 C_all_models_failed
```

Oracle 失败后，计划明确禁止停在诊断报告：先训练风格表征器，再切真实分布生成先验 Route C；因此 M1/M2 与 Oracle 解蒸馏未被进入，属于决策树规定的剪枝，不是漏跑。

## 1. P0：信息流与条件泄漏修复

- 条件交换测试状态：**passed**，模型在 `style_latent=None` 下仅由 Style-ID 驱动。
- 正确 ID / 错误 ID 的 LL 速度余弦相似度 +0.999999，L2 差 +0.000886；该旧 WEAVE 权重对 Style-ID 几乎不敏感，正是 P0 要暴露的信息流问题。
- 冻结 50 张源图 / 200 个跨风格任务：源图对目标原型相似度 +0.1501，对自身风格原型 +0.6011。
- 固定 VAE posterior mean、seed、独立验证集、CSD 原型与真实 D5 多尺度 latent 描述符（`runs/v5/cache/`）。

## 2. P1：Latent Oracle 是否可解

| 臂 | 可改频带 | λ_struct | CSD ΔT | ΔS | ΔM | DINO-C | ΔT>0 比例 | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| O1_weak | ll | +0.50 | +0.00305 | -0.19855 | +0.20159 | +0.8722 | +0.470 | 4985.0 MiB |
| O1_strong | ll | 2.00 | -0.00121 | -0.17314 | +0.17193 | +0.8889 | +0.490 | 5038.0 MiB |
| O2_weak | ll_lh_hl | +0.50 | -0.00380 | -0.28547 | +0.28167 | +0.7943 | +0.380 | 5038.0 MiB |
| O2_strong | ll_lh_hl | 2.00 | +0.00333 | -0.22852 | +0.23185 | +0.8508 | +0.410 | 5038.0 MiB |
| O3_weak | all | +0.50 | -0.01158 | -0.31666 | +0.30508 | +0.7483 | +0.400 | 5038.0 MiB |
| O3_strong | all | 2.00 | -0.00131 | -0.26697 | +0.26566 | +0.8097 | +0.400 | 5038.0 MiB |
| O_unconstrained | all | +0.00 | -0.01367 | -0.34169 | +0.32802 | +0.6732 | +0.380 | 5038.0 MiB |

**G1 裁决**：`oracle_success=False`，最佳约束臂 `O2_strong` 仅 ΔT=+0.00333；无约束 Oracle 的 ΔT 也为负，说明不是内容约束过强，而是原始 latent 风格目标本身不能把图像推向目标风格原型。进入 B1 风格表征器补救。

## 3. B1：真实 D5 风格表征器与 Oracle 重测

- `LatentStyleEncoder` 在 15,545 张真实作品 latent 上训练 12 epoch，最终验证分类准确率 **+0.8768**，训练耗时 120.1 s。

| 臂 | λ_struct | CSD ΔT | ΔS | ΔM | DINO-C | ΔT>0 比例 |
|---|---:|---:|---:|---:|---:|---:|
| B1_unconstrained | +0.00 | +0.00868 | -0.26323 | +0.27191 | +0.8104 | +0.500 |
| B1_weak | +0.50 | +0.00425 | -0.13008 | +0.13433 | +0.8973 | +0.590 |
| B1_strong | 2.00 | +0.00189 | -0.11016 | +0.11205 | +0.9103 | +0.570 |

**B1 裁决**：`b1_success=False`，最佳臂 `B1_weak` ΔT=+0.00425，仍远低于 +0.03；按计划切换 **Route C 真实艺术分布生成先验**。

## 4. Route C：Style-Conditional Rectified Flow

监督目标：`z_tau=(1-tau)*eps + tau*z_real`，`v*=z_real-eps`；条件只有 Style-ID，不使用参考 GT、不反传 CSD、不使用 GAN/RGB 训练损失。

- 纯噪声生成检查（`pure_noise`）：对角-非对角 CSD 原型 margin **+0.0089**，`generation_prior_credible=False`。

| 生成条件 \ 原型 | ER | Imp | Min | Roc | Uki |
|---|---:|---:|---:|---:|---:|
| ER | 0.028 | 0.078 | 0.216 | -0.003 | -0.002 |
| Imp | 0.026 | 0.097 | 0.195 | -0.000 | -0.011 |
| Min | 0.020 | 0.046 | 0.221 | -0.024 | -0.020 |
| Roc | 0.024 | 0.056 | 0.210 | -0.015 | 0.000 |
| Uki | 0.026 | 0.053 | 0.217 | -0.015 | -0.013 |

### C3：200 跨风格任务扫描

| 配置 | τ0 | 结构引导 | CSD ΔT [95%CI] | ΔM | DINO-C | LPIPS | 5 类中改善 | Ukiyo-e ΔT |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| t0.35_g0 | +0.35 | False | -0.09160 [-0.1097, -0.0718] | +0.42391 | +0.3130 | +0.6542 | 1/5 | -0.21444 |
| t0.35_g1 | +0.35 | True | -0.04308 [-0.0619, -0.0224] | +0.36002 | +0.4608 | +0.5441 | 1/5 | -0.14626 |
| t0.55_g0 | +0.55 | False | -0.06349 [-0.0816, -0.0445] | +0.37148 | +0.4797 | +0.5108 | 1/5 | -0.19254 |
| t0.55_g1 | +0.55 | True | -0.00931 [-0.0289, +0.0118] | +0.30224 | +0.6218 | +0.4023 | 2/5 | -0.13015 |
| t0.75_g0 | +0.75 | False | +0.00325 [-0.0130, +0.0205] | +0.23364 | +0.7902 | +0.2771 | 2/5 | -0.10760 |
| t0.75_g1 | +0.75 | True | +0.02633 [+0.0116, +0.0427] | +0.16606 | +0.8595 | +0.2196 | 2/5 | -0.06871 |

扫描选出 `t0.75_g1`；gate_passed=False。Style-ID 交换敏感度：正确 ID 余弦 +0.1764，错误 ID +0.1750，margin +0.0014，selective=False。

## 5. 最终 600 跨风格 + 150 同风格（双 seed）

| 口径 | CSD ΔT | CSD ΔS | CSD ΔM | DINO-S | DINO-C | CLIP-S | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 600 cross (双 seed 均值) | +0.02890 | -0.14457 | +0.17346 | +0.41624 | +0.80933 | +0.68277 | +0.22004 |
| 150 same (双 seed 均值) | -0.14273 | -0.14273 | +0.00000 | N/A | +0.80967 | N/A | +0.21859 |

- 两 seed 的 600 cross ΔT 差值：+0.00033（随机路线按要求报告平均值与方差，不挑最好 seed）。
- 与同协议旧 WEAVE 600 cross 锚点 ΔT=+0.020284 / DINO-C=0.878826 / LPIPS=0.265337 对照。

| 统一成功标准 | 结果 |
|---|---|
| csd_dT_challenge_0.04 | FAIL |
| csd_dT_beats_old_WEAVE | PASS |
| dino_c_within_0.01 | FAIL |
| lpips_within_0.01 | PASS |
| same_style_no_extra_damage | PASS |

**最终裁决：`C_all_models_failed`**

### 24h 结束必须回答的问题

| 问题 | 结论 |
|---|---|
| VAE_latent_can_express_strong_style_transfer | No: P1 O1/O2/O3 + B1 encoder Oracle all dT<+0.01 |
| original_supervision_wrong | Yes: paired pseudo-GT endpoints replaced by real distribution flow; see p1/b1 reports |
| direct_operator_vs_ODE | Not entered: Oracle failed, so M1/M2 branch was skipped by the decision tree |
| amortisation_bottleneck | Not the primary cause: per-image Oracle itself could not reach target style |
| need_generative_prior_randomness | Route C tested; two-seed mean cross600 CSD_dT=+0.02890 |
| model_to_continue | C_all_models_failed |

## 6. 工程约束、剪枝分支与复现

本轮实际只启动决策树裁决到的路径。计划中的 M1/M2、B2 内容 Pareto、Oracle 解蒸馏属于「Oracle 成功」或「监督修复成功」后的后续分支；本轮 Oracle 与 B1 均失败，按 §4/§6/§8 被剪枝。为满足「尽量都实现好」，这些分支已在 `tools/v5/branches/route_a.py` 与 `run_pruned_branches.sh` 中实现并通过 smoke test（M1 输出零初始化恒等、M2 可微 ODE 有限、监督项接线正常），但没有与 Route C 抢 GPU 并行运行。

- 单 GPU 任务、串行决策树、一个 tmux 驱动；每阶段写 `runs/v5/PROGRESS.md`。
- 训练：15,545 张真实 D5 latent 常驻 GPU（972 MiB），2.28M 参数 flow，有效 batch 256，fp16 autocast，峰值显存约 11.4 GiB（贴近 3060 12GB 上限且不触发 WDDM 分页）。
- 渲染：固定 shape TRT decoder b8 + batch 64 / save_threads 8 档位；750 张 PNG 与 `metrics.csv`/`dino_metrics.csv` 复用 V4 冻结协议。
- 随机生成路线强制两个 seed (42/43)，不允许挑最优单张。

## 7. 产物索引

- `runs/v5/p0_report.json` — 存在
- `runs/v5/v5_eval200_tasks.json` — 存在
- `runs/v5/v5_oracle100_tasks.json` — 存在
- `runs/v5/v5_eval200_baseline.json` — 存在
- `runs/v5/p1_oracle/oracle_matrix_report.json` — 存在
- `runs/v5/b1_style_encoder/b1_report.json` — 存在
- `runs/v5/b1_style_encoder/latent_style_encoder.pt` — 待生成/位于远端 I: Latent_Style WEAVE
- `runs/v5/route_c/flow_full.pt` — 待生成/位于远端 I: Latent_Style WEAVE
- `runs/v5/route_c/noise_check.json` — 存在
- `runs/v5/route_c/quick_report.json` — 待生成/位于远端 I: Latent_Style WEAVE
- `runs/v5/route_c/best_config.json` — 存在
- `runs/v5/route_c/final_report.json` — 存在
- `runs/v5/PROGRESS.md` — 存在
