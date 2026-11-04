# WEAVE 12 小时实验进度（PROGRESS）

## 0. 口径钉死（2026-11-04，优先级最高，覆盖本文件后续一切判据）

**唯一官方评测口径（2026-11-04 19:45 定稿）= 750 all-pairs 全量混合（600 跨风格 + 150 同风格，全部计入）+ 512px + TRT `vae_decoder_512_b8_fp16` + eval batch 8 + decode batch 8 + CLIP ViT-B/32 + DINOv2-small + `ref_policy=first_image_of_target_style_test_dir`。**

锚点 `base_ep10` 512 口径 750 实测（`runs/12h/anchors/base_ep10_750_512/`）：dino_s **0.49023** / dino_c **0.87962** / clip_s **0.70463** / lpips **0.26203**。这是**当前唯一合法对照**。

750 晋级门槛（四条全过才算候选）：dino_s ≥ **0.50523**、dino_c ≥ **0.86462**、clip_s ≥ **0.71463**、lpips ≤ **0.27203**。

**停训交接、可用命令、显存实测、WIP 清单见 §11（唯一有效交接点）。**

（**作废记录**：旧 256 口径锚点 0.48587 / 0.82902 / 0.70940 / 0.25850 及其门槛 0.5009 / 0.8140 / 0.7194 / 0.2685 已于同日作废，不得再引用。作废原因见下条。）

**750 是主判据；600 跨风格 / 150 同风格只作为解释性拆分，不得单独当晋级依据。** 下文 0.0 节中基于 600 跨风格锚点平移出的门槛（.4107/.8099/.6878/.2727）已降级为附注，不再作为结论口径。

**分辨率口径纠正（2026-11-04 终版，覆盖此前"训练降 256"方案）**

- 事实：原始数据是 **512px**；训练一直是原生 **512**（`data/train/.latent_cache/packed/` 为 4×64×64）；被写死成 256 的是**评测**（`tools/eval_fast.py` 里 `image_size=256`，源图按 256 编码成 4×32×32，decode 256）。
- 纠正：**训练不动（512，不重编码、不降分辨率）**；**评测改 512**（`--image-size 512`，源图按 512 编码 4×64×64，decode 用现成的 `vae_decoder_512_b8_fp16.engine`）。P0 特征库、P1/P2 生成图 decode、推理全部 512，三者一致。此前"新建 256 latent 缓存"的方案已取消，不作数。
- **512 锚点重评（评测重跑，不是重训）**：用 `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt` 在 512 口径下重跑 750 all-pairs，产物 `runs/12h/anchors/base_ep10_750_512/`；新锚点与按同规则（+0.015 / −0.015 / +0.010 / +0.010）重算的门槛见下表。
- **256 口径数字全部作废**：旧锚点 0.48587 / 0.82902 / 0.70940 / 0.25850 及其门槛 0.5009 / 0.8140 / 0.7194 / 0.2685，以及 V1 的 0.4971、V2 的 0.47148 / 0.45947 等，均为 **256 口径、已作废**，不得再用于晋级或横比。原因：训练 latent 是 512（4×64×64），256 评测把源图降到 4×32×32，分辨率与训练不一致，评测本身是错的环节。

512 锚点实测（`runs/12h/anchors/base_ep10_750_512/`，`--image-size 512`、b8、TRT `vae_decoder_512_b8_fp16`、8 步 ODE、AdaIN 2.0、wall 169s）：

| 512 口径锚点 `base_ep10`（750 all-pairs） | 数值 | 门槛（同规则 +0.015/−0.015/+0.010/+0.010） |
|---|---|---|
| dino_s | **0.49023** | ≥ **0.50523** |
| dino_c | **0.87962** | ≥ **0.86462** |
| clip_s | **0.70463** | ≥ **0.71463** |
| lpips | **0.26203** | ≤ **0.27203** |

拆分（仅解释性）：600 跨风格 dino_s 0.39186 / dino_c 0.87883 / clip_s 0.66975 / lpips 0.26534；150 同风格 0.88370 / 0.88282 / 0.84417 / 0.24881。

**此后一切判据只用这张 512 表。**

---

方案原文：[PLAN_12H.md](PLAN_12H.md)（逐字保存，冲突时以方案末尾"补充执行要点"为准）。

## 0.0 口径事故与结论（2026-11-04）

### 唯一合法基线（锚点）

`exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt` + **CLIP ViT-B/32**（`openai/clip-vit-base-patch32`）+ b8 + 8 步 ODE + AdaIN 2.0 + 256px + `ref_policy=first_image_of_target_style_test_dir`；产物 `runs/12h/anchors/base_ep10_600/`。

| 子集 | n | dino_s | dino_c | clip_s | lpips |
|---|---|---|---|---|---|
| 600 跨风格 | 600 | **0.39572** | **0.82493** | **0.67779** | **0.26267** |
| 150 同风格 | 150 | **0.84650** | **0.84540** | **0.83587** | **0.24178** |
| 750 全量 | 750 | 0.48587 | 0.82902 | 0.70940 | 0.25850 |

此后跨 run 比较只允许对这张表，且必须同时标注 `eval_batch`。CLIP-L/14 口径已于同日删除，任何 L/14 数字不得与本仓库其他数字横比。

### 0.041 的拆解

旧行 clip_s `.6425` 是 **CLIP ViT-L/14 + `brk_a_ll03_10ep/epoch_0010` + eval batch 2**；锚点是 **B/32 + 同一 ckpt + b8**。

| 效应 | 变量 | clip_s | dino_s | dino_c | lpips |
|---|---|---|---|---|---|
| 主干效应 | L/14 → B/32，同 ckpt | **+0.0353** | −0.0020 | +0.0310 | **−0.0426** |
| ckpt 效应 | base_ep10 → canonical，同 B/32 | +0.0059 | +0.0071 | −0.0217 | — |
| 合计 | | +0.0412（≈0.041） | | | |

**训练贡献为零**：0.041 全部由主干与 ckpt 选择解释，与训练改进无关。

### 附加污染：eval batch 也改生成图

L/14 那次 eval batch=2，B/32 锚点是 b8；同 DINO 主干下 dino_c 差 **0.031**。以后所有数字必须带 `eval_batch`，batch 不同不得横比。

### 结论

A1–A4 相对正确锚点，风格增益 ≤0.0066、内容损失 ≥0.0232，**无一达到基线**。

### 【附注·已失效】旧判据（基于 600 跨风格锚点，不得再用于晋级）

> 以下门槛基于 600 跨风格锚点（.39572/.82493/.67779/.26267）平移，**已降级为附注**；官方口径见文首 750 表。

| 条件 | 门槛 | 锚点值 |
|---|---|---|
| dino_c | ≥ 锚点 − 0.015 = **0.8099** | 0.82493 |
| lpips | ≤ 锚点 + 0.010 = **0.2727** | 0.26267 |
| clip_s | ≥ 锚点 + 0.010 = **0.6878**（才算有风格收益） | 0.67779 |
| dino_s | ≥ 锚点 + 0.015 = **0.4107** | 0.39572 |

**四条全过才算候选，任一条不过就是无效。**

旧判据（`dino_c > 0.78` 或 `lpips < 0.33`）作废：它由错误基线（.7939 / .3053）推出，已从本文档与 `PLAN_12H.md` 移除。

### 【附注·历史】A1–A4 在旧判据下的逐行判定（四个全部不过）

| arm | eval 路径 / batch | dino_s ≥.4107 | dino_c ≥.8099 | clip_s ≥.6878 | lpips ≤.2727 | 判定 |
|---|---|---|---|---|---|---|
| A1 | repro_paper / 32 | 0.42597 ✅ | 0.567629 ❌ | 0.715354 ✅ | 0.448064 ❌ | **无效（2/4）** |
| A2 | repro_paper / 32 | 0.41865 ✅ | 0.743400 ❌ | 0.702548 ✅ | 0.312708 ❌ | **无效（2/4）** |
| A3 | TRT-256-fp16 / 8 | 0.397284 ❌ | 0.797853 ❌ | 0.684442 ❌ | 0.282768 ❌ | **无效（0/4）** |
| A4 | TRT-256-fp16 / 8 | 0.398285 ❌ | 0.801750 ❌ | 0.683209 ❌ | 0.289904 ❌ | **无效（0/4）** |

A1/A2 的 clip_s 虽过线，但由内容崩塌抬升（dino_c .5676/.7434），不能计作风格收益。

## 0. 固定基线（不得中途更换）

| 项 | 值 |
|---|---|
| 评测协议 | `repro_paper`：DINOv2-small + CLIP ViT-B/32 + 750 all-pairs + `ref_policy=first_image_of_target_style_test_dir` + 8-step ODE + AdaIN 2.0 + LPIPS(alex) + fp32 |
| 推理 batch | 固定 32（生成 / metric / decode 均 32；不得中途切换） |
| 复现基线（04:02） | dino_s 0.491502 / dino_c 0.810384 / clip_s 0.712828 / lpips 0.259390 |
| 论文目标 | dino_s 0.4916 / dino_c 0.8103 / clip_s 0.7128 / lpips 0.2593 |
| 参考 `fast_paper_b32` | dino_s 0.491226 / dino_c 0.811184 / clip_s 0.714236 / lpips 0.259333；clip_s 偏 +1.44e-3，**不并入常量** |
| canonical ckpt | `WEAVE/archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt`（禁止移动/覆盖） |
| 判定口径 | 750 四指标 + 600 跨风格 / 150 同风格拆分；FM loss 下降不算成功；**判定门槛（2026-11-04 修正）**：600 跨风格须同时满足 `dino_c ≥ 0.8099`、`lpips ≤ 0.2727`、`clip_s ≥ 0.6878`、`dino_s ≥ 0.4107`（相对锚点绝对门槛，见 0.0 节）；旧判据 `dino_c > 0.78 / lpips < 0.33` 作废 |
| 早停 | 诊断/扫描 run 一律 `internal_early_stop_enabled=false`；正式候选 `min_steps>=2000` |
| 代码 SHA | 本地 HEAD `2fb4387ad336`（远端 WEAVE 副本无 `.git`，注册表用本地 SHA）；T8 的 `flow.py`/`config_schema.py` 改动为未提交工作区改（md5 另记） |

## 1. 阶段总览

| 阶段 | 时间窗 | 内容 | 状态 |
|---|---|---|---|
| P0 | 0:00–0:30 | 诊断代码三修 + 严格 ckpt + schema 两键 + 环境锁定 | ✅ 代码完成（CPU 自测通过），GPU 复跑待空卡 |
| P1 | 0:30–2:00 | A1–A4 小集过拟合；A5–A9 机制诊断 | 🔄 A1–A4 训练完成（A1–A3 已评测；A4 未评测）；A5–A9 未执行 |
| P2 | 2:00–5:00 | T0–T8 训练目标 | ⏳ 未执行（T0/T8 配置已备，未启动） |
| P3 | 5:00–8:30 | M0–M6 容量；S0–S3 Style Memory；S4–S5 条件注入 | ⏳ 未执行 |
| P4 | 8:30–10:30 | R1–R6 组合与真实轨迹 | ⏳ 未执行 |
| P5 | 10:30–12:00 | seed 7/123 复验、完整评测、决策报告 | ⏳ 未执行 |

## 2. P0 明细

| 项 | 要求 | 结果 | 证据 |
|---|---|---|---|
| A3 监督 | 用 `FlowMatchingObjective._build_training_target()` 的真实 endpoint | ✅ 已改：endpoint / x_t / 速度目标全部来自真实目标构造 | `tools/diagnose_saturation.py::analysis_a3` |
| A2 标签方差 | 改 `t=0`、固定风格条件，测多个候选目标的标签方差 | ✅ 已改：`t=0`（x_t==content，预测只算一次），固定 `style_id`+`style_latent`，Top1/4/8 报 cos 均值/标准差/极值 + 标签两两 cosine | `analysis_a2` |
| A4 真实轨迹 | 调真实 `model.integrate_transport()`，不用系数假 AdaIN | ✅ 已改：4 个模式（net_only / real_ode / last_step_adain / every_step_adain）走真实推理路径 | `analysis_a4` + `_temporary_cfg` |
| 严格 ckpt | 打印 expected/missing/unexpected 三列表，非空即失败 | ✅ `strict=False` 仅用于取三列表，任一非空 `SystemExit`；形状不匹配也失败 | `_load_checkpoint` |
| schema | `stop_after_global_steps` / `save_step_milestones` 声明 + 未知键不进 extra | ✅ 默认 `0` / `[]`；未知键抛 `ValueError`；`start_epoch` 历史键白名单忽略 | `config_schema.py::TrainingConfig` |
| 自测 | smoke + 守护测试 | ✅ smoke（CPU, 526,312 参数）跑通；守护测试全绿 | `archives/20261104/p0_guard_checks.py`、`archives/20261104/diagnose_smoke.json` |

P0 代码改动文件（2 个，均未 commit）：

- `tools/diagnose_saturation.py`：A2/A3/A4 三修 + 严格 ckpt（重写，314 → 约 390 行）。
- `config_schema.py`：`TrainingConfig` 新增两字段、未知键校验、里程碑归一化。

一次性验证脚本（非源码，归档目录）：`archives/20261104/p0_guard_checks.py`、`archives/20261104/scan_training_keys.py`。

## 3. P1-A 受控过拟合（A1–A4）

| ID | 数据规模 | 配对 | 预算 | 状态 | 早停 | 750 四指标 | 600/150 拆分 | 与基线差 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 32 | 固定 | 2,000 steps | ✅ 05:27 | 否（disabled） | dino_s 0.4642 / dino_c 0.5673 / clip_s 0.7283 / lpips 0.4450 | 600: dino_s 0.4260, clip_s 0.7154, dino_c 0.5676, lpips 0.4481；150: dino_s 0.6169 | dino_s −0.0273, dino_c −0.2431, lpips +0.1857 | 能拟合（loss 1.23→0.568），但内容严重崩塌；继续 A2–A4 |
| A2 | 128 | 固定 | 2,000 steps | ✅ 05:39 | 否（disabled） | dino_s 0.4923 / dino_c 0.7517 / clip_s 0.7283 / lpips 0.3076 | 600: dino_s 0.4186, dino_c 0.7434, clip_s 0.7025, lpips 0.3127；150: dino_s 0.7869, dino_c 0.7847, clip_s 0.8312, lpips 0.2872 | dino_s −0.0122, dino_c −0.0592, lpips +0.0483 | 拟合变难（loss 1.23→0.887）；仍偏内容崩塌 |
| A3 | 512 | 固定 | 2,000 steps | ✅ 05:48 | 否（disabled） | dino_s 0.4827 / dino_c 0.8029 / clip_s 0.7135 / lpips 0.2781 | 600: dino_s 0.3973, dino_c 0.7980, clip_s 0.6845, lpips 0.2828；150: dino_s 0.8242, dino_c 0.8225, clip_s 0.8293, lpips 0.2594 | −0.0004 / −0.0077 dino_c / +0.0187 lpips | 未退化（dino_c .798 / lpips .283），内容保住；风格无净增益（跨风格 dino_s ≈ 基线）|
| A4 | 128 | 每 epoch 重采样 | 2,000 steps | ✅ 训练 05:56 完成（EXIT=0）；**未评测** | 否（disabled, armed=False） | 未评测（无 `eval_A4`） | 未评测 | — | 跑了但未评测；ckpt `runs/12h/A4/step_002000.pt`；末步 loss 0.9646（LL .0654 / LH .4395 / HL .5055）；cos(v,u)、‖v‖/‖u‖ 无 artifact |

约束：microbatch 不得大于子集长度；不得 `drop_last` 导致有效步为空；每 run 记录 L_LL/L_LH/L_HL、cos(v,u)、‖v‖/‖u‖（训练集 + 固定验证样本）。

**按 2026-11-04 新判据（见 0.0 节），A1–A4 全部判无效**：A1/A2 过 dino_s 与 clip_s 但内容崩塌（dino_c .5676/.7434、lpips .4481/.3127）；A3/A4 四条全不过。上表"判定"列是当时旧口径下的记录，保留不改。

## 4. P1-B 不训练机制诊断（A5–A9）

| ID | 测量 | 状态 | 结论 |
|---|---|---|---|
| A5 | Top1/4/8 目标速度条件方差 | 未执行 | 无 artifact |
| A6 | 五个时间分桶 velocity error | 未执行 | 无 artifact（仅 A3 有一次 CPU 诊断，见第 10 节） |
| A7 | Teacher-forcing vs 自回归 ODE | 未执行 | 无 artifact |
| A8 | 固定内容、交换 style ID | 未执行 | 无 artifact |
| A9 | 真实 ODE / 仅网络 / 不同 AdaIN 位置 | 未执行 | 无 artifact |

## 5. P2 训练目标（T0–T8）

| ID | 方案 | 600 步快筛 | 2,000 步复验 | 750 四指标 | 判定 |
|---|---|---|---|---|---|
| T0 | 原始 FM（对照） | 未执行（`configs/12h/T0.json` 已备） | — | — | 无 artifact |
| T1 | Cross-only（identity_prob=0） | 未执行 | — | — | 无 artifact |
| T2 | 20% 真 Identity | 未执行 | — | — | 无 artifact |
| T3 | Top1 配对 | 未执行 | — | — | 无 artifact |
| T4 | Top4 配对 | 未执行 | — | — | 无 artifact |
| T5 | 无桥噪声（bridge_sigma=0） | 未执行 | — | — | 无 artifact |
| T6 | 早期偏置采样（Logit-Normal μ=-1，须关 legacy compat） | 未执行 | — | — | 无 artifact |
| T7 | Huber | 未执行 | — | — | 无 artifact |
| T8（α=0.5） | 内容锚定 HF 端点 | 未执行（`configs/12h/T8_a05.json` 已备；代码已改并过守护测试） | — | — | 无 artifact |
| T8（α=1.0） | 内容锚定 HF 端点 | 未执行（`configs/12h/T8_a10.json` 已备；代码已改并过守护测试） | — | — | 无 artifact |

T8 判据：显著降低训练难度 **且** 提升 DINO-S 才算有效；只降 FM loss 判"目标有问题"。

## 6. P3 容量与条件结构

### P3-A 容量扫描（先 T0 监督，600 步快筛）

| ID | Width | Depth | 状态 | 750 四指标 | 判定 |
|---|---|---|---|---|---|
| M0 | 64 | 4 | 未执行 | — | 无 artifact |
| M1 | 64 | 8 | 未执行 | — | 无 artifact |
| M2 | 48 | 12 | 未执行 | — | 无 artifact |
| M3 | 96 | 4 | 未执行 | — | 无 artifact |
| M4 | 96 | 8 | 未执行 | — | 无 artifact |
| M5 | 128 | 4 | 未执行 | — | 无 artifact |
| M6 | 96 | 6 | 未执行 | — | 无 artifact |

大模型降低 microbatch、增加梯度累积，保持有效 batch 与总曝光量一致。

### P3-B Style Memory 容量（先把 token 数参数化）

| ID | Memory tokens | 主干 | 状态 | 750 四指标 | 判定 |
|---|---|---|---|---|---|
| S0 | 256 | 64×4 | 未执行 | — | 无 artifact |
| S1 | 64 | 64×4 | 未执行 | — | 无 artifact |
| S2 | 32 | 64×4 | 未执行 | — | 无 artifact |
| S3 | 64 | 96×4 | 未执行 | — | 无 artifact |

### P3-C 条件注入

| ID | 改动 | 状态 | 判定 |
|---|---|---|---|
| S4 | Style-AdaLN 替代单纯 Cross-Attention | 未执行 | 无 artifact |
| S5 | Style-conditioned HF velocity head | 未执行 | 无 artifact |

## 7. P4 组合与真实轨迹（R1–R6）

| ID | 实验 | 启动条件 | 状态 | 判定 |
|---|---|---|---|---|
| R1 | 最佳目标 + 最佳主干 | P2/P3 出现独立收益 | 未执行 | 无 artifact |
| R2 | HF residual 读取 `h.detach()` | 共享主干梯度竞争明显 | 未执行 | 无 artifact |
| R3 | epoch 4 后冻结主干，只训风格分支 | 风格早期峰值后退化 | 未执行 | 无 artifact |
| R4 | 2-step 可微 ODE rollout + 局部 HF 分布损失 | TF/ODE 轨迹差异明显 | 未执行 | 无 artifact |
| R5 | 4-step rollout 对照 | R4 有效 | 未执行 | 无 artifact |
| R6 | Route B 独立伪目标质量测试 | T8 证明内容锚定端点有价值 | 未执行 | 无 artifact |

## 8. P5 最终复验

| 项 | 内容 | 状态 |
|---|---|---|
| seed 复验 | seed 7、123（约 35 min） | 未执行 |
| 完整评测 | 固定 eval 协议 750（约 30 min） | 未执行 |
| 逐风格分析 | 误差归因（约 15 min） | 未执行 |
| 报告 | 200→2,000 steps 指标曲线 + 决策表（约 10 min） | 未执行 |

## 9. 风险与约束

- 只读盘点时（2026-11-04 12:48 CST）GPU 空闲：`nvidia-smi` 475 MiB / 12288 MiB，util 3%，无 tmux 会话，无训练/评测 python 进程。
- 12 小时预算实际消耗：GPU 训练与评测集中在 05:20–05:56（A1 05:20–05:27、eval_A1 05:28–05:31、A2 05:33–05:39、eval_A2 05:40–05:43、A3 05:43–05:48、eval_A3 05:49–05:52、A4 05:52–05:56）；此后至 12:48 无任何实验进程，P1-B 起全部未执行。

## 10. 只读盘点（2026-11-04 12:48 CST）

| ID | 状态 | 关键数字 / 产物 | 证据 |
|---|---|---|---|
| A1 | 有结果 | 2,000 steps；750 dino_s .4642 / dino_c .5673 / clip_s .7283 / lpips .4450；600 跨风格 dino_c .5676 / lpips .4481 → `degenerate` | `runs/12h/eval_A1/results_split.json`；`registry.jsonl` 行 A1_32_fixed |
| A2 | 有结果 | 2,000 steps；750 .4923 / .7517 / .7283 / .3076；600 跨风格 dino_c .7434 / lpips .3127 → `degenerate` | `runs/12h/eval_A2/results_split.json`；`registry.jsonl` 行 A2_128_fixed |
| A3 | 有结果 | 2,000 steps；750 .4827 / .8029 / .7135 / .2781；600 跨风格 dino_c .7980 / lpips .2828 → `ok`；末步 L .9802（LL .0614 / LH .4604 / HL .5014）；CPU 诊断 cos：t[.25,.5) LL .583 / LH .291 / HL .347，t[.5,.75) LL .563 / LH .279 / HL .278；‖v‖/‖u‖ 同桶 .646/.343/.375 与 .564/.332/.329 | `runs/12h/eval_A3/results_split.json`；`registry.jsonl` 行 A3_512_fixed；`runs/12h/diag_A3.json` |
| A4 | 跑了但未评测 | 2,000 steps；ckpt `runs/12h/A4/step_002000.pt`（另有 `epoch_0500.pt`）；末步 L .9646（LL .0654 / LH .4395 / HL .5055）；peak 1.88/2.12 GB；bstep .159 s；无 eval 目录 | `runs/12h/A4.log`；`runs/12h/A4/step_002000.pt`；无 registry 行 |
| A1 机制探针（换参考图 / patch 打乱） | 未执行 | 无输出文件 | 无 artifact |
| A1_probe（100 步 step-time 探针） | 有结果（只记 step-time） | 100 steps；bstep 末 .292 s；无评测 | `runs/12h/A1_probe/step_000100.pt`；`runs/12h/A1_probe.log` |
| A5–A9 | 未执行 | — | 无 artifact |
| T0 / T8 α=.5 / T8 α=1.0 | 未执行 | 配置已备：`configs/12h/T0.json`、`T8_a05.json`、`T8_a10.json`；T8 代码已改（未提交）并过守护测试 | 无 artifact |
| T1–T7 | 未执行 | — | 无 artifact |
| M0–M6 | 未执行 | — | 无 artifact |
| S0–S5 | 未执行 | — | 无 artifact |
| R1–R6 | 未执行 | — | 无 artifact |
| P5（seed 复验 / 完整评测 / 逐风格 / 报告） | 未执行 | — | 无 artifact |

- 磁盘事务已移交独立会话；本线不再做任何清理/搬迁。`I:` 可用约 138G。
- 每个 run 必须落：`config.json`、配置 SHA、代码 SHA、seed、有效 batch、曝光量、四指标、`results.json`，并登记 `runs/registry.jsonl`。
- 所有新开关默认关闭、默认数值行为不变；不 commit（git 线统一提交）。

## 11. Infra 收尾（2026-11-04 19:45，停训后交接，唯一有效交接点）

**GPU / 进程**：无训练、无实验进程；GPU 峰值占用 147 MiB / 12288 MiB（已空）。

**评测路径（唯一可用命令，不要改写）**

```bash
bash tools/12h/eval_path_trt.sh <tag> <ckpt> <out_dir> 8
```

内部等价于 `tools/12h/eval_fast_trt_fp16.py --protocol paper --image-size 512 --batch 8 --decode-batch 8 --metric-batch 32 --decoder trt --workers 4 --no-save-images --gen-dtype fp32 --vae-dtype fp32 --sync-timing --backbone <tag>`；decoder 固定 TRT `vae_decoder_512_b8_fp16.engine`。

冒烟记录（2026-11-04 19:39–19:41，重跑锚点非重训）：`runs/12h/anchors/base_ep10_750_512_smoke/`，750 条，ALL dino_s 0.49023 / dino_c 0.87962 / clip_s 0.70463 / lpips 0.26203，与 §0 锚点**逐位一致**，wall 171 s，EXIT=0。说明 512 评测路径可复现、可直接起跑。

**锚点性质（如实记录）**：`exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt` 是 **512 训练**（latent 4×64×64）的 ckpt，但历史上被 **256 评测**；现按 512 重评，得到 §0 的四个数，是当前唯一合法对照。它仍带 AdaIN 2.0（`tools/eval_fast.py` 内固定 `args.adain_scale = 2.0`），因为该 ckpt 属 V2/ODE 家族。若后续计划要求"无 AdaIN"统一口径，必须**同时重跑锚点**，否则新旧数字不可横比。

**ckpt 约定（固化）**：裸 `state_dict`（不带 `module.` / `_orig_mod.` 前缀），加载时 `SB_STRICT_CKPT=1` 强制严格匹配；`tools/12h/run_v3s.sh` 与训练入口已 `export SB_STRICT_CKPT=1`。

**分辨率 / decoder 事实**

- 训练 latent：`data/train/.latent_cache/packed/packed/*.pt` → `[N,4,64,64]`，512 原生，不重编码、不降分辨率。
- 评测：`--image-size 512` + TRT 512 engine（fp16）+ eval batch 8 + decode batch 8；TRT 512 解码已验证 PSNR 59.12 dB。
- 训练侧如需**可微 decode**（生成侧特征要反传到 latent），必须另起一个 torch/diffusers VAE（`v2/objective_swd.py::decode_latents`，**仅训练用**，VAE 参数 `requires_grad=False`）。**评测侧一个字没动**：engine、image_size、decode-batch、metric 定义全部原样。

**VRAM 实测（12 GB，512 decode 且梯度反传到 latent）**

| 配置 | 峰值显存 | 每张耗时 | 结论 |
|---|---|---|---|
| fp32 权重，microbatch 1 | 7.00 GB | 1.09 s | 可用，但在自证脚本组合场景下仍 OOM |
| fp32 权重，microbatch 2 | OOM | — | 不可用 |
| bf16 权重，microbatch 1 | 3.79 GB | 0.65 s | 可用 |
| bf16 权重，microbatch 2 | 7.06 GB | 0.64 s | 可用（上限） |
| bf16 权重，microbatch 4 | 13.58 GB | 16.2 s（抖动） | 不可用 |
| bf16 vs fp32 decode 输出 | — | — | PSNR 50.08 dB，max\|Δ\| 0.030，精度可接受 |

结论：有效 batch 32 只能做到 **microbatch 2 × accum 16（bf16）** 或 **microbatch 1 × accum 32（fp32）**；按 bf16 b=2 估算 ≈20–21 s/step，4000 步 ≈23 h。

**梯度检查点：未启用。** 只在一次性临时探针里试过（bf16 + ckpt 在 microbatch 4 峰值 12.29 GB 仍 OOM，b=8 OOM），收益不足，**正式代码里没有任何 checkpointing 路径**，探针文件已删除。

**本轮改动清单（改了什么 / 为什么 / 是否影响评测口径）**

| # | 文件 | 改了什么 | 为什么 | 影响评测口径 |
|---|---|---|---|---|
| 1 | `tools/eval_fast.py` | 新增 `--image-size`（默认 512），把原先写死的 256 全部参数化（源 latent 缓存键、源图加载、签名、结果元数据）；variant 白名单加 `V3S` | 用户要求"评测改 512"，修掉写死 256 的错位 | **是**（本次已用冒烟+锚点验证） |
| 2 | `tools/12h/eval_path_trt.sh` | 显式加 `--image-size 512` | 固化口径，避免默认值漂移 | **是**（同上） |
| 3 | `tools/12h/summarize_results.py` | 新增；读 `results.json` 打 ALL/CROSS/SAME 四指标 | 汇总便利 | 否（只读） |
| 4 | `docs/12h/PROGRESS.md` | §0 口径改写 + §11 交接 | 记录唯一合法口径 | 否 |
| 5 | `v2/model.py` | 输出头支持 `residual_head_zero_init`（默认 `False`） | V3-SWD 零初始化头 | 否（默认行为不变，V2 路径不变） |
| 6 | `v2/config.py` | 新增 V3-SWD 字段（全部有默认值）；白名单加 `V3S` | 配置承载 | 否（默认值不改既有路径） |
| 7 | `v2/train.py` | 新增 `train_swd_variant` / `_swd_diagnose`，仅 `--variant V3S` 进入 | P1 训练入口 | 否（**WIP，默认不启用**） |
| 8 | `v2/inference.py` | `_GENERATOR_VARIANTS` 加 `V3S` | 推理识别新 variant | 否（仅显式 variant=V3S 生效） |
| 9 | `v2/features.py` `v2/distances.py` `v2/data512.py` `v2/objective_swd.py` `v2/style_encoder.py` | 新增 V3-SWD 专用模块 | P0/P1 | 否（默认路径不 import） |
| 10 | `tools/build_style_library.py` `tools/build_style_encoder.py` `tools/12h/selfcheck_v3s.py` `tools/12h/run_v3s.sh` `configs/v2/V3S.json` | 新增 P0/P1 工具与配置 | 特征库 / 自证 / 一键训练（**未开训**） | 否 |
| 11 | `v2/objective_swd.py` 内 `decode_latents` | 训练专用 diffusers VAE 可微 decode | 生成侧特征梯度要回到 latent | 否（**仅训练侧**，评测仍走 TRT） |
| 12 | `tools/12h/_probe_decode.py` | 临时显存探针 | 实测显存 | 否（**已删除**，本地+远端） |

**WIP 产物（默认关闭 / 未启用，不阻碍起跑）**

- 特征库：`runs/v2/swd/library_5styles_512.pt`（无 encoder）与 `library_5styles_512_enc.pt`（含 128 d encoder，112.4 MB）。enc 版内嵌报告：类内半径 78.26 < 类间质心距 98.27（ratio 1.26），最近质心留出准确率 0.792（随机 0.20）。`runs/v2/swd/library_report.json` 对应**无 encoder 旧版**，勿混用。
- Style Encoder：`runs/v2/swd/style_encoder_512.pt`（val_acc 0.6420，只用真实作品训练、已冻结）。
- V3-SWD 代码：**未开训**。自证进度如实记录——已过：零初始化头（max|W|=0）、identity（`max|G(x)-x|=0.0`、decode PSNR ∞）、SWD 下降（11.44→0.033，max\|grad\| 2.6e-2）、MMD 下降（0.515→0.011，max\|grad\| 9.5e-4）、损失确定性、特征库类内<类间；**未完成**：hinge=0、生成器梯度非零、py_compile——脚本在 512 可微 decode 阶段 CUDA OOM 中断（fp32 microbatch 1 单独可跑 7.0 GB，组合场景仍不够）。下次起跑请直接用 bf16 权重 microbatch 2。

**已作废 / 已停止路线（不得再引用）**

- **256 口径一切数字**（旧锚点 0.48587/0.82902/0.70940/0.25850、旧门槛、V1 0.4971、V2 0.47148/0.45947）：训练 512 / 评测 256 不一致，作废。
- **"训练降 256 + 新建 256 latent 缓存"**：用户否决（原图与评测都是 512，应改评测而不是改训练），作废。
- **GAN-V3**（对抗损失 / FM / PatchNCE / 目标像素回归）：用户叫停，**未写实现、未开训**即作废；原因 = 与分布匹配路线冲突、训练不稳定。当前 V3 已改为 V3-SWD（无 GAN / 无 FM / 无 PatchNCE / 无目标像素回归）。
- **V2（style-statistics/AdaIN 家族）**：历史结论保留在 §0.0 / §10，但其中所有数字均为 256 口径，已作废；V2 不再作为候选路线。
