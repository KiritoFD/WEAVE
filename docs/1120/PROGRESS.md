# V26 执行日志（PROGRESS）

> 队列：4090 `tmux v26k`（`exp/v26/gpu4090/queue_k.py`）／3060 `schtasks v26p0`（`exp/v26/gpu3060/queue_p0.py`）
> 全部步骤幂等（已有 measured summary / 输出 JSON 自动跳过）；单步失败不中断队列。
> 本文件由总控实时更新。

---

## 2026-11-21 启动

- **代码落地**：`exp/v26/` 公共层（`sr_cfg` / `diagnose_q32` / `diagnose_refattn` / `k2_style_subject_audit` / `dynamic_style_swd` / `sg25_hook`）+ 双卡队列；`tests/v26/` 13 契约 / 19 测试全绿。提交 `b0ff40b`（含 C2nt 降级决策 `docs/1120/07_c2nt_demotion_decision.md` 与全套文档修订）。
- **4090**：`git archive HEAD | tar -x` 同步 → tmux `v26k`（`/opt/conda/envs/cu121/bin/python`）。
- **3060**：scp 同步 → `.bat + schtasks v26p0`（`C:\Program Files\Python312\python.exe`）。
- **两处运行时 bug 修复并重启动**：① `Tensor.norm(dim=(1,2,3))` 在 torch 2.5 触发 `linalg.matrix_norm` 限制（K0 / K1-C）→ 改 flatten+`norm(dim=1)`；② 3060 队列 `_few_step`/`_swd_arm` 缺函数内 import → NameError。均已修复、同步、重启（幂等跳过已完成臂）。
- **协议级 bug 发现与全部 SR/FS 臂作废重跑（重要）**：K1-A 逐样本化修复后暴露 `u_glob` 取数错误——`load_sr_env` 误用 `fib_n_750[tid]`（test750 样本纤维）而非协议要求的 **SG-5 目标画派纤维 `fib_n_sg5`**（`r3.eval_o_tau_list` 口径）。受污染臂：4090 `SR1–SR7` 全部（`u_glob_t` 错误，解释了 FS/SR 早期"近恒等输出"读数），3060 `FS_*` 全部。已修复（`load_sr_env` 现场提取 SG-5 5 张参考图的 795D 纤维）、两机作废重跑（提交 `71c3f1d` + ROOT 导入修复）。
- **保留有效的早期读数**：`SR0_t45` 复用 V25 raw（不受影响，读数逐位复现 42.17%）；`selfref_identity_audit`（自参考纤维口径正确）有效——**发现 ref==src 时输出显著偏离源（latent MAE 0.451，DINO-C 0.707，identity_pass=False）**：C2nt 的参考条件并不支持自参考恒等，Same150 bypass 是必要的协议补丁而非模型能力。
- **第二次重大 bug：CFG chunk 顺序反转（已修复，全部 SR1–SR7 / FS / K1-B 作废重跑）**：baseline 分支误写为 `v_pos + wj*(v_sc − v_pos)`，而协议口径（`r3.eval_o_tau_list`）为 `v_uncond + wj*(v_cond − v_uncond)`——等效**负 CFG**，导致 FS/基线全部"近恒等输出"（Top1 0.8–2.3%）。回归测试契约 15 已固化（`test_baseline_cfg_direction`）。同时新增 docs 公平对照 1 要求的**单条件分支臂 `SR0b_pos`**（v_pos alone，与 SR1–SR6 同 NFE 比较）。u_glob 修复后的 SR1–SR6 旧读数（Min 98–100% 现象）实为"无 CFG 锚定的 v_pos+排斥"行为，不作结论。
- **K1-B 有效结论（作废前的一次有效观察，待重跑确认）**：R0–R6 干预对 `dino_s_vs_target_ref` 的影响 ≤ ±0.02（R3 V=0 反而 +0.02）——**16×16 局部参考注意力在当前 C2nt 中近乎惰性，参考响应主要由全局风格通路承载**。此结论若重跑复现，将直接指导 PRISM-Flow 任意参考化设计（优先全局弦通路而非局部注意力）。
- **4090 为共享 GPU**：机上另有外部推理服务占用 ~12GB 显存（host PID 589217/1091267，**不得触碰**）；本队列所有 VRAM 预算按 ≤12GB 约束运行（已验证 SR 全矩阵可在此水位完成）。

## 实时结果（滚动追加）

### 4090 — SR-CFG 矩阵（C2nt t45 基线 SR0 复用 V25 raw，读数逐位复现：DINO-C .5448 / D-S .5235 / Top-1 42.17%，分画派 5.8/93.3/90.8/0.0/20.8 ✓）

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Top-1 ↑ | 备注 |
|---|---:|---:|---:|---:|---|
| SR0 (基线, w=0) | .5448 | .5235 | .5205 | 42.17% | Min 90.8 / Imp 93.3 / Roc 0 |
| SR1 (w=0.3, P=I) | .6109 | .5173 | .4744 | 22.33% | **Min 90.8→15.0 崩塌**；内容↑失真↓但风格↓ |
| SR2 (w=0.6, P=I) | .5899 | .5215 | .4922 | 24.67% | Min 21.7；同方向 |

**早期判读（待 K1-C 速度场归因确认）**：源参考差分引导**未**打破源锁定——反而整体降低风格化强度（Top-1 ↓，极简路线崩塌），同时提升内容保持。初步证伪"当前 C2nt 速度场差分可直接抵消源锁定"（docs 预注册的 FAIL 出口之一，不否定 Source-Style Lock 本身）。方向几何解释待 `K1-C`（‖v_pos‖ vs ‖v_src‖ / cos(diff, v_pos)）。

### 3060 — P0-D 少步扫描

（运行中，滚动追加）

---

## 产物回传红线

- 4090 每阶段完结：`runs/v26/gpu4090/**` → `G:\WEAVE\runs\v26\gpu4090\`（scp，校验 SHA256）。
- 3060 队列完结：`runs/v26/**` + `runs/v25/sg25_outputs/`（SG-25 新 tag）本地归档。
- 报告：`docs/v26/4090/K0_Q32_FEATURE_AUDIT.md` / `K1_REFATTN_CAUSAL_AUDIT.md` / `K2_STYLE_LOSS_AND_SUBJECT_AUDIT.md`（由各模块自动生成）。
