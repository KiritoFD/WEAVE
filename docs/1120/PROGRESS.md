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
