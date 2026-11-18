# V22 进度文档（P1 主模型正确化 + P2 One-Shot）

**计划**：`E:\writing-block (2).md`（V22-P1）、`E:\writing-block (1).md`（V22-P2） · **设备分工**：4090 主力跑 P1 消融队列；3060 跑旁路任务（P2-0 双分支重训 + 风格编码诊断）
**基准提交**：`731110e` → P0 完成于本文档首次提交（`bdb1a83` 之后）

---

## 状态总览

| 阶段 | 状态 | 产物 |
|---|---|---|
| P0 统一接口 + 数值单测 | ✅ 完成（4090 上 15/15 通过） | `exp/v22/core.py`, `tests/v22/test_core.py` |
| P0 Fix 1（T4 缓存版本绑定） | ✅ 完成 | `exp/v22/t4.py`（四种缓存情形：v22缓存/原始raw/不完整/版本变化） |
| P0 Fix 2（Same150 旁路拆分） | ✅ 实现，待队列实测 | `S_blind_same150` / `S_self_ref_same150` |
| P0 Fix 3（反事实锚点对照） | ✅ 实现，待队列实测 | `CF2_wrong_tgt_same_t4` / `CF4_wrong_tgt_chroma_anchor` |
| P0 Fix 4（缓存/版本绑定 + 750 唯一性校验） | ✅ 完成 | `validate_raw_meta`；meta 记录 git_sha/ckpt_sha256/bank_sha256/tasks_sha256/t4_hash/seed |
| P0 Fix 5（V21 无放回参考采样） | ✅ 完成 + 单测 | `exp/v21/prism_attn_flow.py::sample_without_replacement` |
| P0 复现核验（D5/D6/D2） | ✅ **PASS**（三方比对归因） | `runs/v22/p0_repro_report.json` |
| P1 消融队列（23 新臂 + 5 alias） | 🔄 4090 tmux `weave-v22` | `runs/v22/queue_state.json` |
| P2-0 双分支重训 | 🔄 3060 | `exp/v22/train_dual.py` |
| P2 风格编码诊断 | 🔄 3060 | `exp/v22/reference.py` |

---

## P0 复现核验：结论与归因（2026-11-19）

### 三方比对设计
1. **V20 参考实现同进程重跑**（`integrate_slf_core_batch`）——排除实现差异；
2. **V22 统一求解器**（`PrismCore.integrate_flow`）——被测对象；
3. **原始 raw latents**（`runs/v20/raw/V20_D*_core_slf_*`，原始进程产物）——跨进程 GPU 非确定性底线。

### 结果（Cross600 600 任务；Same150 bypass 150 任务全部逐位一致）

| 臂 | V22 vs V20重跑 | V20重跑 vs 原始 | V22 vs 原始 | 判定 |
|---|---:|---:|---:|---|
| repro_D5 | **0（逐位）** | 7.73e-3 | 7.73e-3 | 等价 |
| repro_D6 | **0** | 4.82e-3 | 4.82e-3 | 等价 |
| repro_D2 | **0** | 6.39e-3 | 6.39e-3 | 等价 |

**归因**：V22 与 V20 在同进程同 GPU 下**逐位相同**（diff=0.00e+00），证明统一接口是忠实重构。与原始 raw 的偏差（max 4.8–7.7e-3，mean ~8.9e-5）在两条实现上**完全一致**，纯由跨进程 GPU 浮点非确定性（cudnn autotune + bf16 autocast）造成，不构成实现回归。PASS 判据：`impl_bitwise_equal && v22_vs_orig ≤ gpu_nondet_floor × 1.5`。

### 过程中发现并修复的真实 Bug
1. **prog 浮点不等价**：初版 V22 传 `prog + 0.5/steps` 给速度场，V20 是 `(i+0.5)/steps`——浮点不等价（如 `1/10+0.5/10 ≠ 1.5/10`），经时间嵌入放大造成 max 7.7e-3 偏差。修正后 `v22_vs_v20 = 0`。验证了 P0.3 单测对积分器逐位行为的敏感性。
2. **`ensure_t4_latents_750` 引用不存在的 `run_aniso_teacher_arm`**（Fix 1）：T4 缓存存在时从不触发；缓存缺失时会 ImportError。已由 `exp/v22/t4.py::build_or_load_t4_latents_750`（配置 hash `04007b4f0355`）替换，`exp/v20/run_v20.py` 改为调用它。
3. **V21 4-Shot 有放回采样**（Fix 5）：`torch.randint(0,12,(B,n_shots))` 允许同一参考重复；改为 `sample_without_replacement`（random scores topk），单测验证 64×4 无重复。

### 单测覆盖（15/15 通过，CPU/GPU 均可运行）
- **P0.2**：四分支批量 vs 四次独立前向逐位一致；drop_style 后 tid/u 不变性；drop_content+lscale=0 后内容不变性；`wc=wj=wp=1` 代数恒等式 `v=vsc+vs0-v00`；predict_guided_velocity 与手动组装（含 rescale）一致。
- **P0.3**：常速度场 RK2 精确；线性时间场中点律精确；per-sample tau0；Heun 步长收敛；rescale_phi 零方差稳定性（ConstModel 无 NaN）；同 seed 逐位确定性。
- **等价性**：`test_equivalence_with_v20_integrate`（CPU，实测 0 差）。
- **门控解耦**：original/fixed/C1 三种模式 g 分量正确性。
- **Fix 4/5**：`validate_raw_meta`（750 唯一 id）；无放回采样。

---

## P1 消融队列实测（滚动回填）

### A组：锚点贡献（✅ 完成）

| 臂 | Top-1 ↑ | DINO-S | CLIP-S | Affinity | DINO-C | LPIPS | ms/img | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `A0_pure_noise`（无任何锚点） | 45.83% | 0.5467 | 0.7393 | 0.6430 | 0.5550 | 0.5811 | 91.5 | 无锚点 Flow 仍有强生成能力，Aff 已追平 StyleGallery 0.6432 |
| `A1_src_anchor`（源 latent） | 19.83% | 0.4695 | 0.6931 | 0.5813 | **0.7464** | **0.2948** | 91.2 | 源锚点严重源锁；Flow 不做"微编辑" |
| `A2_prism1_anchor`（PRISM-1 学生） | 57.17% | 0.5163 | 0.7295 | 0.6229 | 0.6103 | 0.5640 | 91.1 | **PRISM-1 可替代 T4 教师**：-3.3pp Top-1 换取锚点成本 137ms→~5ms |
| A3 = D5（T4 教师，基线） | 60.50% | 0.5672 | 0.7497 | 0.6584 | 0.5335 | 0.6022 | 134 | 对照 |
| A4 = T4-only（V19 官方 measured） | 59.00% | 0.4618 | 0.6963 | 0.5791 | 0.6783 | 0.5166 | 137 | Flow 相对 T4-only：+1.5pp Top-1 / +10.5pp DINO-S |

**A组结论**：① T4 教师锚点贡献 +40.7pp Top-1（60.5% vs 源锚点 19.8%），是最大单项；② Flow 的本质是"从结构化锚点继续生成"，而非编辑源图；③ PRISM-1 学生锚点以 -3.3pp Top-1 / +7.7pp DINO-C 换取锚点成本 ~27× 降低，构成新的速度-质量 Pareto 候选。

### B组：CFG 分支（部分完成，🔄 B4/C组继续中）

| 臂 | Top-1 | DINO-S | Affinity | DINO-C | LPIPS | ms/img | 发现 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0_sc_only`（单分支，无 CFG） | 45.50% | 0.5195 | 0.6171 | 0.6518 | 0.5473 | **33.9** | 单前馈 2.7× 提速档位 |
| `B1_cs_j`（双分支 v0c+wj(vsc-v0c)） | 42.50% | 0.4902 | 0.5937 | 0.7007 | 0.5228 | 53.6 | P2 §2.1 双分支公式：内容强、风格弱 |
| `B2_no_wp`（去纯风格分支） | 48.83% | 0.5191 | 0.6180 | 0.6601 | 0.5436 | 91.4 | wp 贡献 +11.7pp Top-1（D5 60.5→48.8） |
| **`B3_no_wc`（去内容引导 wc）** | **67.00%** | 0.5579 | 0.6521 | 0.5047 | 0.6101 | 91.4 | **新 SOTA**：+6.5pp vs D5，+14.5pp vs StyleGallery；Min 100% / Roc 64.2% / Uki 45.8% |
| `B4_no_rescale` | 🔄 | | | | | 91.3 | |
| B5 = D5（全四分支） | 60.50% | 0.5672 | 0.6584 | 0.5335 | 0.6022 | 134 | 对照 |

**B组核心发现（四分支 = 风格-内容权衡拨盘）**：`wp`（纯风格方向）强推 Top-1（去掉 -11.7pp）；`wc`（内容引导）反压 Top-1（去掉 +6.5pp）但换 DINO-C（0.5047→0.5335）。这**部分反驳**了 P2 §2.1 的假设——`v_s0` 并非"内容破坏源"，而是风格特异性的主要引擎；真正需要的是 `wc/wp` 连续旋钮（P3-N3）而非直接删除。`B1`（双分支）内容最好（DINO-C 0.7007）但风格崩（0.4902），提示 P2-0 重训需配 `wp` 保留或更强参考条件。

### P2 诊断（3060，✅ 完成）

- **fiber 分离比**：同风格 vs 跨风格 128-D fiber 距离 = train **8.80×** / holdout **5.81×**（same 0.169/0.254 vs cross 1.49/1.48）→ StyleFiberEncoder 对留出画作具强风格分辨力，Reference Adapter（§3.2）前提成立（详见 [`p2_reference_diagnostics.json`](p2_reference_diagnostics.json)）。
- **RGB→latent encoder**：候选权重（`sd0_s10_best_*`, `a0_encoder_only`, `C_edit_best`）在本地 `deliver/checkpoints/` 存在；3060 侧无。P2-2 需先做 0.18215 scaling 双向编码验证（未验证前不得宣称端到端 RGB→RGB）。
- **Start-Process 教训**：3060 上 PowerShell `Start-Process` 分离进程在 ssh 会话结束后被静默清除（日志 0 字节、无错误）；改用同步 ssh + 本地后台任务托管。

---

## P2（3060 旁路）

1. **P2-0 双分支重训**（`train_dual.py`）：从 `slf_final.pt`（5000-step 预训练）出发，`drop_content=False` 恒成立、`layout_scale=1.0`、drop_style=0.12、80% Core 采样、1800 updates——修复"纯风格分支 CFG 把模型变成风格条件生成"的根因。
2. **风格编码诊断**（`reference.py`）：① 同/跨风格 fiber 距离比（判定 128-D StyleFiberEncoder 对未见风格分辨力）；② 固定内容换 5 参考（SG-5）输出敏感度（双向防"忽略参考"与"参考内容泄漏"）；③ RGB→latent encoder 可用性盘点（P2-2 前置）。

## 下一步
- 4090 队列完成后：回填各组 750 实测表 + 配对 bootstrap 等效区间判定（Top-1 ±2pp / DINO-S ±0.005 / DINO-C ±0.01 / LPIPS ±0.01）→ 产出 `PRISM-Core-Simple` 候选。
- 3060 双分支模型完成后：先以 B1 分支（双分支推理）快评，再上 4090 正式 750 评测。
