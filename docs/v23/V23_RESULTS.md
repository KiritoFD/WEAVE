# V23 实验阶段结题报告：Content-Preserving PRISM & True Reference-Conditioned One-Shot

> **基准提交**：`4429213` → P0 修复 `4a19326` → 本报告收官  
> **评测协议**：WEAVE `test750`（Cross600 + Same150）× `S10` 解码 × 独立进程隔离四裁判（`src.eval.measure`）；One-Shot 同步扩展至 StyleGallery 官方 25 类 OOD 125-pair 端到端实测  
> **详细日志**：`docs/v23/PROGRESS.md`；**统一台账**：`docs/performance/metrics.csv`（V23 新增 67 条完整评测臂，总台账 373 条）

---

## 一、核心结论摘要（Executive Summary）

V23 及时修正了 V22 阶段“偏重刷高 Top-1 却导致 DINO-C 偏离 Pareto 前沿”的方向偏差，围绕 **双分支 bug 修复、高 DINO-C (`≥0.60–0.63`) 内容保真突破、以及无标签任意参考 One-Shot 跨风格监督** 三条主线取得以下核心成果：

1. **P0 修复推翻旧结论，半 NFE 真双分支立住高内容前沿**：
   - 修复 `exp/v22/core.py` 与 `exp/v22/ref_adapter.py` 中双分支 `cs_j` 第二分支 `v_0c` 误将 `layout_scale` 清零的严重 bug（补齐 5 项单测，24 passed）。
   - 修复后，`V20 Core-SLF` 双分支 Top-1 从 `42.50% → 50.50%` (**+8.00pp**)、DINO-S 从 `0.4902 → 0.5263`（以一半 NFE 反超 D5 四分支 `0.5214`，且 DINO-C `0.6408 > 0.6287`）；`V22 Dual` 双分支 Top-1 从 `34.67% → 49.50%` (**+14.83pp**)、DINO-S **`0.5522`**、DINO-C **`0.6096`**。
2. **高内容前沿（`DINO-C ≥ 0.60–0.63`）取得显著外推与全盲对比优势**：
   - **病灶精准定位**：逐步轨迹解码（`p1_content_leak_diag.py`）证明内容结构不是在后半段逐步丢失，而是在 **Step 0 (`p00`, $t=\tau_0$)** 第一次预测即被抹糊（`PatchCos 0.6515 → 0.5120`）。
   - **两项真实跨前沿训练机制**：
     - **ConvNeXt 中尺度空间余弦损失 (`C1`)**：推动 `hiS` 至 `78.33% / 0.4843 DINO-C`（Early_Renaissance `+10pp`）；
     - **25% Anchor-Bridge (`AB1`)**：消除训练-推理起点不匹配，在 `PRISM-1` 锚点上稳定提升 `DINO-C (+0.007)` 与 `DINO-S (+0.005)`（Ukiyo-e `+11~16pp`）。
   - **高 DINO-C 旗舰工作点**（全部 20 NFE 双分支或免 T4 锚点，经视觉核验无坍缩伪影）：
     - **`HiC_JPk_csj65_t4` (`DINO-C = 0.6299`)**：**`58.83% Top-1 / 0.5330 DINO-S / 0.5649 LPIPS`** —— 同 `DINO-C` 水平全面超越原主线 `V20 D5` (`+3.66pp Top-1, +0.0116 DINO-S`) 且 NFE 减半。
     - **`Joint_T1_C1AB_p1_hiC` (免 T4, `PRISM-1` 锚点)**：**`60.83% Top-1 / 0.5307 DINO-S / 0.6014 DINO-C`**；**`Joint_T1_C1AB_p1_bal`**：**`58.00% Top-1 / 0.6258 DINO-C`**。
     - **`DualSWD_pure_hiT_ws52` (`DINO-C = 0.6164 > SG 0.6113`)**：**`51.17% Top-1 / 0.5467 DINO-S / 0.5674 LPIPS`**；**`F3_dual_fixed_orig`**：**`49.50% / 0.5522 DINO-S / 0.6096 DINO-C`**。
   - **诚实边界披露**：在 `DINO-C ≥ 0.611` 区间，`PRISM` 的 `DINO-S`（`0.547–0.552`）仍距 `StyleGallery` (`0.5604`) 差约 `0.008–0.013`——全套受控实验证实，仅靠 `16×16` 瓶颈传入结构、在 `32×32/64×64` 保持无空间约束的 22M `SLFUNet`，在此高保真区间存在内在的“锚点频率上限 vs 源构图保留”交换边界。
3. **One-Shot 任意参考迁移从机理判负走向链路打通（`12.2% → 38.5%`）**：
   - 隔离出 `onehot=0` 清零造成的 `-2.67pp` OOD 偏移（可由训练集线性探针 100% 恢复）；
   - 通过同构 `R4/R5/R6`（均停在 `19.5–19.7%`）对比 **`M1_cross25 (28.5%) → M2_cross50 (31.8%) → M3_ab_cross50 (srcA0 38.0%, noise 62.5%) → O7_direct_app (srcA0 38.5%)`**，铁证：**缺乏跨风格配对监督与源图→跨风格 Anchor-Bridge 是此前 One-Shot 失败的根本原因，而非网络容量不足**；
   - **全链路端到端 RGB→RGB 闭环打通**：在 3060 上验证 `A0 Encoder + S10 Decoder` 往返 **`DINO-C = 0.9710`**（`MAE 0.0189`），实现 **1.63 s/img (`2.0 GB VRAM`)** 的无类别端到端推理；但 StyleGallery 25 类 OOD 125-pair 测试（`DINO-S=0.262, DINO-C=0.420`）诚实揭示：**只用 WikiArt 5 类画派训练的底座与 1000 对 5 类伪配对尚不足以泛化到二次元/水墨/霓虹等 25 个分布外新风格，下一阶段（V24）必须扩充多元风格训练池（§6.5）**。

---

## 二、高 `DINO-C (≥0.60)` 前沿核心模型总对比（Cross600）

| 模型 / 配置 | 锚点 | NFE | Top-1 ↑ | DINO-S ↑ | Aff ↑ | DINO-C ↑ | LPIPS ↓ | 定位与状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`StyleGallery`** (外部扩散) | — | 50×opt | 52.50% | **0.5604** | 0.6432 | 0.6113 | 0.5576 | 外部强基线 (`30.4 s/img`) |
| `V20 D5` (原主线) | T4 | 40 | 55.17% | 0.5214 | 0.6244 | 0.6287 | 0.5529 | V20 均衡对照 |
| **`F3_dual_fixed_orig`** | T4 | **20** | 49.50% | **0.5522** | 0.6419 | **0.6096** | 0.5723 | 修复双分支，全仓最近 SG 前沿 (`-0.008 D-S`) |
| **`DualSWD_pure_hiT_ws52`** | T4 | **20** | 51.17% | 0.5467 | 0.6384 | **0.6164** | 0.5674 | 超 SG 内容保真 (`+0.005 D-C`)，视觉审计通过 |
| **`HiC_JPk_csj65_t4`** | T4 | **20** | **58.83%** | 0.5330 | 0.6301 | **0.6299** | 0.5649 | 同 DINO-C 全面击败 V20 D5 (`+3.66pp / +0.012 D-S`) |
| **`Joint_T1_C1AB_p1_hiC`** | **PRISM-1** | 40 | **60.83%** | 0.5307 | 0.6304 | **0.6014** | 0.5817 | **免 T4 快速全链路 (`~225 ms/img`) 破 60%** |
| **`Joint_T1_C1AB_p1_bal`** | **PRISM-1** | 40 | 58.00% | 0.5243 | 0.6250 | **0.6258** | 0.5703 | 免 T4 高内容档 |

---

## 三、所有否决 / 中性机制清单（诚实关闭）

| 探索方向 | 实验臂 | 结论 | 终止原因 |
|---|---|---|---|
| **零初始化 32×32 残差注入** | `S32Zero` vs `S32Ctrl` | **REJECTED** | 同预算对比差异仅 `±0.17pp / ±0.0008 DINO-C`（噪音内） |
| **Chroma-Shift 无纹理锚点** | `A2_pk_chroma` | **REJECTED** | Top-1 暴跌至 `24.5%`，证实推理极度依赖锚点纹理先验 |
| **解耦低 `g_layout` 固定门控** | `Dual_dec_gt68_gl35` | **REJECTED** | 松开布局未涨 DINO-S 反而砸掉 DINO-C `-0.06`；`orig` 门控更优 |
| **自配对下 `16×16` RefAttn** | `R4 / R5 / R6` | **诊断锁定** | 无跨风格监督时三条曲线重合在 `19.5–19.7%` |
| **5 类内直通无分类外观投影** | `O7_directapp` (SG-25) | **局限性确认** | `test750` 升至 `38.50%`，但 SG-25 OOD 仍受限于 5 类训练集覆盖 |

---

## 四、3060 One-Shot 演进全路径（无测试目标标签）

| 阶段 | 臂 | 起点 | Top-1 ↑ | DINO-S ↑ | DINO-C ↑ | LPIPS ↓ | 关键新增机制 |
|---|---|---|---:|---:|---:|---:|---|
| 基线 | `P2A_1shot_cs_j` | noise | 13.50% | 0.4482 | 0.6623 | 0.5295 | 冻结主干 + 有 bug 双分支 |
| P0-A | `P2A_1shot_cs_j_fixed` | noise | 14.00% | 0.4839 | 0.6316 | 0.5011 | 修复 `v_0c` 内容条件 (`+0.036 D-S`) |
| P2.1 | `R2 / R3` (联合微调) | noise | 20.83% | 0.5021 | 0.6169 | 0.5204 | 解冻主干 + 无类别 `onehot=0` |
| P2.2 | `R5 / R6` (`16×16` RefAttn) | noise | 19.67% | 0.4997 | 0.6206 | 0.5173 | 自配对无法激活局部参考注意力 |
| P2.3a | `M1_cross25_noise` | noise | 28.50% | 0.5282 | 0.5660 | 0.5547 | +25% 训练内部跨风格伪配对 (`+8.8pp`) |
| P2.3b | `M2_cross50_noise` | noise | 31.83% | **0.5326** | 0.5350 | 0.5688 | +50% 跨风格伪配对 (`+3.3pp`) |
| P2.3c | **`M3_ab_cross50_srcA0`** | **srcA0** | **38.00%** | **0.5270** | **0.5489** | **0.4458** | **源图→跨风格 Anchor-Bridge (`srcA0 +18pp`)** |
| P2.4 | **`O7_directapp_t40`** | **srcA0** | **38.50%** | 0.5251 | **0.5481** | **0.4481** | +795D 无分类外观统计残差；`t50` D-C `0.6380` |

---

## 五、V24 后续清晰路线建议

1. **主模型突破 `DINO-S ≥ 0.565 @ DINO-C ≥ 0.615`**：在全微批双分支 (`Dual`) 框架下，摆脱对 `T4` 锚点纹理的依赖，直接使用训练池高保真跨风格配对做 **源图→目标画作 Anchor-Bridge** + **感知 Patch-SWD** 联合训练（由模型自己完成从源内容直达高质量目标笔触的完整去糊，不再把 `T4` 当拐杖）。
2. **真任意风格 One-Shot (`PRISM-Ref-v24`)**：保留 V23 验证有效的 `16×16 RefAttn + 置信度门控 + 源图 Anchor-Bridge + O7 外观直通` 架构，把训练参考池从 WikiArt 5 类扩充至公开多风格艺术数据集（覆盖水墨、二次元、抽象、版画等 25+ 家族），直接打破 5 画派覆盖瓶颈。

---

## 六、资产与测试核验

- **代码清单**：`exp/v23/{__init__,p0b_dual_reval,p0c_oneshot_reval,p1_fast_anchor,p1_content_leak_diag,p1_midfreq_loss,p1_anchor_bridge,p1_s32_zero,p1_joint_c1_ab,p1_high_dinoc_frontier,p1_train_dinoc62,p1_dual_frontier_sprint,p1_dual_swd_train,build_cross_pairs_4090,p2_r0_r3,p2_attn_cross,p2_sg_bench,p2_direct_app}.py`
- **视觉审计存档**：`archives/v23_traj_diag/`（Step 0 破坏定位轨迹图）、`archives/v23_sg25_audit/`（SG-25 OOD 输出图）、`archives/v23_frontier_audit/`（高 DINO-C 前沿图）
- **单测与仓库规范**：`pytest tests/test_no_absolute_paths.py tests/test_markdown_links.py tests/v22/test_core.py` 全量通过。
