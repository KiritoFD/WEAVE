# V25 模型谱系图（DOC-01，2026-11-20）

> **任务**：[agent2.md §八](../19_pull_together/agent2.md) A09/DOC-01 交付物之一。范围：V19–V25 主线模型谱系 + V25 机制 A/B 学生判负档案。
> **数据口径**：全部读数 = WEAVE `test750`（Cross600 读数 + Same150 恒等控制）× `S10` 解码 × 进程隔离四裁判（[../../src/eval/measure.py](../../src/eval/measure.py)）；台账 [../performance/metrics.csv](../performance/metrics.csv)（505 行，其中 V25 臂 53 行）。
> **证据源**：[04_round1_results.md](../19_pull_together/04_round1_results.md)、[GPU4090_RESULTS.md](GPU4090_RESULTS.md)、[GPU3060_RESULTS.md](GPU3060_RESULTS.md)、[audits/](audits/a01_round1_ci.md)（a01/a02/a04/a08）、[../v22/V22_P1_RESULTS.md](../v22/V22_P1_RESULTS.md)、[../v23/V23_RESULTS.md](../v23/V23_RESULTS.md)、[../performance/MASTER_LEDGER.md](../performance/MASTER_LEDGER.md)。
> **状态语义**：`active` = 当前引擎/主配置在用；`frozen` = 归档保留、不再演进；`rejected` = 判负/禁用，仅作档案。

## 〇、谱系总图

```text
                    T4 锚 (exp/v22/t4.py, AnisotropicS10Teacher 2 步, T4_HASH 04007b4f0355)
                    B4 锚 (runs/v6/P1_B4_distill50, 218K C3UNet 前馈蒸馏; 与 T4 distinct-scale, ratio 1.21)
                      │（A02: 两锚血缘不同——训练侧教师缓存用 B4，评测主配置用 T4）
                      ▼
  V19 PRISM-1 ──单步锚分量──▶ V20 SLFUNet (22M 分段视界流) ──冻结权重──▶ V25 推理消融/收敛配置
  s2_prism1_v2               slf_core_v2.pt                            (Round 1/1b + Round 2 配方)
  1.50M, 17ms                D5/D6/D6H solver + 4 分支/半 NFE CFG       │
  Top-1 44.33–50.17%         + 密度门控                                 ├──▶ 机制 A 空间门控（判负档案）
                             Top-1 55.67% @D6                          ├──▶ 机制 B0/B1 蒸馏（判负-修正中）
                                       │                               └──▶ B05 引擎注册表（hybrid=默认）
                                       ├──▶ V22 tmatrix/Dual（T0–T5 训练矩阵, 21.95M 冠军档 76.83%）
                                       │        └──▶ V23 DualSWD/HiC（高内容前沿 + cs_j bug 修复）
                                       │                 └──▶ V24 One-Shot O 系（O3 参考条件 + label-free 教师）
                                       │                          └──▶ V25 机制 C1 教师链（进行中）
                                       └──▶（V22/V23 的 hybrid 线性混合谱系 = V24 M0/M1 复用臂 → V25 曲线 C）
```

---

## 一、逐节点档案

### N0 · PRISM-1（V19 `V19_S2_prism1_v2`）— active

| 项 | 值 | 出处 |
|---|---|---|
| 权重路径 | `runs/v19/students/s2_prism1_v2.pt` | [../../src/v25/config.py](../../src/v25/config.py) `P1_CKPT_RELPATH` |
| 参数量 / 延迟 | 1.50M 活跃潜参数；单步 17.0 ms/img（4090），~29 ms @3060 | [../performance/MASTER_LEDGER.md](../performance/MASTER_LEDGER.md) §1.1 |
| 结构 | 单步前馈（NFE=1）；锚 = latent chroma-shift α=0.20（λ=1.0 配方内部固定）；无 CFG 分支、无门控 | [audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md) `EVAL_PRISM1_L100` 行 |
| 强度旋钮 | λ ∈ [0.75, 1.6]（直接方向，越大编辑越强）；最有效区间 λ∈[0.75,1.0]（ΔS/\|ΔC\|=0.216） | [audits/a04_strength_report.md](audits/a04_strength_report.md) §二.1 |
| 判别读数 | λ=1.0：DINO-C 0.7440 / DINO-S 0.4430 / LPIPS 0.5057 / Top-1 44.33%；λ=1.3：0.7032/0.4504/0.5339/50.17%；λ=1.6：0.6573/0.4561/0.5633/52.83% | [04_round1_results.md](../19_pull_together/04_round1_results.md) 曲线 A |
| 定位 | "极速高内容基座"：风格在同水位全场弱于 D6/Hybrid（同水位 D-S 差 ≥0.04），不能覆盖中高风格区间；源锁定率 0.487（五臂最高，"改不动源"） | [04_round1_results.md](../19_pull_together/04_round1_results.md) §三 P0-1；[audits/a06_route_report.md](audits/a06_route_report.md) F3 |
| 状态 | **active**——PRISMCore `prism1` 引擎 + Hybrid 混合分量 + 机制 B 热启动底座 | [../../src/v25/config.py](../../src/v25/config.py) ENGINE_REGISTRY |

### N1 · V20 SLFUNet（`slf_core_v2.pt`）— active（V25 主配置基座）

| 项 | 值 | 出处 |
|---|---|---|
| 权重路径 | `runs/v20/slf_flow/slf_core_v2.pt` | [../../src/v25/config.py](../../src/v25/config.py) `V20_CKPT_RELPATH` |
| 参数量 | 22M（SLFUNet 分段视界流，仅 16×16 瓶颈层注入结构语义） | [GPU4090_RESULTS.md](GPU4090_RESULTS.md)（"22M SLFUNet"）；[04_round1_results.md](../19_pull_together/04_round1_results.md) §四 |
| 锚/CFG/Gate/NFE | T4 锚；CFG 四分支 `full`（40 NFE）或半 NFE 双分支 `cs_j`（20 NFE，V23 修复后立住）；Gate = original sigmoid((dsd−mid)·550) 密度门控；solver D5 τ∈[0.28,0.66] / D6 τ∈[0.40,0.74] / D6H τ∈[0.48,0.78] | [audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md)（solver hash D5 `07904c2bb16b` / D6 `e14eb16ca1e9` / D6H `f52948ae3451`） |
| 强度旋钮 | τ_min ∈ [0.40, 0.72]（反向，τ↑=编辑↓）；最有效区间 τ_min∈[0.66,0.70]（0.606）；死区 [0.50,0.58]；**内容上限 DINO-C≈0.7055 实测饱和** | [audits/a04_strength_report.md](audits/a04_strength_report.md) §二.2、§六 |
| 判别读数 | M0_v20_d6（D6/full）：0.6688/0.5107/0.5374/55.67%；B_d6_t066（τ-shift）：0.7007/0.5028/0.5213/55.00%；P02_d6h_csj（D6H/cs_j）：0.6782/0.5121/0.5350/55.00% | [04_round1_results.md](../19_pull_together/04_round1_results.md) 曲线 B / P0-2 |
| 状态 | **active**——V25 Round 1 消融与教师三元组的共同 ckpt；收敛主配置候选"单模型 D6"即此权重 | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) 共享前置 |

### N2 · V22 tmatrix/Dual — frozen

| 项 | 值 | 出处 |
|---|---|---|
| 权重路径 | `runs/v22/tmatrix/slf_T4Ssnap1500.pt`（合法最优冠军）及 `slf_{arm}.pt` 训练矩阵族 | [../../exp/v22/eval_tmatrix.py](../../exp/v22/eval_tmatrix.py) `:133`；[../../exp/v23/p1_anchor_bridge.py](../../exp/v23/p1_anchor_bridge.py) `:44` |
| 参数量 | 21.95M（冠军档） | [../v22/V22_P1_RESULTS.md](../v22/V22_P1_RESULTS.md) §一 |
| 结构 | 从 `slf_final` 热启动 1800 updates 的 T0–T5 同预算训练矩阵；固定常数门控 C0（fixed g=0.5）为默认；钟形曲线定位 + 三维推理拨盘（g/wc/τhi） | [../v22/V22_P1_RESULTS.md](../v22/V22_P1_RESULTS.md) §四 |
| 判别读数 | 冠军档 g=0.45,wc=1.0,τhi=0.66：Top-1 76.83% / DINO-S 0.5781 / DINO-C 0.4779 / LPIPS 0.6268；零内容代价档 g=0.50,wc=1.5：Top-1 69.00%（配对 CI[+5.83,+11.00] SIG） | [../v22/V22_P1_RESULTS.md](../v22/V22_P1_RESULTS.md) §一 |
| 定位 | "PRISM-Core-Simple" 定型阶段；V22 的 T4 缓存链路修复（哈希校验）沿用为全仓锚缓存契约 | [../v22/V22_P1_RESULTS.md](../v22/V22_P1_RESULTS.md) §二 |
| 子节点 Dual | 权重 `runs/v22/attn_flow/prism_dual_branch.pt`（[../../exp/v22/train_dual.py](../../exp/v22/train_dual.py) `:35`）；cs_j bug 修复后双分支 Top-1 34.67%→49.50%、DINO-S 0.5522、DINO-C 0.6096（`F3_dual_fixed_orig`，20 NFE） | [../v23/V23_RESULTS.md](../v23/V23_RESULTS.md) §一/§二 |
| 状态 | **frozen**——V25 全部推理消融走 `slf_core_v2.pt` + T4 锚主系，tmatrix/Dual 谱系未进入 V25 消融（高风格档 DINO-C 0.478–0.532 偏离内容前沿的旧教训） | [../v22/V22_P1_RESULTS.md](../v22/V22_P1_RESULTS.md) §一"诚实边界" |

### N3 · V23 DualSWD/HiC — frozen（DualSWD 臂为备选单模型第二候选）

| 项 | 值 | 出处 |
|---|---|---|
| 权重路径 | `runs/v23/dual_swd/slf_DualSWD_pure.pt` / `slf_DualSWD_both.pt`（`train_dual_swd_arm` 产物） | [../../exp/v23/p1_dual_swd_train.py](../../exp/v23/p1_dual_swd_train.py) `:49,203-204` |
| 结构贡献 | **修复 cs_j 双分支 `v_0c` 误清零 layout_scale 的 bug**（Top-1 42.50→50.50%）；高 DINO-C 前沿（C1 ConvNeXt 中尺度损失、AB1 Anchor-Bridge）；跨风格伪配对监督与 srcA0 桥是 One-Shot 打通的根本（12.2→38.5%） | [../v23/V23_RESULTS.md](../v23/V23_RESULTS.md) §一 |
| 判别读数 | `HiC_JPk_csj65_t4`：Top-1 58.83% / DINO-C 0.6299 / DINO-S 0.5330（20 NFE）；`DualSWD_pure_hiT_ws52`：51.17%/0.6164/0.5467；`F3_dual_fixed_orig`：49.50%/0.6096/0.5522 | [../v23/V23_RESULTS.md](../v23/V23_RESULTS.md) §二 |
| V25 交叉 | Round 1 `B_dswd_t058`（DualSWD 权重 + ws5.2，独立 lineage）：0.6770/0.5156/0.5351/51.83%——风格更强（同水位插值 +0.007）但 Top-1 −3.8pp，暂列单模型第二候选 | [04_round1_results.md](../19_pull_together/04_round1_results.md) 曲线 B；[audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md) 约束 6（隔离条款） |
| 状态 | **frozen**（DualSWD 权重仍可作对照臂；V23 标签教师包 `runs/v23/train_cross_pairs_1000.pt` 仍是 V24 O 系臂的训练包） | [audits/a03_teacher_quality.md](audits/a03_teacher_quality.md) §三 血缘 |

### N4 · V24 One-Shot O 系 — active（PRISMRef 默认引擎；教师链演进中）

| 项 | 值 | 出处 |
|---|---|---|
| 权重路径 | `runs/v24/oneshot_val/O3_rgb_cnx_gated.pt`（O3 主臂）；O3_label_free = 同配方换 V24 label-free 教师包 | [../../src/v25/config.py](../../src/v25/config.py) `O3_CKPT_RELPATH`；[audits/a03_teacher_quality.md](audits/a03_teacher_quality.md) §三 |
| 参数/结构 | SLFUNetV24OneShot：Q=灰度 DINO 16×16 布局、V=RGB ConvNeXt v16 局部、u_glob=795D fiber 全局门三通路并联，置信度门 R=g·R+(1−g)·u_g；10 步 RK2 + wj 引导；`style_id` 形参零使用（`oh=torch.zeros`，结构性免标签） | [audits/a08_science_review.md](audits/a08_science_review.md) §六；[audits/a05_feature_geometry.md](audits/a05_feature_geometry.md) §一 |
| 强度旋钮 | τ0 ∈ [0.40, 0.60]（反向）；单调无异常；**wj 混杂**（t40 wj=2.3 / t60 wj=1.9，非纯 τ0 单变量）；默认冻结 wj=2.2（t45 canonical） | [GPU3060_RESULTS.md](GPU3060_RESULTS.md) ①；[audits/a04_strength_report.md](audits/a04_strength_report.md) §六 rejected |
| 判别读数 | O3 t45：0.5845/0.5245/0.4213/35.33%；O3_label_free t45：0.6769/0.4666/11.33%（swap 0.1311 vs O3 0.442）；O3_self_only t45：0.7233/0.4734/12.33% | [GPU3060_RESULTS.md](GPU3060_RESULTS.md) ①②③ |
| 判负支系 | V24 M3/M4/M5（avg_pool 伪 z0 锚）、M6（z_p1 充当 D6 锚的双重伪造链）——**禁止**进入任何 V25+ 教师/锚缓存 | [audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md) 表 (c) |
| 状态 | **active**——PRISMRef `o3_ref` 默认引擎（fallback 保底：原 O3 权重不删）；C1 教师重生成链（reference_teacher_v2）进行中 | [GPU3060_RESULTS.md](GPU3060_RESULTS.md) 队列下一步 |

### N5 · V25 收敛配置（当前最简主配置候选，冻结判定待 CI 收尾）

| 项 | 值 | 出处 |
|---|---|---|
| 模型 | `slf_core_v2.pt`（单模型，无 Hybrid）为候选；**Hybrid 保持 `DEFAULT_CORE_ENGINE="hybrid"`**（B05 骨架集中切换点） | [04_round1_results.md](../19_pull_together/04_round1_results.md) §四；[../../src/v25/config.py](../../src/v25/config.py) `DEFAULT_CORE_ENGINE` |
| CFG/锚/Gate/强度 | cs_j 双分支 20 NFE、ws_fig=4.2；锚 T4（默认）/ PRISM-1（快速档，~省 225ms，判读 INCONCLUSIVE）；Gate：original 密度门控（A01 #2 FAIL 后保留）；strength = τ_min ∈ [0.40,0.66]，tau_hi=0.78 | [04_round1_results.md](../19_pull_together/04_round1_results.md) §四 + 顶部 CI 修正块 |
| Round 2 训练配方 | `Tr_T1`（drop_c 30%）@D6h **PASS**（Top-1 +1.00pp CI 不含零）；种子复验方向一致（seed2 d5 56.67%）；T1–T3 权重 `runs/v25/r2_tmatrix/` | [audits/a01_round1_ci.md](audits/a01_round1_ci.md) 组 6；[GPU4090_RESULTS.md](GPU4090_RESULTS.md) Round 2 |
| 教师三元组（机制 A/B 锁定） | `ckpt=slf_core_v2.pt + anchor=B4(canon) + solver=D6H + branches=cs_j + gate=original`；缓存 `runs/v25/cache/train_latents_4400.pt`（577MB，含 teacher_spec） | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) 共享前置；[audits/a08_science_review.md](audits/a08_science_review.md) §二（锚不一致 FAIL） |
| 工程骨架 | [../../src/v25/config.py](../../src/v25/config.py)（B05 双接口规约 + 强度映射只消费 a04 JSON）、[../../src/v25/artifact_registry.py](../../src/v25/artifact_registry.py)（B03 血缘）、[../../src/v25/result_writer.py](../../src/v25/result_writer.py)（B06 状态机） | 各文件 docstring |

### N6 · 机制 A/B 学生（判负档案）— rejected

| 臂 | 权重/位置 | 结构 | 判定 | 出处 |
|---|---|---|---|---|
| 机制 A `A0_global` | `runs/v25/mechanism_a/`（门控 `gate_A*.pt`，EMA） | γ=0.55 全局混合基线（B4 lineage，D-C 0.7361） | 基线 | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) 机制 A 表 |
| 机制 A `A1_heuristic` k50/k100 | 同上 | 启发式空间门 κ(1−ē)，零训练 | PASS≈0（+0.0017/−0.0004 噪声）→ 无增量 | [audits/a08_science_review.md](audits/a08_science_review.md) 附录 A |
| 机制 A `A2_learned` / `A2_em` | 同上 | 1000 updates 学习门控（em=能量锁定 0.55） | **FAIL**（ΔS −0.0111/−0.0083，Minimalism 守卫 −9.2/−12.5pp）；名义 A2 读数 ~85–90% 为 λ_c 标定残差（Δ=+0.037）导致的强度漂移伪影 | [audits/a08_science_review.md](audits/a08_science_review.md) §一 |
| 机制 A `A3_smooth` | 同上 | 平滑约束学习门 | PASS 但 ΔC CI [−0.0004,+0.0002] ≡ 退化为常数门 | [audits/a08_science_review.md](audits/a08_science_review.md) 附录 A |
| 机制 B0 `student_B0.pt` | `runs/v25/mechanism_b/` | PRISM-1 热启动 + 显式 γ 残差分支 361K，2000 updates | **判负-修正中**：γ=0 精确回退成立（residual=0.0 逐位）；γ>0 主证被 A08 判为伪迹/混杂（见 [ABLATION_LEDGER.md](ABLATION_LEDGER.md)） | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) 机制 B0；[audits/a08_science_review.md](audits/a08_science_review.md) §二 |
| 机制 B1 | 未跑（本窗口不跑，重试协议已记录） | 6000 updates 更宽学生 | pending（B1 前置 = R2 bypass 修复重评） | [GPU4090_RESULTS.md](GPU4090_RESULTS.md) |

> 机制 A/B/C 代码：[../../exp/v25/mechanism_a.py](../../exp/v25/mechanism_a.py) / [../../exp/v25/mechanism_b.py](../../exp/v25/mechanism_b.py) / [../../exp/v25/mechanism_c.py](../../exp/v25/mechanism_c.py)；A08 补救 R1/R2 的零训练重评脚本已就绪：[../../exp/v25/r12_anchor_reeval.py](../../exp/v25/r12_anchor_reeval.py)（B4 锚 ctx750 重评 7 臂 + B0 bypass 修复 γ 扫描）。

---

## 二、锚点谱系（跨节点共享资产）

| 锚 | 身份 | 生成路线 | 状态 |
|---|---|---|---|
| T4 锚 | `T4_HASH = 04007b4f0355`（config hash，漂移测试守卫） | AnisotropicS10Teacher 2 步各向异性 S10-VGG 优化输出（[../../exp/v22/t4.py](../../exp/v22/t4.py)） | active——评测侧默认锚（P1-3 INCONCLUSIVE 后维持） |
| B4 锚 | ckpt sha `2716436fe0ee…`（`runs/v6/P1_B4_distill50/P1_B4_distill50.pt`） | 218K C3UNet 前馈蒸馏 + canonicalizer(c1_gentle)；**"T4 等价"系书面断言，实测 distinct-scale（divergence_ratio=1.21 > 0.60，task_l2_p50=28.0 ≈ 锚-源距离 28.7）** | active（训练侧教师）——与 T4 禁止互相归因 |
| PRISM-1 锚 | s2_prism1_v2 λ=1.0 输出 | 免 T4 快速档（省 ~225 ms） | 快速档定位（对齐后仍略逊 −0.005~−0.007，`P13b_ancp1_t044`） |
| z_src 锚 | 源图干净 latent | — | **rejected**（风格坍缩：Top-1 7.17%、LPIPS 0.2245 ≈ 近恒等） |
| avg_pool 伪锚 | V24 M3/M4/M5 legacy | `avg_pool(3×3 replicate)` | **rejected + 禁用**（a02 表 (c)） |

锚语义可移动 Top-1 ~48pp（P1-3：T4 55.00% vs z_src 7.17%）——任何"删组件"式归因必须走 A01 配对 CI，不得当同配置读数直减（[audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md) §二 (d)）。

## 三、状态汇总

| 节点 | 权重 | 状态 | 一句话 |
|---|---|---|---|
| PRISM-1 | `runs/v19/students/s2_prism1_v2.pt` | active | 极速高内容基座 + Hybrid 分量 + B 热启动 |
| V20 SLFUNet | `runs/v20/slf_flow/slf_core_v2.pt` | active | V25 主配置基座与教师 ckpt |
| V22 tmatrix | `runs/v22/tmatrix/slf_T4Ssnap1500.pt` | frozen | 76.83% 冠军档归档；T4 缓存契约沿用 |
| V23 DualSWD/HiC | `runs/v23/dual_swd/slf_DualSWD_*.pt` | frozen | cs_j 修复来源；备选单模型第二候选 |
| V24 O3 | `runs/v24/oneshot_val/O3_rgb_cnx_gated.pt` | active | PRISMRef 默认引擎；label_free 保守（教师链修复中） |
| V25 收敛配置 | B05 `ENGINE_REGISTRY` | active（候选冻结待 CI） | hybrid=默认核心引擎；单模型 D6 为压缩候选 |
| 机制 A 门控 | `runs/v25/mechanism_a/` | rejected | 空间重分配无增量（A1/A3/em 联合证据） |
| 机制 B0 学生 | `runs/v25/mechanism_b/student_B0.pt` | rejected（修正中） | γ=0 回退成立；主证待 R12 重评 |

> 相关文档：结论账本 [ABLATION_LEDGER.md](ABLATION_LEDGER.md)；复现手册 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
