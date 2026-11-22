# P0-0 审计裁决 — 三陷阱全部实锤, 原方案前提被否定 (2026-11-22)

> **执行**: `exp/v26/gpu4090/p0_host_audit.py` + `p0_common.py` (三红线设施)
> **数据**: `runs/v26/gpu4090/p0_host_audit.json` / `p0_dualgate_*.json` / `p0_teacher_fuse_*.json`
> **协议**: 96 cross 任务四锚对比 (V23_HiC_JPk 宿主, 免标签条件), 教师 600 配对熔断直测

---

## 一、结果

### 1.1 接口审计 (陷阱 2 摸底)

三宿主同为 **SLFUNet 21.95M**, 条件 199d = [time64 | **onehot5** | u_vivid128 | drops2] —
连续向量与离散 onehot **并存且都由 drop_style 联合丢弃**:

| 宿主 | ckpt | 原生锚 | 原生条件来源 |
|---|---|---|---|
| V20_D2 | runs/v20/slf_flow/slf_core_v2.pt | T4 (泄漏) | match_scene_submode: 5×8 bank 按源场景选 u |
| V23_HiC_JPk | runs/v23/joint_c1ab/slf_Joint_Pk_C1AB.pt | T4 (泄漏) | 同上 |
| V25_Tr_T1 | runs/v25/r2_tmatrix/slf_Tr_T1.pt | T4 (泄漏) | 同上 |

**泄漏审计**: T4 锚生成 `tid_used = tid_true` (t4.py L68 实锤);
`compute_prism1_anchor_750` 的 chord=bary[tid_true] 同样泄漏;
唯一合法通道 = PRISM-1 **oneshot 模式** (u_tgt = encode(fib_ref), routed_student.py:413)。

### 1.2 锚点解耦 (陷阱 1 实锤 — 悬崖 40pp)

96 cross 任务, 全部免标签条件 (u_vivid=encode(fib_ref), tid_pred=argmax cos(u_ref, bary),
色度平移/dsd 均由参考预测, 红线 2 断言通过):

| 锚 | Top-1 | DINO-C 对源 | 性质 |
|---|---:|---:|---|
| **A_t4 (泄漏上界)** | **60.42%** | 0.6525 | tid_true 生成的 T4 |
| B_p1_1shot (合法) | 20.83% | 0.7479 | PRISM-1 oneshot chord |
| C_z0perp (合法) | 20.83% | 0.7792 | 色度矩对齐 |
| D_src (合法) | 19.79% | 0.8035 | 源锚 SDEdit |

**悬崖 ~40pp ≫ 5pp 门控**: 三个合法锚全部塌回 20% 区; DINO-C 完全反向
(锚越弱内容越"好"——因为风格根本没画上去, 输出≈源图微调色)。

### 1.3 教师熔断 (陷阱 3 预警 — 红线 1 成功拦截一次假成功)

教师重造 (V23 宿主 + z_p1_tr 锚 + true tid core8 条件, 600 配对) 直测
**Top-1 = 17.33%, 单类最低 0%** → **RuntimeError 熔断停机**。
若无数师熔断, P0-1 将在这个 17% 教师上空转数小时 — 红线机制验证有效。

### 1.4 根因追溯 (T4 强锚的条件依赖)

T4 锚 = `AnisotropicS10Teacher.solve_batch(..., ti=tid_true)` 的 2 步各向异性 S10 蒸馏轨迹
(routed_student.py:177-186) — **强锚生成器本身以真目标画派 id 为条件**。

## 二、统一图像

**SLFUNet 家族 (V20/V23/V25) 的强风格能力 = 泄漏强锚 (tid 条件) + 正确 onehot 的联合产物,
两者不可拆分**: 免标签锚塌 40pp; 弱锚 (PRISM-1 1 步) 撑不起宿主 (教师熔断 17.33%);
条件 bank 的 u 虽是连续向量, 但其选择依赖 tid + 源场景, 且 onehot 通道与 u 联合训练。

**因此 "轻量 E_ref 对齐原生条件空间 + 冻结宿主" 的 P0 原方案前提被实测否定**:
冻结宿主在免标签条件下不具备 60% 能力可供继承, Cross600 上的任何"成功"都将是
5 类隐式分类器假象 (陷阱 3)。

## 三、修正路线 (三选项, 待用户决策)

| 选项 | 内容 | 可行性关键 | 成本 |
|---|---|---|---|
| **A. 强锚免标签化** | 把 AnisotropicS10Teacher 的 tid 条件替换为参考图统计 (LD4 per-style Score → oneshot Score, 色度 palette → 参考调色), 重生成免标签 T4 级强锚 → 宿主保持冻结 | aniso_teacher 内部 tid 依赖度探针 (下一步 30 min); 若 tid 只用于 palette/score 选择则改造小 | 锚重生成 ~1h + 宿主免标签复评 |
| **B. P0-2 开放域重训** | 78k WikiArt 风格污染→异块复原流 + 5 类教师流联合, 宿主解冻 | 接受重训成本; 教师流仍需强锚 (依赖选项 A) | 4-6h |
| **C. 混合** | 先 A (免标签强锚, 宿主冻结验证上限), 后 B (开放域解冻) | A 的探针先行 | A→B 串联 |

**无论哪条路线, 三条防呆红线 (p0_common.py) 继续强制执行**; 教师熔断已在真实场景完成首杀。

## 四、与 16% 天花板 (L7) 的合并结论

V24 家族 16% 天花板 + SLFUNet 家族锚点悬崖 = 同一根源的两次显影:
**该家族的跨画派能力从未脱离"真目标 id 条件"存在过**。闭集 60% 的全部数字
(T4 锚 60.4%、宿主 58.83%) 都在泄漏条件下取得; 免标签世界的真实起点是 ~20%。
任何"继承闭集能力"的方案必须先解决**强锚/强条件的免标签化**, 否则都是从 20% 从头爬坡。

---

## 五、P0 第 1 步执行结果: AnisotropicS10RefTeacher 熔断 FAIL → 止损转流 B (2026-11-22)

`AnisotropicS10RefTeacher` (三处 tid 依赖 → 免标签替换: 参考色度矩对齐 / vgg_ext 描述子距离梯度 / 信任域几何保留) 按红线 1 熔断:

| 指标 | 读数 | 门槛 | 判定 |
|---|---:|---:|---|
| Top-1 | **36.67%** | ≥45% | FAIL |
| per-style | Imp 76.7 / Min 60.0 / Roc 43.3 / **ER 0.0 / Uki 3.3** | 单类 ≥20% | FAIL |
| DINO-C | **0.7379** | ≥0.62 | PASS |

**免标签能力阶梯确立**: 家族裸起点 20% → 单图 VGG 描述子迭代教师 **36.7%** (+16.7pp) → 泄漏上界 60.4%。
失败模式 = **ER/Uki 双画派坍缩** (与 V15 "Imp 不平衡先验坍缩" 同构) — 命中用户止损判据
("纯 VGG-SWD 迭代教师推不动 ER/Rococo" — 单图 VGG 统计量的表达天花板)。

**按两步止损规则执行**: 跳过离线教师路线 → 直接**流 B (78k WikiArt 强风格污染→异块复原
Direct Bridge) 1500 upd 探针** (完全不依赖任何教师, 靠真实画作流形驱动笔触合成)。
