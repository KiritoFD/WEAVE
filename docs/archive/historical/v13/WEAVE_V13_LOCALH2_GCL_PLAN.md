# WEAVE V13：LocalH2-GCL —— 局部笔触生成 + 真正参与梯度的内容 Query 保持

**预注册 · 2026-11-14 · 执行机：3060（12GB）· 训练预算：4000 真实 optimizer updates 起步（effective batch 32–64）**
**协议：全部正式效果臂 = 完整 test750（cross600+same150），禁用 Dev200/Val200/Pilot80 筛选。**

> 底座 = V11b-H2（已证实的中频生成收益：DINO-S .4401→.4482，DINO-C .7718→.7770）。
> 本阶段三件事：① 修 V11b 的 8 处 bug（原 checkpoint 归档不动）；② 新增局部风格
> Attention（RSL 式）+ 真正可微的 GCL（Q + R），与局部风格目标联合优化；
> ③ A0–A7 消融隔离"局部性 / GCL / 风格特异性"三个问题。
> **不训练大型外部模型；无 GAN、无 RGB 重建损失、无跨画派伪 GT。**

## 0. 与前序阶段的关系

- V10.2：手工 latent 统计无风格信号 → 表示须换（V11 P0 已定 DINO 域）。
- V11 密度 Score：无画派特异方向 → **本阶段不再用密度 Score 做方向**；
  局部风格方向来自 **RSL 式 Attention 目标 + 学习式 E_s**（判别式对比），
  GCL 是**内容保护项**而非方向来源。
- V12-GCL：LD4 末端硬闸门不 binding（风格方向与内容 Query 梯度近正交）→
  V13 把 GCL 移进**生成阶段的联合优化**，让它真正接受检验。

## 1. V11b bug 清单与修复位置（P0；原 checkpoint/代码归档不动）

| # | bug | 现状（已核实） | 修复 |
|---|---|---|---|
| 1 | 采样随机性 | `sample_band()` 用 `randn_like()`，无 generator | 新 sample：逐任务固定 generator（seed=任务索引哈希），全实验共享 |
| 2 | 风格打乱连带 sigma | `sigmas[str(style_id)][band]` 随 sid 变 | 错误风格实验**只换条件不换噪声尺度**（sigma 用正确画派或全局固定） |
| 3 | sigma 计数器 | `estimate_sigmas()` 的 `sums[sid]["n"]` 被 H1/H2 双重累计 → **sigma 系统性 ÷√2** | 按带独立计数；**旧模型用原 sigma 保留**，新训练用修复版 |
| 4 | 训练步数语义 | `step` 计 microbatch | 新训练以 **optimizer updates** 为准；checkpoint 核实：旧模型 accum=1 → 8000 step = 8000 真实 updates（无欠训练） |
| 5 | OneCycleLR 跨度 | total_steps=updates（microbatch 语义） | total_steps = 真实 optimizer updates |
| 6 | edit_bands @no_grad | 梯度精修不可能 | 新实现**可微** edit（GCL 阶段用） |
| 7 | resume 空挂 | args.resume 解析但循环无恢复逻辑 | 恢复 model/opt/sched/step/RNG |
| 8 | load_flow 默认 width=40 | 实际 ckpt width=56 | 从 ckpt metadata 读 width |

## 2. 迁移到 3060

3060 已有：数据集（wikiart_new_512 + distinct5_512/train ✓）、S10 decoder ✓、
DINOv2-small HF 缓存 ✓、torch 2.11 栈 ✓。需上传（全部小件）：

| 资产 | 本地 → 远程 | 大小 |
|---|---|---|
| `runs/v11b/flow_v1_w56/ckpt_{25,50,75,100}.pt` | → `I:/Latent_Style/WEAVE/v13/ckpt/` | 4×5.7MB |
| `docs/v11b/P0_PROFILE.json` | → `I:/Latent_Style/WEAVE/v13/` | <1KB |
| **LD4 latent batches**（v9 `runs/v9_local_4070/{dev,test}/LD4/batch_*.pt`，z_final+ids） | → `I:/Latent_Style/WEAVE/v13/ld4_lat/` | ~48MB |
| LD4 基础图 | **不上传**——3060 用 S10 从 LD4 latents 重解码生成 | 0 |
| distinct5_512/train | **已在远程** `I:/Latent_Style/WEAVE/datasets/distinct5_512/train` | 0 |

远程工作目录：`I:/Latent_Style/WEAVE/v13/`（独立于旧工作树，勿覆盖）。

## 3. 新模块：Local H2 Attention（RSL 式）

两级 Haar：`x ↔ (LL2 3×128×128, H2 9×128×128, H1 9×256×256)`。
H2 是主要编辑空间（V11b 收益来自 H2 而非 H1）。

| 组件 | 初始方案 |
|---|---|
| 内容 Query Q_c | LL2 + 边缘方向 + 局部结构编码 |
| 风格 Key/Value | 真实画作 H2 patch 经冻结 E_s 编码的局部特征 |
| 每画派原型 M_s | 32–64 个 |
| Attention token | H2 下采样到 32×32 |
| 特征宽度 | 64 |
| 注入方式 | `f' = f + W_o·Up(A_s)`，W_o 零初始化（初始严格等于 V11b-H2） |
| 训练 | 冻结 H2 Flow，只训 Adapter |

**禁止直接回归 donor 的 H2 数值**（v10.1 判负）；Value 是可学习的风格隐特征。
Memory 必须跨画作：训练/推理不得检索目标画作自身。

**前置 Sim 门槛**：先训轻量局部笔触编码器 E_s（真实画作 H2 patch、跨画作对比、
内容分层采样防题材捷径），冻结后构建 Memory，并验证
`Sim(E_s(H_real), M_correct) > Sim(E_s(H_real), M_wrong)` 在画作级 holdout 成立。
不成立 → 不训 Adapter（判停）。

## 4. 真正的 GCL（可微、联合优化）

- **GCL-Q**：`L_GCL-Q = Σ_l w_l‖Q_c^l(y) − sg[Q_c^l(x0)]‖₁`——Q_c 冻结、多尺度、
  **必须依赖被修改的 H2**（仅 LL2 会让损失恒定）；
  初始 Q_c 输入 = 由 LL2+H2 重建的 LL1 及局部梯度特征。
- **GCL-R**：局部窗口关系 `R_ij(x)=softmax(Q_i·Q_j/√d), j∈N(i)`；
  `L_GCL-R = Σ_i‖R_i(y)−R_i(x0)‖₁`——可微连续正则（不再用 V12 的硬闸门），
  并以 V12 旧 selfsim 接口作对照。
- 联合：`L = L_RSL + λ_Q·L_GCL-Q + λ_R·L_GCL-R`；训练阶段保留真实画作 H2 的
  Flow Matching 监督；推理时仅对 H2 做 1–2 步梯度精修（不反传 WEAVE 主干）。
- λ_Q/λ_R 在推理期做少量预注册扫描；**记录每次编辑的 RSL/GCL 梯度范数与局部
  关系变化**（避免 V12"约束不发挥作用"重演——本轮 GCL 在优化回路里，必须有效）。

## 5. 消融矩阵 A0–A7（同一训练权重 + 同一固定噪声，只改推理配置）

| 臂 | H2 Flow | Memory | RSL | GCL | 目的 |
|---|---|---|---|---|---|
| A0 | 无（LD4） | — | — | — | LD4 基准 |
| A1 | 原版 | — | — | — | V11b H2 基准 |
| A2 | 修正版 | — | — | — | bug 修复收益 |
| A3 | 修正版 | ✓ | ✓ | — | 纯局部风格驱动 |
| A4 | 同 A3 | ✓ | ✓ | Q | GCL-Q 贡献 |
| A5 | 同 A3 | ✓ | ✓ | Q+R | 完整 GCL |
| A6 | 同 A3 | 全局共享 Memory | ✓ | Q+R | 区域性是否重要 |
| A7 | 同 A3 | 错误画派 Memory | ✓ | Q+R | 目标条件特异性 |

A0 = LD4（复用 merged-750）；A1 = V11b 已有评测（复用）；A2 起为新权重。

## 6. 成功判据（merged-750 口径，源图聚类配对 bootstrap）

| 指标 | V11b H2 现值 | V13 目标 |
|---|---:|---|
| DINO-S | 0.4482 | ≥0.451 |
| DINO-C | 0.7770 | ≥0.772 |
| LPIPS | 0.4927 | ≤0.4927（争取逼近 LD4 的 0.4872）|
| Top-1 | 0.3333 | ≥0.3517（LD4）|
| Affinity | 0.5666 | ≥0.570 |
| 风格特异性 | 弱 | A3/A5 正确条件稳定优于 A7 错误条件 |

必报：逐画派分解、G/Margin、局部结构变化。DINO-S 上升不等于学到笔触。

## 7. 3060 实施顺序与判停

| 阶段 | 任务 | 放行条件 |
|---|---|---|
| P0 | 迁移 + 8 bug 修复 + 旧模型复现（新固定噪声，不要求逐位一致） | 750 输入完整、加载/推理通过 |
| P1 | E_s 编码器 + 五画派 Memory | 画作级 holdout Sim 门槛通过 |
| P2 | 冻结 H2 Flow，训 Local Attention Adapter | 不同 Style-ID 产生可测的不同局部更新 |
| P3 | 可微 GCL-Q/GCL-R | 梯度与有限差分一致、对 H2 更新实际生效 |
| P4 | A0–A7 × 750 | 四个问题分别回答 |
| P5 | 仅 P4 有收益时联合微调 | 超越冻结 Adapter |

判停：P1 Sim 门槛不过 → 不训 Adapter；P2 无 Style-ID 差异 → 检查目标函数；
P4 无收益 → 本路线关闭，记录全部负结果。

## 8. 代码结构

```
exp/v13/
  __init__.py
  sample_fixed.py      # 修复版采样/编辑（generator + sigma 修正 + 可微）
  train_fixed.py       # 修复版训练（真实 updates / OneCycle / resume / 按带 sigma）
  bands.py             # 复用 exp.v11b.bands（import）
  dataset_real.py      # 复用 exp.v11b.dataset_real
  model.py             # 复用 exp.v11b.model + 新 Adapter
  stroke_encoder.py    # E_s 对比学习
  memory.py            # M_s 构建与检索
  adapter.py           # Local H2 Attention（零初始化残差）
  gcl.py               # GCL-Q / GCL-R（可微）
  run_750.py           # A0–A7 推理
  run_v13_remote.py    # 3060 入口
runs/v13/…            # 产物（untracked）
docs/v13/V13_RESULTS.md
```
