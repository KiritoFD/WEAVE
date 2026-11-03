# 05 方法主线

## 1 任务与约束

- 形式：`y = f(x, s)`，`x` 内容图、`s ∈ {5 个已知风格}`；推理时**没有参考图**（`docs/archive/618/summary.md#L9-L25`）。
- 与 reference-based 方法（StyleGallery/CSGO/SCSA/StyleShot）根本不同，不能直接借用其指标结论。
- 轻量：单卡 3060 12G，端到端训练分钟级；训练 4 epoch ≈1.4 分钟（`docs/archive/merged-20261103/713/SUBMISSION_HANDOFF_2026-07-15.md#L156-L162`）。

## 2 现有模型机制（2026-07 论文版）

```
x(4×64×64) --VAE--> latent
   │
   ├─ Haar DWT → (LL, LH, HL, HH) 各 4ch → 拼接 16ch
   ├─ time embedding + 4×ResBlock + cross-attn(style tokens)
   ├─ 4 个速度头：v_LL, v_LH, v_HL (+ 可选 v_HH)
   └─ 沿 t 积分 → iDWT → 逐步 AdaIN → y latent → VAE decode
```

**训练目标**（[flow.py#L203-L219](file:///g:/GitHub/Latent_Style/WEAVE/flow.py#L203-L219)、[#L253-L273](file:///g:/GitHub/Latent_Style/WEAVE/flow.py#L253-L273)）：

- 端点 `target`：若 `structure_aligned_target=True`，LL 用内容图的 LL（可混入比例 `ll_partial_alpha` 的目标 LL），LH/HL/HH 用目标风格图的小波系数，再 iDWT 回像素域 latent。
- 速度监督：`v = DWT(target - content)`，即每个子带学"从内容到目标"的端点速度，而不是随机噪声端点。
- 插值：`x_t = (1-t)·content + t·target (+ bridge noise)`；loss 为各子带加权 MSE（`w_ll/w_lh/w_hl/w_hh`）。
- 训练时对 target 做 AdaIN（`train_adain_scale`），推理时对每步输出做 AdaIN（`endpoint_adain_scale`）。

**推理**（[model.py#L1442](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1442)）：

- `num_steps` 步 Euler；可选 RK4 / Heun（[model.py#L1195](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1195)）。
- 论文配置：8 步、`endpoint_adain_scale=2.0`（训练 1.0）；2.5 会崩（`docs/archive/merged-20261103/713/SUBMISSION_HANDOFF_2026-07-15.md#L150-L156`）。
- 能耗约束，条件扩散式多样本采样未用于主结果（`docs/archive/merged-20261103/713/SUBMISSION_HANDOFF_2026-07-15.md#L160`）。

**条件路径**（这是 713 的核心结论）：

- 监督信号（target HF）很强，但把 target 图 HF 直接当**条件输入**很弱（`docs/archive/merged-20261103/713/SUBMISSION_HANDOFF_2026-07-15.md#L123-L136`）。
- 坐标无关的 per-orientation HF 码（oriented）能升 DINO-S 且不崩内容；raw spatial 注入会泄漏布局、DINO-C 崩到 0.40。
- 放大残差幅值只升 off-style，不改变修正方向。
- HH 头低能量、相位敏感、与感知风格梯度弱对齐，628 判 DEAD、713 也未翻案。

## 3 五个已定位瓶颈

| 瓶颈 | 证据 | 对应路线 |
|---|---|---|
| 训练目标是"平均笔触"（确定性 ODE 收敛到 E[X\|c]） | 616 `README#L9-L35` | A / B |
| 参考条件弱：target HF 作 condition 无效；Style-ID 查表对未见风格泛化差 | 713 handoff L123-L136；`Plan.md#L52` | C |
| 训练/推理 AdaIN 不一致（1.0 vs 2.0） | config 与 713 handoff | 07_ISSUES |
| 有效数据仅 5k；18k 无 latent | [01_DATA.md](01_DATA.md) | C / P0 |
| latent HF 的信息上限未知（VAE 重建是否抹掉纹理） | `Plan.md#L60-L64` | D |

## 4 下一代 A/B/C/D

| 路线 | 假设 | 机制 | 依赖 | 验收 |
|---|---|---|---|---|
| **A 输出级监督** | latent flow 目标与感知风格弱相关，解码后监督更直接 | VAE/TAESD 解码 → VGG16 特征 → patch SWD/Gram + 内容 + 自相似；只对 2-4 步 Euler 反传 | VGG16、解码器、≥5G 显存；**batch 只能 1**（`Plan.md#L33-L48`） | 600 DINO-S +0.03 且 DINO-C 掉 ≤0.03 |
| **B 内容锚定风格操作** | 参考图应提供"操作"而非像素搬运 | `iDWT(T_LL(LL_c,s), T_HF(HF_c,s))`，空间结构全程来自内容 | A/C/D 之后；需教师权重 | 待定（`Plan.md#L68-L72`） |
| **C 参考 Style Encoder** | Style-ID 记忆泛化差，DINOv2 patch 参考更可迁移 | DINOv2 patch tokens + InfoNCE（同风格不同内容）+ hybrid conditioner | dinov2-base 768d cache（已有）；**18k latent 缺** | 600 DINO-S +0.03，Identity 掉 ≤0.05 |
| **D 纹理模块** | 若 VAE 重建上限足够，latent HH 值得做 | 先跑重建上限探针；再开多尺度纹理 refiner（`texture_module.py`） | 探针结果 | 600 DINO-S +0.02 且 LPIPS ≤+0.02 |

依赖关系：

```
P0 诊断（VAE 重建上限 / 5k vs 18k / 参考图交换 / 三方对比）
   ├─ A：可独立先跑（不依赖 18k）
   ├─ C：依赖 DINO cache + 18k latent（阻塞）
   ├─ D：依赖 P0 重建上限结论；上限不够则直接砍掉 HH 子路线
   └─ B：排队，等 A/C/D 给出方向后再定教师
```

## 5 设计要点（避免重复踩坑）

1. **不再把"更大残差"当风格增强**：713 已证方向错误。
2. **不再用 raw spatial target HF**：内容崩是结构性的，不是调参能救。
3. **A 的梯度必须穿过解码器**：只对 2-4 步展开（`output_supervision.unroll_steps`），否则显存与时间都不可接受。
4. **C 的 InfoNCE 必须同风格不同内容**：否则退化成新的 Style-ID 记忆。
5. **任何新路线先过 600 跨风格主表**，150 同风格只作 Identity 补充，不得混算。

## 6 参数与规模

- 可训练参数：论文口径约 1.04M；`docs/79/main_table_v4_remote.csv` 写 903K（旧 v4 快照，**未核对**）。
- VAE 冻结：SD1.5 EMA，latent scale 0.18215（`docs/HANDOVER.md#L140`）。
- 风格条件：`StyleConditioner`（`style_id → 160d`）+ 可选 reference tokens（`reference_encoder.py#L126`）。
