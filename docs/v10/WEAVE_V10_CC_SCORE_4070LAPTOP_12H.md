# WEAVE V10：Content-Conditioned Local Style Score from Real Artistic Distributions

**预注册计划 · 2026-11-14 · RTX 4070 Laptop（~6GB 可用）· 12h 预算**

> 本文档是 V10 的预注册契约：先写死假设、协议、验收门槛与判停条件，再动手实现。
> 进展见 [V10_PROGRESS.md](V10_PROGRESS.md)。协议级前提沿用根 `AGENTS.md`：解码一律 S10、
> 评测一律 `eval_protocol/tasks_750.json` + `src/eval/measure.py` referee。

---

## 0. 假设与定位（一句话）

**从真实画作 patch 分布学习"内容条件化的局部风格密度"，用其闭式 score
∇log p(v|q,t) 替代 R1 的分类器梯度作为推理期风格方向。**

与既有路线的区别（为什么不是换皮）：

| 不是 | 原因 |
|---|---|
| R1 / LD1-LD2 | 不以分类边界为目标，而是趋近真实分布的密度 |
| LD3（v9 局部能量） | LD3 用固定 kmeans 原型的 softmin 能量（无条件、单模态倾向）；V10 是多样本条件 KDE（多模态 + 内容条件） |
| V6 全局矩 / Patch-SWD | 不只匹配边际统计，匹配条件分布 |
| V8 C2 | 不用固定空间区域原型（匹配由内容结构决定，与坐标无关） |
| StyleGallery / SMS | 不依赖大型扩散 attention 或每画派 LoRA |

**三个此前缺失的性质**：多模态（画派=模式集合，非原型均值）、条件性（q 决定匹配对象）、
可验证梯度（实验 A 直接检验方向，而不是只看密度）。

**核心风险（预注册写明）**：密度高 ≠ 人眼更好；画派与题材相关时模型可能学内容捷径
（实验 B 专审颜色捷径）；S10 latent 是否保留笔触信息未经证明（实验 A 的退化恢复梯级同时
是探针）。

---

## 1. 资源盘点（已核实）

| 资源 | 路径 | 规模/说明 |
|---|---|---|
| 真实画作 latent 池 | `weights_local/sd_pool_orig/0{style}_{name}.pt` | **15,539 幅**逐画作 SD-VAE 后验 mean latent (4,64,64) fp16 + `ids`/`files`（wikiart_new_512 原图）。ER 1474 / Imp 8989 / Min 1084 / Roc 2519 / Ukiyo 1479 |
| 任务缓存 | `runs/v9_local_4070/cache/n2c_z.pt` | 750 任务逐任务 `z_src/z0/z_n2`（N2C 优化起点；LD 系列同款） |
| v9 style bank | `runs/v9_local_4070/bank/style_bank.pt` | LD3 用的 kmeans 原型（作对照基线） |
| R1 critic / SPDHead | `src.infer.load_critic` + `runs/v8/train/N2C/step_04000.pt` | 对照梯度 + rank-0 预条件 |
| S10 decoder | `deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt` | `src.utils.load_s10_decoder/decode_s10` |
| referee | `src/eval/measure.py` | DINO-S/C、CLIP-S、Affinity、CSD、LPIPS（独立进程） |
| 750 任务 | `eval_protocol/tasks_750.json` | Cross600+Same150；**150 源图与画作池 stem 零重叠（2026-11-14 已验证）** |

v9 拆分沿用：`md5(source_stem)%5==0` → dev(170 任务)，否则 test(580)。

---

## 2. Patch 特征：内容 q、风格 f、内容可预测成分 hψ、残差 v

每画作 latent (4,64,64) → 64 个不重叠 8×8 patch。全部特征 autograd 可导（软统计，无 hard binning）。

**内容结构 q（D=84）**：`LL`（2×2 平均得 4ch×4×4=64）+ LL 梯度 dh/dv/dd RMS（12）+ LL per-ch mean（4）+ patch 边缘能量（4）。
**风格特征 f（D=34）**：
- 颜色：per-ch mean(4)、per-ch std(4)、通道两两相关(6)；
- 纹理：latent 域 dh/dv/dd RMS per-ch(12)、去框模糊残差 std per-ch(4)、Laplacian per-ch(4)。

**跨画派共享内容预测器 hψ**：MLP `q(84) → 128 → f(34)`，在**全部五画派训练 split 的 patch** 上以
重建 L2 训练（不做画派条件——故意的：让 h 只学内容可解释的共性）。**残差 v = f − hψ(q)**，
v 与 q 各自标准化（训练 split 统计）。

**必须的消融**：V10-B vs V10-C 对比即 hψ 的存在性检验；若 hψ 把画派信息一并抹掉，
C 的恢复方向/混淆矩阵应显著劣于 B——那本身就是结论。

## 3. 数据划分（防 patch 泄漏）

- **按画作划分**（不是按 patch）：每画派以 `md5(painting_id)%10<9` → train(90%)，其余 → holdout(10%)。
  train 建 bank / 训练 hψ / 训练 DSM；holdout 只用于实验 A/B/C 评测。
- 完整性检查（P0 自动执行并记录）：train/holdout patch 计数表；holdout 画作 id 与 train 无交集；
  150 任务源图与全池零重叠（已验）。
- 已知不平衡：Impressionism 8989 vs Minimalism 1084 —— KDE 评测按 per-style 各自库内比较，
  混淆矩阵实验对每库子采样到相同 patch 数上限（每画派 ≤200k patch）。

## 4. 闭式条件 KDE Score（先验证，不训练）

对目标画派 t 的 bank `B_t = {(q_j, v_j)}`：

```
p_σ(v|q,t) ∝ Σ_j exp( -||v−v_j||²/2σ² − ||q−q_j||²/2τ² )
s_t(v|q)   = ∇_v log p = ( Σ_j w̃_j v_j − v ) / σ²        （w̃ 为归一化条件核权重）
```

样本级方向 `g_s = ∇_z Σ_r s_t(v_r(z)|q_r(z))`（经特征可导，autograd 直达 z）。

**变体（即核心消融 V10-A/B/C）**：

| 变体 | bank 内容 | 回答的问题 |
|---|---|---|
| V10-A | (v, v)，q 不参与权重 | 多模态分布本身是否优于全局统计/LD3 原型？ |
| V10-B | (q, f) 条件 + f 原值 | 内容条件匹配是否带来正确方向？ |
| V10-C | (q, v) 条件 + 残差 v=f−hψ(q) | 内容去偏是否减少泄漏、提升方向质量？ |

超参固定为预注册：σ=1.0（v 标准化空间）、τ=2.0（q 标准化空间）、KNN 截断 top-4096
（其余权重置零，等价截断核）、bank 每画派子采样 ≤200k patch。**不调参挑选结果**；
敏感性检查（σ∈{0.7,1.0,1.5}）只作附录报告。

## 5. 实验协议（P2 的三个实验，全部自动）

### 实验 A：真实画作自动去风格化—恢复（方向一致性）

holdout 画作每画派 40 幅 → 退化梯级 `z^- = D_α(z_real)`：
① 纹理衰减（latent 域去 3×3 框模糊残差 ×α）② 方向平滑（x/y 各向异性模糊）
③ Haar 高频削弱（LH/HL/HH ×α）④ 弱颜色规范化（向源池全局 mean/std 收缩 β）。
α∈{0.3,0.5,0.7} 固定种子组合成 3 档 + 单项 4 种，共 7 个退化算子。

指标：`A_grad = cos( g_t(z^-), z_real − z^- )`（方向余弦，per-degradation 平均 + 逐画派表）。
**对照**（同协议、同一批 z^-）：R1 margin 梯度（v9 DirectionComputer 同款）、
LD3 原型局部能量梯度、V6 式全局矩匹配梯度（per-ch mean/std 对画派池统计）。

**门槛（P2→P3 的判据）**：V10 最优变体的 A_grad 平均值 > R1 且 > LD3，
且在 ≥4/5 画派上同向；若不满足 → **判停**（见 §8）。

### 实验 B：颜色捷径审计

bank 按特征族拆分：`color-only`（8 维）、`texture-only`（26 维）、`joint`(34)、`joint+residual`(34)。
四变体在实验 A 同一评测集上比较 A_grad；另报每变体更新量在 color/texture 子空间的能量占比。
**判读**：若 texture-only ≤ color-only 的方向一致性，则该路线与 Hist2Style 同病（颜色账），预注册结论为"未解决核心问题"。

### 实验 C：跨画作泛化与混淆

- holdout patch 的 5-way 混淆矩阵：argmax_t log p_t(v|q)（V10-B/C）与 R1 分类对照；
- 退化梯级上的密度单调性：log p 随 α 增大应单调下降（Spearman ≥0.9 记合格）；
- 同内容跨画派 probe：对每对画派 (t≠t') 取 t' holdout 中与 t bank q 最近邻的 patch，
  检验条件化是否把"内容像 t 的 t' patch"仍判给 t'（条件性起作用的直接证据）。

## 6. P3：接入 WEAVE 受约束推理优化（V10-E）

> **口径修正（2026-11-14，用户决定）**：任务级评测与决策一律用**完整 test750**
> （对比读数取 cross600；same150 恒等任务饱和、无区分度）。**不使用 dev 预览集**——
> dev 只有 136 个 cross 任务，Top-1 粒度 0.74pp，足以淹没真实效应。
> LD 系基线的 750 口径由 v9 的 dev+test eval 行合并重算（task 互补无重叠）。
> 完整 raw 图集现可通过 `datasets/wikiart_raw` 符号链接访问（J:\WEAVE_DATA，
> 数字命名 + `wikiart_new_512_images.tgz`；与池子 `Style__artist_painting` 命名的
> 映射 manifest 待查，仅影响 P0 探针与定性出图，不影响 latent 空间实验）。

冻结 B4/N2C（复用 `cache/n2c_z.pt` 起点），对**全部 750 任务**做 patch 级 score 梯度上升：

```
g_safe = g_s − max(0, ⟨g_s, ∇L_c⟩)/ (||∇L_c||²+ε) · ∇L_c     （一阶内容投影）
z_{k+1} = Proj_C( z_k + η_k · normalize(g_safe) )
```

- `L_c = λ_low·C_LL(z,z0) + λ_struct·L_selfsim(z,z0)`（复用 v9 `content_loss`/content risk 思路；
  λ_low=1, λ_struct=0.5 预注册）。
- 预算 C：沿用 N2 硬预算 `cost_tot(z,z0,z_src) ≤ eps=0.10` + `project_rms(·,z0,rho=0.15)`；
  **新增内容闸门**：`C_LL(z,z0) ≤ 1.2 × C_LL(z_n2,z0)`（防 LD3 式内容持续退化）。
- 步长回溯 η∈{0.25,0.5,1.0}，接受条件 = 预算内 ∧ 内容闸门 ∧ score 上升；k∈{1,2,4} 三臂。
- 区域 mask：v10 score-gain 版 top-25%（对照 v9 risk-mask）。
- **注意**：梯度投影只有一阶保护，必须解码验证——dev 全量 S10 解码 + referee。

## 7. P4：条件 DSM 小网络（仅在 A–C 过关后）

`D_θ(ṽ,q,t,σ)`：MLP ≈0.5M 参数（v+σ噪声, q, t embedding, σ → 预测 v），
σ∈{0.1,0.3,0.5,1.0} 对数间隔采样，patch 级 batch 8192，fp16，~30min 预算。
`score_θ = (D_θ(ṽ,q,t,σ) − ṽ)/σ²`。与闭式 KDE 在**同一**恢复/编辑评测上比较；
不优于 KDE → 保留无训练方法（这是特性不是失败：零训练成本即结论）。

## 8. 判停条件（预注册，不许事后挪门）

1. **实验 A 不达标**（V10 最优 ≤ R1 或 ≤ LD3）→ 不训练 DSM、不做 P3 接入；转向诊断
   （特征是否仍混内容 / latent 是否保留笔触），写 postmortem。
2. **实验 B 判颜色账**（texture-only 不优于 color-only）→ 判"分布路线未解决核心问题"，
   记录后停止，除非 texture-only 在排除颜色子空间后仍显著为正。
3. **P3 无效**（在完整 cross600 上：V10-E 最优臂 Top-1 相对 LD0 无提升，且
   dino_s/margin/affinity 均无改善）→ 不进入最终确认，写 750 级结论。
   （原 dev 预览门槛作废，理由见 §6 口径修正。）
4. **P4 DSM 不优于 KDE** → 保留 KDE，报告即可。

## 9. 验收标准（S10 统一协议，基准 = LD4 test580）

| 指标 | LD4 | 基础目标 |
|---|---:|---:|
| DINO-S ↑ | 0.4461 | **≥0.452** |
| DINO-C ↑ | 0.7748 | ≥0.770 |
| Top-1 ↑ | 35.56% | ≥35.56% |
| LPIPS ↓ | 0.4836 | ≤0.486 |

另要求：自动去风格化—恢复方向一致性显著优于 R1（A_grad 差 ≥ +0.05 平均），
且 dev/test 趋势一致。DINO-S/CSD 等只做独立评价，**不进入** score 构造或推理梯度。

## 10. 12h 阶段预算

| 阶段 | GPU 上限 | 内容 | 产物 |
|---|---|---|---|
| P0 | 1h | S10 解码一致性探针（decode(z_real) vs 原图）；patch 特征提取吞吐实测；split 完整性 | `runs/v10_local_4070/p0_report.json` |
| P1 | 2h | 五画派 bank（train split）、hψ 训练、三变体 KDE 就绪、σ 敏感性附录 | `bank/`、`hpsi.pt`、`kde_bank.pt` |
| P2 | 2h | 实验 A/B/C 全量 + 对照 | `p2_results.json` + 表 |
| P3 | 2.5h | 最优变体接入，**750 全量** E1/E2/E4 + mask 对照（读数取 cross600） | latents + 750 eval |
| P4 | 2.5h | 条件 DSM 训练 + 同协议对比（仅当 §8 允许） | `dsm.pt`、对比表 |
| P5 | 2h | 750 级确认、逐画派分析、与 LD4-750 基线对照 | `eval/`、`V10_RESULTS.md` |

调度按 P0 实测吞吐动态调整，但**不得挤占独立验证**。

## 11. 代码布局

```
exp/v10/
├── __init__.py
├── features.py      # patch 特征 q/f（可导）、标准化、hψ MLP
├── splits.py        # 画作级 train/holdout 划分 + 完整性检查
├── bank.py          # 建库：per-style (q,v) bank（含 color/texture 拆分视图）
├── kde.py           # 闭式条件 KDE score（V10-A/B/C + 拆分审计视图）
├── destyl.py        # 7 个退化算子（latent 域，autograd-safe）
├── baselines.py     # R1 / LD3 / V6 全局矩 三条对照梯度
├── exp_a_direction.py
├── exp_b_shortcut.py
├── exp_c_generalize.py
├── optim.py         # g_safe 投影 + 预算/内容闸门/回溯
├── run_v10.py       # V10-E 臂 runner（复用 v9 cache 协议）
├── decode_and_eval.py  # 同 v9，OUT 指向 v10
└── manifest.py
```

产物根：`runs/v10_local_4070/`（untracked；manifest 记 sha256 与超参）。
