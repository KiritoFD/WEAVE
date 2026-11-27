# A03：O2 Teacher 质量与筛选偏差诊断（2026-11-20）

任务来源：[agent2.md §四 A03](../../19_pull_together/agent2.md)。科学目标：量化归因 `O3_label_free` 风格偏弱（t45 `D-S 0.4666` vs `O3 0.5245`）中"教师保守/筛选偏向近恒等"的成分，给出参数级改进建议（机制 C1，[02_mechanism_plan.md §C.2](../../19_pull_together/02_mechanism_plan.md)）。工具 `tools/v25_cpu/a03_teacher_quality.py`（CPU-only，numpy+stdlib+`torch.load(map_location="cpu")`），单测 `tests/v25/test_a03_teacher.py`。

数据源（全部本地，未执行教师优化、无 CUDA/ssh）：
- `runs/v24/ref_teacher_pairs_2000.pt`（65.6 MB < 100 MB 上限，含 `idx_c/idx_r/z_cr/mid_cos` + 标量 `mean_all_cos/mean_kept_cos`）
- `runs/v23/train_cross_pairs_1000.pt`（标签教师对照包，字段同构）
- `runs/v25/r3_oneshot_minimal.json`（O3 / O3_label_free / O3_self_only 三臂）
- `runs/v24/p0_o1_o3.json`、`docs/v24/PROGRESS.md`（教师包血缘确认）

## 一、代码级机制链（file:line）

`exp/v24/o2_ref_teacher_pairs.py`：

| 行号 | 事实 |
|---|---|
| `:58-59` | 参考图**全池均匀随机抽取**（`idx_r = (idx_c + randint(1,N)) % N`），无跨画派约束 → 同画派 (content, reference) 对合法出现 |
| `:90-91` | `Adam(lr=0.055)`、默认 `--steps 22`：Adam 每步单坐标位移 ≈ lr（符号一致时），**22 步编辑预算的硬上限 ≈ 22×0.055 ≈ 1.21 latent-σ**——即使目标函数完美，也物理上无法产生大编辑 |
| `:103-108` | `loss = (l_swd + 0.7·l_pal + 0.25·l_loc) + 2.2·l_content + 0.15·l_reg`；`l_content` 是对**源图** ConvNeXt s2/s3 余弦距离，每步都在把输出往源拉 |
| `:116-119` | `mid_cos` = 0.5·(cos_s2 + cos_s3)，衡量对**源**的内容保持度 |
| `:126` | `keep = mid_scores.topk(2000)` —— **筛选判据 = 内容保持度 Top-K**，系统性地筛掉编辑幅度最大的 400 对 |

**"内容相似度筛选 ⇒ 偏近恒等"的机制链条**：筛选分数（对源的 mid_cos）随编辑幅度单调递减 → Top-K 保留最小编辑的 2000 对、丢弃最强编辑的 400 对 → 训练对里"参考风格变化"与"内容变化"被同时压缩 → 学生（`O3_rgb_cnx_gated` 同配方换包，`exp/v24/p2_o3_labelfree.py:52`）学到"少改即安全"。**且筛掉的那 400 对恰是唯一可能教风格的对**——若教师本可产出强编辑，Top-K 会把它们全部剔除。

## 二、数据级：包内分布与筛选分解（V24 label-free 教师，2000/2400）

### 2.1 kept mid_cos 分布（n=2000）

| 统计量 | 值 |
|---|---:|
| mean（=包内标量 `mean_kept_cos`） | 0.96463 |
| std | 0.01168 |
| min（=Top-K 阈值，精确） | **0.94829** |
| q25 / q50 / q75 / q95 | 0.95536 / 0.96178 / 0.97213 / 0.98756 |
| max | 0.99842 |
| frac > 0.95 | **95.0%** |
| frac > 0.98 | 13.35% |
| **frac ∈ C1 目标带 [0.80, 0.90]** | **0%** |
| frac < 0.80 | 0% |

### 2.2 被筛掉的 400 对（精确重构）

Top-K 精确分割 ⇒ rejected 均值 = (`mean_all`×2400 − `mean_kept`×2000)/400：

- `mean_all_cos = 0.96095` → **rejected_mean = 0.94254**。

**关键定量结论**：连被 Top-K **淘汰**的 400 对平均都 ≥ 0.90（C1 带上界）——**全部 2400 个候选无一落入 [0.80, 0.90]**。筛选环节对"教师 cos − 0.90 带上界"超额量的贡献份额 = (0.96463−0.96095)/(0.96463−0.90) = **5.7%**；其余 **94.3% 是优化目标本身**（w_c=2.2 + 22 步 Adam 预算）。即：**仅改筛选规则（后验过滤）救不了这个教师——w_c 必须改**（保留 0% 意味着纯带过滤保留 0 对）。

### 2.3 组成分析（idx // 880 → 画派，train 池按画派分块布局，`exp/v18/__init__.py:clean_balanced_pool`）

- **同画派参考占 kept 的 21.5%**（430/2000；随机抽参考的直接后果；V23 标签教师为 0%，构建时强制 sc≠sr）。同画派参考在构造上封死了"跨风格教学"的上限。
- 但在当前目标下同/跨画派对的 mid_cos **无差异**（same 0.96533 vs cross 0.96444，Welch t=1.41，点二列相关 r=0.032）——目标函数把所有组成都压到近恒等，组成问题被目标问题掩盖；一旦放开 w_c，同画派对的 21.5% 会立刻成为无效对，必须在抽样层先排除。
- 按内容画派的 kept 均值：ER 0.9707 / Imp 0.9609 / Min 0.9600 / Roc 0.9595 / Uki 0.9703（极差仅 1.1pp——均匀近恒等）。
- **per-family 保留率**（rejected 身份未保存，按均匀抽样期望 480 候选/画派估计，±1.96×sd≈±0.040）：

| 画派 | kept | 保留率估计 | 95% CI |
|---|---:|---:|---|
| Early_Renaissance | 431 | 0.898 | [0.818, 0.978] |
| Impressionism | 388 | 0.808 | [0.728, 0.888] |
| Minimalism | 352 | **0.733** | [0.653, 0.813] |
| Rococo | 394 | 0.821 | [0.741, 0.901] |
| Ukiyo_e | 435 | 0.906 | [0.826, 0.986] |

Minimalism 最低（−17.3pp vs Uki）且方向合理：极简内容对任何编辑最敏感、最易被内容分筛掉——但该内容恰是风格最脆弱、最需要"温和强编辑"示范的类。

### 2.4 V23 标签教师对照（同字段构，1000/1500）

| 统计量 | V23 标签教师 | V24 label-free |
|---|---:|---:|
| mean_all → mean_kept | 0.61316 → 0.64498 | 0.96095 → 0.96463 |
| rejected 均值（精确） | 0.54951 | 0.94254 |
| 筛选位移 | +0.0318 | +0.0037 |
| kept q50 / q95 / max | 0.62539 / 0.77593 / 0.94359 | 0.96178 / 0.98756 / 0.99842 |
| 带内 [0.80,0.90] / 低于带 | 2.8% / **96.8%** | 0% / 0% |
| 同画派对 | 0%（构造） | 21.5% |
| 保留率极差 | Imp 0.523 ↔ Min 0.950 | Min 0.733 ↔ Uki 0.906 |

两个教师在 C1 带两侧**对称失败**：标签教师 96.8% 低于带（过猛，mid_cos 0.64），label-free 教师 100% 高于带（过保守，mid_cos 0.96）。C1 三档 (2.2/1.2/0.7) 本质是在这两个失败端点之间插值。

**度量口径警告**：V23 `mid_cos` = ConvNeXt 空间 s32/s16 余弦（`exp/v23/build_cross_pairs_4090.py:127-133`），V24 = PatchSWD token s2/s3 余弦（`exp/v24/o2_ref_teacher_pairs.py:116-119`）——同为"对源的 ConvNeXt 中层内容保持度"、同为 [0,1] 方向，但**非同一公式**，0.645 vs 0.965 的对比是方向性结论而非逐点可比。

## 三、学生级交叉验证：教师幅度 → 学生幅度论证链

血缘（代码确认）：`O1/O2/O3`（标签臂）训练包 = V23 `train_cross_pairs_1000.pt`（`exp/v24/p0_o1_o3_value.py:432`）；`O3_label_free` = 同一 `train_o_arm` 配方仅换包为 `ref_teacher_pairs_2000.pt`（`exp/v24/p2_o3_labelfree.py:52`）；`O3_self_only`（R3 控制）= reference := 源图本身，即**恒等教师**（mid_cos ≡ 1.0）。

教师幅度阶梯与学生在同一评测协议下的读数（cross600）：

| 教师 | 教师 mid_cos | 学生 Top-1 | D-S | D-C | Affinity |
|---|---:|---:|---:|---:|---:|
| `self_only`（恒等） | 1.000 | 12.33% | 0.47342 | 0.7233 | 0.5768 |
| `label_free`（V24 包） | 0.9646 | 11.33% | 0.46661 | 0.6769 | 0.57349 |
| `O3`（V23 标签包） | 0.645* | 35.33% | 0.52449 | 0.58453 | 0.6234 |

*V23 度量变体（§2.4 警告）。

### 归因分解（工具 `attribution_chain`，t45）

| 量 | Top-1 | D-S | D-C |
|---|---:|---:|---:|
| 待解释差距 `O3 − label_free` | **24.00pp** | 0.05788 | −0.09234 |
| 恒等控制已覆盖 `O3 − self_only` | 23.00pp | 0.05107 | −0.13877 |
| **教师边际贡献 `label_free − self_only`** | **−1.00pp** | **−0.00681** | **−0.04643** |
| 恒等控制份额 | **95.8%** | **88.2%** | — |

t55 同构：Top-1 差距 15.17pp、控制份额 90.1%；D-S 差距 0.0450、控制份额 98.2%；教师边际全部为负。

**解读**：label-free 教师包的学生**不高于恒等教师控制臂**（三指标边际 ≤ 0）——这 2000 对真参考配对作为监督信号，其风格信息净贡献 ≈ 0。因此：

1. `O3_label_free` vs `O3` 的 24.00pp / 0.0579 D-S 差距中，**≥88%（D-S）～96%（Top-1）在恒等教师处就已存在**；按"低于恒等控制"的读法甚至 ≥100%——教师保守性解释了风格差的全部主体。
2. 教师并非"没动"：label_free 相对恒等把 D-C 降了 0.0464（确实做了编辑），但**单位内容牺牲换来的 D-S 为负**（`style_per_content`：O3 +0.368，label_free −0.147；t55 为 +0.463 vs −0.019）——编辑是**无方向性扰动**而非风格迁移，与 mid_cos 0.96 的"贴源 + 噪声"形态一致。
3. 在教师保守性内部：**Top-K 筛选仅贡献 5.7%**，目标函数（w_c=2.2、22 步 Adam 预算 ≈ 1.2σ 位移上限）贡献 **94.3%**（§2.2）。

**对任务原始命题的精确回答**："'教师筛选偏向近恒等'造成 label_free 风格差"——若"筛选"指 Top-K 后验筛选本身，则该命题**被拒绝**（5.7%）；若指教师管线整体保守（目标 + 预算 + 筛选），则**被证实**（≥88–96%，边际为负）。改进必须落在目标函数层。

## 四、BLOCKED 项（本地不可算）与导出清单

本地无 4400 行训练 latent 缓存（`runs/v18/cache/ccf_pool_cache.pt` 仅在 GPU 主机），故**逐对 `latent_L2 = ||z_cr − z_src||`、颜色变化、风格增量与 mid_cos 的 Pearson/Spearman 相关无法本地计算**。包内唯一可行代理 `corr(mid_cos, ||z_cr||_RMS)` 无信息量（spearman −0.080 / pearson −0.135——RMS 被逐图源差异主导，非编辑幅度），不作为证据。

**所需字段清单**（逐对，kept+rejected 若重生成）：
1. `mid_cos`（kept + rejected 全体 2400）
2. `latent_l2 = ||z_fin − z_src||_2`
3. `color_fiber_l2`（27D Oklab fiber 的 `||pal(z_fin) − pal(z_src)||`）
4. `swd_to_ref_before / swd_to_ref_after`（风格增量，C1 双指标筛选的第二指标）
5. `idx_c / idx_r / sid_c / sid_r`（家系交叉校验）

**4090 侧导出命令（纯元数据，不导权重，~1 min）**——在 `/root/Workspace/WEAVE`（4090 的 `runs/v24/cache/ref_teacher_pairs_2000.pt` 与 `runs/v18/cache/ccf_pool_cache.pt` 均在）：

```bash
python - <<'PY'
import json
import numpy as np
import torch
c = torch.load("runs/v18/cache/ccf_pool_cache.pt", map_location="cpu", weights_only=False)
z, sid = c["train"]["z"].float(), c["train"]["sid"].long()
p = torch.load("runs/v24/cache/ref_teacher_pairs_2000.pt", map_location="cpu", weights_only=False)
ic = p["idx_c"]
l2 = (p["z_cr"].float() - z[ic]).flatten(1).norm(dim=1)
json.dump({"latent_l2": l2.tolist(), "mid_cos": p["mid_cos"].tolist(),
           "idx_c": ic.tolist(), "idx_r": p["idx_r"].tolist(), "sid_c": sid[ic].tolist()},
          open("runs/v24/ref_teacher_pairs_2000_meta.json", "w"))
r = float(np.corrcoef(l2.numpy(), p["mid_cos"].numpy())[0, 1])
print("latent_l2 mean=%.4f p50=%.4f p95=%.4f; pearson(mid_cos,l2)=%.3f"
      % (l2.mean(), l2.quantile(0.5), l2.quantile(0.95), r))
PY
```

产出 `ref_teacher_pairs_2000_meta.json`（约 0.5 MB）拷回本地后，`a03_teacher_quality.pearson/spearman` 可直接补算相关性并检验"筛选上限"论证。

## 五、给机制 C1 的参数块（可直接交 DEV/G2）

```text
[O2-Teacher-v2 校准 · A03 参数块 2026-11-20]
前提事实：现行教师 2400 候选全部高于 C1 带 [0.80,0.90]（kept 0.9646 / rejected 0.9426），
         纯后验带过滤保留 0 对 → 必须改目标函数，筛选只做二级闸门。

(sampling)
  cross_family_only        = true      # 排除同画派参考（现行 kept 中占 21.5%，构造上封死跨风格教学）
  stratify                 = 20 有向画派组 × 120 候选（对齐 V23 均衡覆盖）
  label_usage_decision     = 二选一，需 G2/总控拍板：
    (a) 严格零标签（保持 O2 铁律）：同画派判定不读 sid，改用无标签代理的
        参考拒绝规则——参考图与内容图的 27D Oklab fiber 余弦 > 0.985 即拒绝
        （风格统计几乎重合 → 无可教增量）；分层改用 fiber 聚类 k=5。
    (b) 抽样层用 v18 train 池 style-block 布局（idx//880）分层、损失保持零标签：
        家系身份只进抽样、不进任何目标/损失/评测；偏离"完全不读标签"字面约束，
        但不注入类重心/类先验（与 V23 教师 bary[tid_r] 的标签注入性质不同）。
    注：b 选件须在实验记录中显式声明"sampling-label-aware"；a 选件无争议但
    fiber 相似对同画派异作品可能误拒（阈值 0.985 需在 600 对诊断子集上标定）。
(objective)
  w_c:  T-conservative 2.2 | T-medium 1.2 | T-expressive 0.7
  w_reg = 0.05（现行 0.15 同样偏保守，降幅与 w_c 同步）
  其余项不动：l_swd + 0.7·l_pal + 0.25·l_loc
(optimizer_budget)
  steps: 22 / 40 / 48（22×lr0.055 的 Adam 位移上限 ≈1.2σ，expressive 档需 ~2× 预算才能物理到达 0.80–0.86）
  lr = 0.055 不变（变量隔离：只改 w_c 与步数）
(selection · 双指标，仅作二级闸门)
  keep iff mid_cos ∈ [0.80, 0.92] AND style_increment ≥ q25(候选)
  style_increment = SWD-to-ref 相对源基线的下降量（需 §四字段 4 导出）
  per_family_retention_floor = 0.30（任一画派低于即回退该档 w_c +0.3）
(expected)
  mid_cos 带预测: T-conservative [0.94,0.97] | T-medium [0.84,0.91] | T-expressive [0.78,0.86]
  双指标后保留率预测: 0.83 / [0.55,0.70] / [0.40,0.60]
  600 对诊断子集先跑（02_mechanism_plan §C.2 流程不变）；验收看 t45 D-S ≥ +0.02 @ D-C ≥ 0.66（§C.4 判据）
风险注记
  · w_c→0.7 时 l_swd 主导可能产生伪影（LPIPS+拼图审计不可省）
  · V23 教师证明"过猛"一侧同样失败（96.8% 低于带）；T-expressive 若 kept q50 < 0.80 应立即回退
  · mid_cos 两代度量公式不同（§2.4），C1 产线统一用 V24 PatchSWD 版本以保持可比
```

## 六、复算命令

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  python -m tools.v25_cpu.a03_teacher_quality          # 本报告全部数字（JSON 到 stdout）

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  python -m pytest tests/v25/test_a03_teacher.py -q    # 8 passed
```
