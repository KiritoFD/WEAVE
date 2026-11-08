# Baseline 评测图与数据（论文引述的 13 个方法 × 3 个数据集）

> 这里存放论文中引述的 baseline **评测输出图**（不是训练数据集），用于与 WEAVE / V6 / V9 的生成结果做视觉和数值对比。
> 每个方法的原始规模是 **每个数据集 750 张**（150 源图 × 5 目标风格），三个数据集分别是
> **D5-512**（512×512）、**P2A-256**（Photo2Art，256×256）、**R5-WikiArt**（hold-out 5 风格，512×512）。

---

## 一、压缩包位置（在 3060 上）

全部位于 3060 的 `I:\Latent_Style\WEAVE\_baseline_stage\`（WSL 侧 `/mnt/i/Latent_Style/WEAVE/_baseline_stage/`）。

| 文件 | Windows 路径 | 大小 | SHA-256 |
|---|---|---:|---|
| `baseline_D5-512.tar.gz` | `I:\Latent_Style\WEAVE\_baseline_stage\baseline_D5-512.tar.gz` | 1.6 GB | `02baa05bcd655edf62aae1ee2599f6a4eeede0faf9b65e50a622d4ee21e77b05` |
| `baseline_P2A-256.tar.gz` | `I:\Latent_Style\WEAVE\_baseline_stage\baseline_P2A-256.tar.gz` | 223 MB | `209993c64926c8f43196e8f6d2e8c3d461a65d8f018a140690efe768c6d05b9e` |
| `baseline_R5-WikiArt.tar.gz` | `I:\Latent_Style\WEAVE\_baseline_stage\baseline_R5-WikiArt.tar.gz` | 1.3 GB | `58d39e7265ddc6f0f8abb199f88ef97b40d5a3d8da2932fd665439319e138b0e` |
| `inventory.json` | `I:\Latent_Style\WEAVE\_baseline_stage\inventory.json` | 8 KB | — |

每个 tar.gz 内部结构一致：

```
jpg/<method>/<原文件名>.jpg      生成图（PNG→JPEG q=90 重编码，尺寸与原图一致）
sheets/<method>.jpg             5 目标风格 × 4 源图 的对比拼图
```

下载后解压到本目录：

```bash
cd docs/baseline
tar -xzf baseline_D5-512.tar.gz      # 得到 jpg/ 与 sheets/
tar -xzf baseline_P2A-256.tar.gz
tar -xzf baseline_R5-WikiArt.tar.gz
```

> 建议先核对 SHA-256。`docs/baseline/images/` 与 `docs/baseline/sheets/` 已加入 `.gitignore`（体积原因），
> 只有 `inventory.json`、本文档和指标 CSV 会进仓库。

---

## 二、逐方法清单与**缺失说明**

图数为该目录下实际文件数；`—` 表示该数据集上从未跑过这个方法。

### D5-512（13/13 齐全）

| 方法 | 图数 | JPEG 体积 | 接触拼图 | 逐任务指标 CSV |
|---|---:|---:|:---:|:---:|
| Identity | 750 | 175.6 MB | ✔ | — |
| AdaIN | 750 | 65.1 MB | ✔ | — |
| WCT (VGG19) | 750 | 1.3 MB | ✔ | — |
| CUT | 750 | 47.7 MB | ✔ | ✔ |
| SaMST | 750 | 7.3 MB | ✔ | — |
| SaMAM | 750 | 203.6 MB | ✔ | ✔ |
| SD-Turbo | 750 | 56.8 MB | ✔ | — |
| StyleID | 750 | 56.2 MB | ✔ | — |
| StyleShot | **745** | 63.2 MB | ✔ | — |
| StyTr2 | 750 | 50.2 MB | ✔ | ✔ |
| AesPA | 750 | 34.3 MB | ✔ | — |
| WEAVE (已发表) | 750 | 12.5 MB | ✔ | ✔ |
| Seedream 4.5 | 750 | 656.5 MB | ✔ | ✔ |

**D5-512 额外附带**（不在论文主表，但磁盘上有，一并打包）：
`stylealigned` 750、`zstar` 733、`weave_oriented_e4` 750、`weave_yfix` 750。

> ⚠️ `stylealigned` / `zstar` 的 750 张是**旧批次**，其源图取自训练池而非冻结测试源图，
> 按 `eval_protocol/tasks_750.json` 严格解析只有 **15 个任务**能对上（3 源图 × 5 风格）。
> 因此它们**不能**进入冻结 600 的排名，只可作定性参考。详见 `eval_protocol/baseline_audit_strict.json`。

### P2A-256（13/13 齐全，两处不足 750）

| 方法 | 图数 | JPEG 体积 | 拼图 | 指标 |
|---|---:|---:|:---:|:---:|
| Identity | 750 | 16.5 MB | ✔ | — |
| AdaIN | 750 | 14.8 MB | ✔ | — |
| WCT | 750 | 15.2 MB | ✔ | — |
| CUT | **588** | 15.2 MB | ✔ | ✔ |
| SaMST | 750 | 12.7 MB | ✔ | — |
| SaMAM | 750 | 10.4 MB | ✔ | ✔ |
| SD-Turbo | 750 | 17.7 MB | ✔ | — |
| StyleID | 750 | 22.2 MB | ✔ | — |
| StyleShot | 750 | 47.7 MB | ✔ | — |
| StyTr2 | 750 | 16.0 MB | ✔ | ✔ |
| AesPA | 750 | 11.6 MB | ✔ | ✔ |
| WEAVE | 750 | 13.6 MB | ✔ | ✔ |
| Seedream 4.5 | **721** | 18.6 MB | ✔ | ✔ |

**缺失**：`cut` 少 162 张、`seedream` 少 29 张。这两处在任何 P2A 横向比较里都必须标注为**非完整 750**，
均值不能直接与其他方法比。

### R5-WikiArt（10/13，缺 3 个方法）

| 方法 | 图数 | JPEG 体积 | 拼图 | 指标 |
|---|---:|---:|:---:|:---:|
| Identity | **缺** | — | — | — |
| AdaIN | **缺** | — | — | — |
| WCT | **缺** | — | — | — |
| CUT | 750 | 47.7 MB | ✔ | ✔ |
| SaMST | 750 | 86.9 MB | ✔ | — |
| SaMAM | 750 | 245.2 MB | ✔ | ✔ |
| SD-Turbo | **1123** | 74.0 MB | ✔ | — |
| StyleID | 750 | 61.0 MB | ✔ | — |
| StyleShot | **740** | 53.8 MB | ✔ | — |
| StyTr2 | 750 | 45.6 MB | ✔ | ✔ |
| AesPA | 750 | 41.9 MB | ✔ | ✔ |
| WEAVE | 750 | 11.9 MB | ✔ | ✔ |
| Seedream 4.5 | **724** | 647.8 MB | ✔ | ✔ |

**缺失/异常**：
1. **Identity / AdaIN / WCT 在 R5-WikiArt 上完全没跑过** —— `results/R5-WikiArt/` 下没有这三个目录。
   因此该数据集的排名少 3 个经典基线，`results/tables/main_table.csv` 里这三个方法在 R5-WikiArt 上的行
   来自别处（`R5-512` 目录），口径不同，不要混用。
2. `sdturbo` 有 **1123** 张，多于 750：目录里混入了多批次渲染结果，需要按文件名筛到冻结任务才能用。
3. `styleshot` 少 10 张、`seedream` 少 26 张。

### 指标 CSV 的覆盖情况

逐任务指标只对 6 个方法存有原始 `metrics.csv`（`eval_outputs/d5_750_baselines/*/metrics.csv`）：
**CUT / SaMAM / StyTr2 / AesPA / WEAVE / Seedream 4.5**。
其余 7 个方法（Identity / AdaIN / WCT / SaMST / SD-Turbo / StyleID / StyleShot）**没有原始逐任务 CSV**，
其数值只存在于我们后来用 Evaluator V2 重新计算的结果里（见下）。

---

## 三、统一口径的指标来自哪里

论文主表统一数字以我们自己的 Evaluator V2 重算结果为准，而不是上面这些历史 CSV：

- `runs/v6/evaluator_v2_all_results.json` —— 13 个 baseline + V6 全量指标（`paper_rgb_v1` 协议）
- `runs/v6/baseline_reval_summary.json` —— 早期一版重算
- `eval_protocol/baseline_audit_strict.json` —— 每个方法在冻结 750 任务上的严格匹配审计
- `runs/blog_sbflow/eval_*/style_scores_compact.json` —— V9 在 60 任务子集上的结果

**注意口径差异**：`evaluator_v2_all_results.json` 里的 750 任务只针对 **D5-512**；
P2A-256 与 R5-WikiArt 至今**没有**用同一套 Evaluator V2 重算过，
它们的数值仍来自历史 `metrics.csv`（CLIP-S / LPIPS / MUSIQ 口径）。跨数据集不可直接比大小。

---

## 四、如何用这些图做对比

1. 先看拼图：`sheets/<method>.jpg` 是 5 列（目标风格）× 4 行（同一源图），横向即可看出该方法的风格强度与内容保持。
2. 再看单图：`jpg/<method>/` 与源图 `data/test/<style>/` 同名可对齐（部分方法文件名带双前缀，需用
   `tools/evaluation_v2/manifest.py` 的严格解析器）。
3. 数值对照见 `docs/v6/EVALUATOR_V2_REPORT.md` 与 `docs/v9/V9_RESULTS.md`。

---

## 五、一句话结论（当前排名位置）

在**同一冻结 600 跨风格任务、同一 `paper_rgb_v1` 协议**下，Target Top-1 排名：

`StyleShot 92.6% > StyleID 73.2% > StyTr2 39.5% > Seedream 37.3% > AdaIN 34.3% > AesPA 31.0% > WEAVE(原版) 16.7% > CUT 5.7% > V6 ScoreDiff 5.0% > V6 D1 3.2% > SaMAM 2.5% > SD-Turbo 0.5% ≈ Identity 0.5%`

WEAVE 各代的位置：**V9（60 任务子集，31.7%）不能直接插进这张表**（子集 + DCAE 潜空间不同）；
**V6 系列目前排在 CUT 之后**，仍显著落后于扩散类强风格方法。
详见 `docs/baseline/RANKING.md`。
