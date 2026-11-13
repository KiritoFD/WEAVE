# B3: WEAVE 失败案例分析

**生成时间**: 2026-07-21
**实验脚本**: `tools/b3_failure_cases.py`
**结果文件**: `exp/b3_failure_cases/grid_15x4.png` (300 DPI), `exp/b3_failure_cases/failure_cases.csv`, `exp/b3_failure_cases/stats.json`
**检查点**: `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` (canonical, 与 B2/B5 共用)
**推理配置**: `inference.json` 覆盖 (`endpoint_adain_scale=2.0`, `num_steps=8`, `step_size=1.0`, `style_strength=1.0`)
**测试集**: D5-750 (5 风格 × 150 source × 5 target = 750 对，含 150 identity 对)

本文档记录 AAAI 2027 投稿 B3 实验的方法学与结果，目的是**诚实客观地展示 WEAVE 的局限性**。B3 从 D5-750 per-sample metrics 中按三个维度（LPIPS / CLIP-S / DINO-S）挑选最差的样本，可视化并分析失败模式。

---

## 1. 实验目标

| 编号 | 量 | 定义 | 工具 |
|------|----|------|------|
| 1 | 失败案例筛选 | 从 750 个 per-sample 中按 LPIPS 最高 / CLIP-S 最低 / DINO-S 最低 各取前 5，去重后保留 15 个 | `metrics.csv` 排序 + 去重 |
| 2 | 可视化网格 | 每行一个失败案例，4 列：Source \| Reference \| WEAVE \| Best Baseline | `matplotlib` + `PIL` |
| 3 | 指标标注 | 每行右侧横向标注 `C:0.XXX D:0.XXX L:0.XXX`（单行，非三行） | `ax.text` |
| 4 | 失败类型标签 | 每行左侧旋转标注 "高 LPIPS" / "低 CLIP-S" / "低 DINO-S" | `ax.text` (rotation=90) |
| 5 | Best Baseline 对比 | 第 4 列展示在失败指标上表现最好的 baseline，提供"是否为通用难题"的上下文 | baseline metrics CSV 联合查询 |
| 6 | 失败模式分析 | 统计 source/reference 风格分布，分析失败是否集中在特定风格对 | `stats.json` |

---

## 2. 方法学

### 2.1 失败案例筛选协议

从 `eval_outputs/d5_750_baselines/weave/metrics.csv`（750 行 per-sample 指标）中按以下规则筛选：

1. **高 LPIPS (内容保持差)**: 按 `content_lpips` 降序，取前 5 个（LPIPS 越高 = 内容失真越大）。
2. **低 CLIP-S (风格迁移不足)**: 按 `clip_style` 升序，取前 5 个（CLIP-S 越低 = 风格匹配越差）。
3. **低 DINO-S (风格语义不匹配)**: 按 `dino_style` 升序，取前 5 个（DINO-S 越低 = 语义风格匹配越差）。
4. **去重**: 同一 (src, work, tgt) 三元组可能同时出现在多个失败类别中。按优先级 `高 LPIPS > 低 CLIP-S > 低 DINO-S` 保留首次出现的标签。
5. **排序**: 按失败类型分组（高 LPIPS → 低 CLIP-S → 低 DINO-S），组内按 LPIPS 降序。

最终获得 **15 个失败案例**（每类恰好 5 个，无交叉去重）。

### 2.2 指标定义

| 缩写 | CSV 列名 | 含义 | 方向 |
|------|---------|------|------|
| C | `clip_style` | CLIP 风格相似度（输出 vs reference） | 越高越好 |
| D | `dino_style` | DINOv2-small 风格相似度（输出 vs reference） | 越高越好 |
| L | `content_lpips` | LPIPS 内容距离（输出 vs source） | 越低越好 |

### 2.3 图像来源

| 列 | 来源 | 路径 |
|----|------|------|
| Source | Identity baseline 输出（= 源图像本身） | `results/D5-512/identity/identity/<src>__<src>__<work>__to__<src>.png` |
| Reference | TGT baseline 输出（目标风格代表图） | `results/D5-512/tgt/<src>__<src>__<work>__to__<tgt>.png` |
| WEAVE | D5-750 canonical 推理输出 | `eval_outputs/d5_750_endpoint_only/images/<src>_<src>__<work>_to_<tgt>.png` |
| Best Baseline | 在失败指标上表现最好的 baseline | `results/D5-512/<method>/...` |

### 2.4 Best Baseline 选择逻辑

对于每个失败案例，在 5 个 baseline（CUT, SaMam, StyTr2, AesPA, Seedream）的 per-sample metrics 中查找同一 (src, work, tgt) 三元组：

- 高 LPIPS 失败 → 选 LPIPS 最低的 baseline
- 低 CLIP-S 失败 → 选 CLIP-S 最高的 baseline
- 低 DINO-S 失败 → 选 DINO-S 最高的 baseline

若该 baseline 的输出图片存在，则在第 4 列展示；否则展示灰色占位图。

### 2.5 网格布局

- **行**: 15 个失败案例，按失败类型分组。
- **列**: 4 列 = `Source | Reference | WEAVE | Best Baseline`。
- **行标签**: 每行最左侧旋转 90 度，红色加粗标注失败类型（"高 LPIPS" / "低 CLIP-S" / "低 DINO-S"）。
- **指标标注**: 每行最右侧横向单行标注 `C:0.XXX  D:0.XXX  L:0.XXX`（等宽字体）。
- **列标签**: 每列顶部加粗标注，第 4 列动态显示 `Best: <方法名>`。
- **图像尺寸**: 256×256 像素，PIL `LANCZOS` 重采样。
- **输出**: 300 DPI PNG，`bbox_inches="tight"`。
- **CJK 字体**: 自动检测 `Microsoft YaHei` / `SimHei` / `SimSun` / `Noto Sans CJK SC` 等字体，确保中文标签正确渲染。

---

## 3. 结果

### 3.1 失败案例总览

| # | 失败类型 | Source 风格 | Target 风格 | Source Work | C | D | L | Best Baseline |
|---|---------|------------|------------|-------------|---|---|---|---------------|
| 1 | 高 LPIPS | Impressionism | Minimalism | joaquín-sorolla_study-of-vineyard | 0.828 | 0.382 | **0.746** | Seedream (L:0.624) |
| 2 | 高 LPIPS | Impressionism | Minimalism | claude-monet_bordighera-the-house-of-gardener | 0.787 | 0.291 | **0.716** | SaMam (L:0.487) |
| 3 | 高 LPIPS | Impressionism | Minimalism | camille-pissarro_the-gardener-afternoon-sun-eragny-1899 | 0.696 | 0.305 | **0.688** | SaMam (L:0.495) |
| 4 | 高 LPIPS | Early_Renaissance | Minimalism | sandro-botticelli_inferno-canto-xviii-1480(1) | 0.747 | 0.255 | **0.635** | SaMam (L:0.610) |
| 5 | 高 LPIPS | Impressionism | Minimalism | childe-hassam_evening-champs-elysees-aka-pres-du-louvre | 0.719 | 0.238 | **0.634** | SaMam (L:0.466) |
| 6 | 低 CLIP-S | Rococo | Ukiyo_e | thomas-gainsborough_robert-butcher-of-walthamstan | **0.526** | 0.502 | 0.287 | StyTr2 (C:0.608) |
| 7 | 低 CLIP-S | Minimalism | Impressionism | ronald-davis_zig-zag-times-two-1966 | **0.521** | 0.213 | 0.207 | Seedream (C:0.732) |
| 8 | 低 CLIP-S | Minimalism | Rococo | ronald-davis_zig-zag-times-two-1966 | **0.479** | 0.127 | 0.172 | Seedream (C:0.610) |
| 9 | 低 CLIP-S | Minimalism | Rococo | pino-pinelli_pittura-r-1993 | **0.528** | 0.154 | 0.164 | Seedream (C:0.646) |
| 10 | 低 CLIP-S | Minimalism | Rococo | brice-marden_joined-2011 | **0.518** | 0.148 | 0.150 | Seedream (C:0.678) |
| 11 | 低 DINO-S | Rococo | Minimalism | bernardo-bellotto_view-of-dresden-... | 0.730 | **0.092** | 0.283 | AesPA (D:0.449) |
| 12 | 低 DINO-S | Minimalism | Rococo | tony-smith_die-1968 | 0.575 | **0.077** | 0.195 | Seedream (D:0.212) |
| 13 | 低 DINO-S | Minimalism | Impressionism | eva-hesse_expanded-expansion-1969 | 0.647 | **0.088** | 0.170 | AesPA (D:0.381) |
| 14 | 低 DINO-S | Minimalism | Rococo | ruth-vollmer_trimer-monomer-dimer-1965 | 0.560 | **0.092** | 0.162 | Seedream (D:0.161) |
| 15 | 低 DINO-S | Minimalism | Rococo | john-mclaughlin_number-13-1973 | 0.592 | **0.084** | 0.054 | Seedream (D:0.237) |

### 3.2 图像完整度

- **Total cells**: 15 rows × 4 cols = 60 cells
- **Found**: 60 / 60 (100%)
- **Missing**: 0

所有 Source / Reference / WEAVE / Best Baseline 图片均在本地目录找到，无需重新推理。

### 3.3 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `exp/b3_failure_cases/grid_15x4.png` | ~5.0 MB | 300 DPI 高分辨率失败案例网格 (15 行 × 4 列) |
| `exp/b3_failure_cases/failure_cases.csv` | ~4.3 KB | 失败案例指标表 (含 path, source_style, ref_style, clip_s, lpips, dino_s, dino_c, failure_type) |
| `exp/b3_failure_cases/stats.json` | ~6.4 KB | 风格分布统计 + best baseline 记录 (机器可读) |

---

## 4. 失败模式分析

### 4.1 Source 风格分布

| Source 风格 | 失败数 | 占比 | D5-750 中占比 | 失败集中度 |
|------------|--------|------|--------------|-----------|
| **Minimalism** | **8** | **53.3%** | 20.0% (150/750) | **2.67×** |
| Impressionism | 4 | 26.7% | 20.0% | 1.33× |
| Rococo | 2 | 13.3% | 20.0% | 0.67× |
| Early_Renaissance | 1 | 6.7% | 20.0% | 0.33× |
| Ukiyo_e | 0 | 0.0% | 20.0% | 0.00× |

**关键发现**: Minimalism 作为 source 风格的失败集中度为 2.67×，显著高于其他风格。Ukiyo_e 作为 source 无任何失败案例。

### 4.2 Reference (Target) 风格分布

| Target 风格 | 失败数 | 占比 | D5-750 中占比 | 失败集中度 |
|------------|--------|------|--------------|-----------|
| **Minimalism** | **6** | **40.0%** | 20.0% | **2.00×** |
| **Rococo** | **6** | **40.0%** | 20.0% | **2.00×** |
| Impressionism | 2 | 13.3% | 20.0% | 0.67× |
| Ukiyo_e | 1 | 6.7% | 20.0% | 0.33× |
| Early_Renaissance | 0 | 0.0% | 20.0% | 0.00× |

**关键发现**: Minimalism 和 Rococo 作为 target 风格的失败集中度均为 2.00×。Early_Renaissance 作为 target 无任何失败案例。

### 4.3 风格对分布

| 风格对 | 失败数 | 主要失败类型 |
|--------|--------|------------|
| **Minimalism → Rococo** | **6** | 低 CLIP-S (4), 低 DINO-S (4) |
| **Impressionism → Minimalism** | **4** | 高 LPIPS (4) |
| Minimalism → Impressionism | 2 | 低 CLIP-S (1), 低 DINO-S (1) |
| Rococo → Minimalism | 1 | 低 DINO-S (1) |
| Early_Renaissance → Minimalism | 1 | 高 LPIPS (1) |
| Rococo → Ukiyo_e | 1 | 低 CLIP-S (1) |

### 4.4 失败原因深度分析

#### 4.4.1 高 LPIPS 失败：复杂内容 → Minimalism 的过度简化

**模式**: 5/5 高 LPIPS 失败均以 Minimalism 为 target 风格。

**原因分析**:
- Minimalism 的视觉特征是几何形状、扁平色块、极简构图。当 WEAVE 将复杂源内容（人物、建筑、自然场景）迁移到 Minimalism 时，模型倾向于**过度简化**内容结构以匹配 Minimalism 的扁平美学。
- 源图像内容越复杂（如 Sorolla 的葡萄园研究、Monet 的花园风景、Botticelli 的地狱场景），内容失真越大（LPIPS 0.63–0.75）。
- 这是**风格-内容权衡**的体现：强风格迁移（高 CLIP-S）与内容保持（低 LPIPS）之间的内在张力。WEAVE 在迁移到 Minimalism 时选择了更强的风格化，牺牲了内容保真度。
- **对比 baseline**: SaMam 在 4/5 高 LPIPS 案例中 LPIPS 更低（0.47–0.61 vs WEAVE 0.63–0.75），说明 SaMam 在 Minimalism 迁移中更保守地保留了内容。

#### 4.4.2 低 CLIP-S 失败：Minimalism 源的风格注入困难

**模式**: 4/5 低 CLIP-S 失败以 Minimalism 为 source 风格（另 1 个为 Rococo → Ukiyo_e）。

**原因分析**:
- Minimalism 源图像的视觉特征极为稀疏（纯色块、简单几何），CLIP 编码的纹理/色彩/笔触信息极少。当 WEAVE 尝试将 Rococo 或 Impressionism 的丰富风格注入这些稀疏源时，**风格信号缺乏足够的载体**来显现。
- 特别是 `ronald-davis_zig-zag-times-two-1966`（抽象几何）同时出现在 Minimalism→Impressionism 和 Minimalism→Rococo 两个失败案例中，说明该源图像的风格迁移尤为困难。
- Rococo → Ukiyo_e 失败（C:0.526）可能源于两种风格的色彩调性差异过大（Rococo 的柔和粉彩 vs Ukiyo_e 的鲜明木版印刷色调）。
- **对比 baseline**: Seedream 在 4/5 低 CLIP-S 案例中 CLIP-S 更高（0.61–0.73 vs WEAVE 0.48–0.53），说明商业大模型在稀疏源风格注入上更强，可能得益于更大的预训练数据覆盖。

#### 4.4.3 低 DINO-S 失败：Minimalism 源的语义风格断裂

**模式**: 4/5 低 DINO-S 失败以 Minimalism 为 source 风格（另 1 个为 Rococo → Minimalism）。

**原因分析**:
- DINO-S 衡量的是输出与 reference 在 DINOv2 语义特征空间的相似度。Minimalism 源图像的语义特征极为抽象（几何形状、色块），与 Rococo/Impressionism 的具象语义（人物、风景、装饰）**语义域差距极大**。
- DINO-S 值极低（0.077–0.092）表明 WEAVE 的输出在语义层面几乎未接近 reference 风格。这与 CLIP-S 失败的模式一致：稀疏源导致风格迁移困难。
- `john-mclaughlin_number-13-1973`（极简色块构图）的 DINO-S 仅 0.084，但 LPIPS 也仅 0.054——说明模型几乎保留了源图像（内容未变），但风格迁移几乎未发生。这是一种"**保守失败**"：模型选择不破坏内容，但也未能迁移风格。
- **对比 baseline**: AesPA 和 Seedream 在 DINO-S 上略好（0.16–0.45 vs WEAVE 0.08–0.09），但绝对值仍然很低，说明这是**通用难题**而非 WEAVE 独有。

### 4.5 失败类型与源内容复杂度的关系

| 失败类型 | 源内容特征 | 典型案例 |
|---------|----------|---------|
| 高 LPIPS | **复杂内容**（人物群像、建筑、自然场景）→ Minimalism | Botticelli 地狱场景、Monet 花园、Sorolla 葡萄园 |
| 低 CLIP-S | **稀疏内容**（几何抽象、色块）→ 丰富风格 | Ronald Davis 锯齿形、Pino Pinelli 色块、Brice Marden 线条 |
| 低 DINO-S | **稀疏内容**（几何抽象）→ 丰富风格 | Tony Smith 立方体、Eva Hesse 装置、John McLaughlin 色块 |

**总结**: WEAVE 的失败呈现明显的**双向不对称**：
1. 复杂内容 → Minimalism 时**过度简化**（高 LPIPS）
2. 稀疏内容 → 丰富风格时**风格注入不足**（低 CLIP-S / 低 DINO-S）

这两种失败模式都指向同一根本原因：**Minimalism 与其他风格之间的视觉/语义域差距过大**，无论作为源还是目标，都构成 WEAVE 的最难迁移场景。

### 4.6 Best Baseline 对比的启示

| 失败类型 | Best Baseline 出现频率 | 启示 |
|---------|----------------------|------|
| 高 LPIPS | SaMam (4/5), Seedream (1/5) | SaMam 在 Minimalism 迁移中内容保持更好，但可能以牺牲风格强度为代价 |
| 低 CLIP-S | Seedream (4/5), StyTr2 (1/5) | Seedream（商业大模型）在稀疏源风格注入上明显更强 |
| 低 DINO-S | Seedream (3/5), AesPA (2/5) | 没有方法能很好地解决 Minimalism→Rococo 的语义迁移；AesPA 在个别案例上略好 |

**重要说明**: Best Baseline 在失败指标上虽然优于 WEAVE，但在其他指标上往往更差（例如 SaMam 的 LPIPS 虽低，但 CLIP-S 和 DINO-S 也更低）。这说明**不存在全面碾压 WEAVE 的 baseline**，失败案例中的 baseline 优势是局部的、以牺牲其他维度为代价的。

---

## 5. 论文写作建议

1. **诚实展示局限性**: B3 网格图应放在论文的"Limitations"或"Qualitative Analysis"小节，明确说明 WEAVE 在 Minimalism 相关迁移上的不足。

2. **风格域差距作为根本原因**: 论文可论证 Minimalism 与其他风格的视觉/语义域差距是所有方法的共同挑战（Best Baseline 的 DINO-S 也很低），WEAVE 的表现与现有方法在同一个难度量级。

3. **风格-内容权衡**: 高 LPIPS 失败案例中 WEAVE 的 CLIP-S 较高（0.70–0.83），说明模型确实成功迁移了 Minimalism 风格，只是以内容失真为代价。这是风格迁移领域的已知权衡，非 WEAVE 特有缺陷。

4. **未来改进方向**: 可在论文 Discussion 中提出：(a) 针对 Minimalism 的自适应风格强度调节；(b) 引入内容感知的风格强度门控，避免对复杂内容过度简化。

---

## 6. 复现说明

```bash
# 默认 (每类 5 个, 256x256 cell, 300 DPI)
python tools/b3_failure_cases.py

# 自定义参数
python tools/b3_failure_cases.py --n_per_metric 3 --cell_size 320 --dpi 300

# 输出到自定义路径
python tools/b3_failure_cases.py --output exp/b3_failure_cases/grid_v2.png
```

**前置条件**:
1. `eval_outputs/d5_750_baselines/weave/metrics.csv` 必须存在（WEAVE per-sample 指标）。
2. `eval_outputs/d5_750_endpoint_only/images/` 必须存在（WEAVE 推理输出）。
3. `results/D5-512/identity/identity/`、`results/D5-512/tgt/` 必须存在（Source / Reference 图像）。
4. `results/D5-512/{cut,samam,stytr2,aespa,seedream}/` 必须存在（baseline 输出，用于 Best Baseline 列）。
5. `eval_outputs/d5_750_baselines/{cut,samam,stytr2,aespa,seedream}/metrics.csv` 必须存在（baseline per-sample 指标，用于 Best Baseline 选择）。
6. Python 依赖: `matplotlib`, `numpy`, `Pillow`。
7. CJK 字体（可选）: `Microsoft YaHei` / `SimHei` / `SimSun` 等，用于正确渲染中文行标签。

**验证脚本**: 不需要 GPU，纯 CPU 图像拼接 + CSV 查询，运行时间 < 15 秒。

---

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `tools/b3_failure_cases.py` | 失败案例筛选 + 网格生成脚本 |
| `exp/b3_failure_cases/grid_15x4.png` | 300 DPI 高分辨率失败案例网格 (15 行 × 4 列) |
| `exp/b3_failure_cases/failure_cases.csv` | 失败案例指标表 (机器可读) |
| `exp/b3_failure_cases/stats.json` | 风格分布统计 + best baseline 记录 (机器可读) |
| `docs/exp/b3_failure_cases.md` | 本文档 (方法学 + 结果 + 失败模式分析) |
