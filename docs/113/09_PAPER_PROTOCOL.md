# 09 · 论文评测协议（Paper Protocol）固化与复现记录

> 状态：协议复原 **已达成 ±1e-3**；batch 依赖根因已定位、修复方案已定（VAE fp32 + 逐样本编码 + 固定 `eval_batch`），论文口径重测待跑。
> 适用范围：主表 750 张全量评测（5 风格 × 30 源图 × 5 目标风格）。旧口径（CLIP-L/14 + 600 张 + AdaIN 1.0 + dinov2-base）**已废弃**，不再作为对照基准。

---

## 0. 一句话结论

论文协议可以作为唯一官方口径：用 `archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt` 在论文协议下复现，四项指标与论文/`config.json` 常量的偏差分别为 **−9.85e-5 / +2.77e-5 / +8.36e-5 / +8.99e-5**，全部落在 ±1e-4 内（要求 ±1e-3）。唯一遗留问题是 CPU/GPU batch 会改变像素（根因见 §3），已给出修复方案；修复只影响 ≤0.25/255 的像素抖动，不改变协议本身。

---

## 1. 协议固定项（逐条落值）

| # | 项目 | 固化取值 | 代码 / 文件位置 | 备注 |
|---|------|----------|-----------------|------|
| 1 | 测试集目录 | `data/test` | `config.json` → `test_image_dir` | 5 个风格子目录 |
| 2 | 参考图选择策略 | `first_image_of_target_style_test_dir`：目标风格目录内按文件名排序后的**第 1 张** | `utils/run_evaluation.py`（`style_ref_reserved[0]` / `--ref_pool strict` 的反面即默认 `default`） | 参考图同时允许作为内容图（`exclude_source_from_style_refs=false`） |
| 3 | 切分 | **750 all-pairs** = 5 风格 × 30 源图 × 5 目标风格；600 跨风格 + 150 同风格仅作拆分报告 | `utils/run_evaluation.py` 主循环；`tools/run_eval_split.py` | `max_src_samples=30` |
| 4 | ODE 步数 | **8** | `inference.json` → `inference.num_steps=8`、`full_eval.num_steps=8` | 步长 `step_size=1.0`，`style_strength=1.0` |
| 5 | AdaIN scale | **2.0** | `inference.json` → `model.endpoint_adain_scale=2.0` | 通过 `--config_override inference.json` 合并到 ckpt 内嵌配置之上 |
| 6 | 生成图像尺寸 | **256×256**（VAE latent 32×32） | `utils/inference.py` / `config.json` | 不允许为提速改分辨率 |
| 7 | VAE | `stabilityai/sd-vae-ft-ema`（`vae_model=ema`） | `inference.json` → `full_eval.vae_model=ema` | 本地快照 `runs/cache/hf/modelscope/stabilityai_sd-vae-ft-ema` |
| 8 | CLIP 主干 | **`openai/clip-vit-base-patch32`**（ViT-B/32，HF backend） | `utils/run_evaluation.py` → `_DEFAULT_HF_CLIP_REPO`；`_load_clip_from_source` | 本地快照 `/mnt/i/Latent_Style/eval_cache/hf/hub/models--openai--clip-vit-base-patch32`（rev `c237dc49…`） |
| 9 | CLIP 预处理 | 256→**224** 双三次（bicubic）resize，CLIP mean/std 归一化 | `_prepare_clip_pixels` / `_clip_image_size_from_runtime` | 与 openai CLIP 标准一致 |
| 10 | CLIP-S 定义 | 生成图与**前 16 张目标风格参考图 CLIP 特征均值原型**的余弦 | `utils/run_evaluation.py` L4108–4128；`max_ref_cache=16` | 原型先归一化 |
| 11 | DINO 主干 | **`facebook/dinov2-small`**（AutoModel，224 输入） | `utils/compute_dino_metrics.py` → `load_dino`（默认 `facebook/dinov2-small`）；`protocol="paper_canonical_dinov2_small"` | 本地快照 `/mnt/i/Latent_Style/eval_cache/hf/hub/models--facebook--dinov2-small` |
| 12 | DINO 预处理 | `Resize(224, BICUBIC)` + `CenterCrop(224)` + ToTensor + ImageNet mean/std | `compute_dino_metrics.DINO_TRANSFORM` | 从 256 生成图直接变换 |
| 13 | DINO 层选择 | **style/content：`last_hidden_state[:,0,:]`（CLS）**；**structure：`hidden_states[-2]` 的 patch token（去掉 CLS）自相似 MSE** | `compute_dino_metrics.extract_features` L78–80 | CLS 先 L2 归一化 |
| 14 | DINO-S 定义 | 生成图 CLS 与**前 30 张**目标风格参考图 CLS 的**最大**余弦 | `compute_dino_metrics` L135 `--max_refs_per_style=30` | 参考图含参考图本身（`exclude_source_from_style_refs=false`） |
| 15 | DINO-C 定义 | 生成图 CLS 与源图 CLS 的余弦 | 同上 | |
| 16 | LPIPS | `lpips` 包，`net=alex`，输入 `[-1,1]` | `run_evaluation.py` L3776（默认 `eval_lpips_net=alex`）、`to_lpips_input` | |
| 17 | 架构 / ckpt | `hf_oriented_nohh`（oriented HF，无 HH 头） | `experiments/architecture/hf_oriented_nohh.json`；ckpt `archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt` | 论文指向的 `runs/submission/...` 已归档到 `archive/`，**文件未丢失** |
| 18 | `eval_batch` | 论文表原始值为 **16**；修复后建议固定 **32** | `archive/.../paper_eval/epoch_0004/summary.json` → `batch_size=16` | 必须写入 `results.json` |
| 19 | 参考特征缓存 | `max_ref_cache=16`（CLIP），DINO 参考 30 张 | `run_evaluation.py` L3994–4093 | 每样本常量，可离线缓存 |
| 20 | 指标开关 | `only_lpips_clip_style=true`（不计算 clip_dir / clip_content 文本项） | `inference.json` → `full_eval.only_lpips_clip_style=true` | |

论文表目标值（`config.json` `_main_table_metrics` 常量，权威口径）：

| 指标 | 论文值（config 常量） | 文档记录值 |
|------|----------------------|------------|
| DINO-S | **0.4916** | 0.4915 / 0.4916 |
| CLIP-S | **0.7128** | 0.7126 / 0.7128 |
| DINO-C | **0.8103** | 0.8103 |
| LPIPS | **0.2593** | 0.2596 / 0.2593 |

> 口径纠偏：需求里写的 “LPIPS 0.4916” 与 `config.json` 常量不符——0.4916 是 **DINO-S**；LPIPS 的论文值是 **0.2593**。以 `config.json` 为准。

---

## 2. 复现结果表（本机实测）

- 机器：`ssh 3060` → WSL2 Ubuntu-22.04，RTX 3060 12G，torch 2.10.0+cu128
- ckpt：`archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt`（12,218,640 bytes）
- 配置：`--config_override inference.json`，`batch_size=16`，`generation_batch_size=16`，`metric_batch_size=16`，`vae_decode_batch_size=16`，`image_save_workers=4`，`save_summary_grid=true`，`max_src_samples=30`，`max_ref_cache=16`
- 产物：`/mnt/i/Latent_Style/_tmp_evalperf/repro_paper`（750 张图 + `metrics.csv` + `dino_summary.json`）

| 指标 | 论文值（config） | 本机实测（本次重跑） | 差值 | 是否 ±1e-3 | 是否 ±1e-4 |
|------|------------------|----------------------|------|-----------|-----------|
| DINO-S | 0.4916 | **0.4915015** | **−9.85e-5** | ✅ | ✅ |
| CLIP-S | 0.7128 | **0.7128277** | **+2.77e-5** | ✅ | ✅ |
| DINO-C | 0.8103 | **0.8103836** | **+8.36e-5** | ✅ | ✅ |
| LPIPS | 0.2593 | **0.2593899** | **+8.99e-5** | ✅ | ✅ |

附带参考（仓库内 canonical 自带记录 `paper_eval_epoch_metrics.json`，同一 ckpt、同一协议）：

| 指标 | canonical 自带值 | 本次重跑 | 差值 |
|------|------------------|----------|------|
| DINO-S | 0.4915969 | 0.4915015 | −9.54e-5 |
| CLIP-S | 0.7128245 | 0.7128277 | +3.19e-6 |
| DINO-C | 0.8103265 | 0.8103836 | +5.71e-5 |
| LPIPS | 0.2593181 | 0.2593899 | +7.18e-5 |

结论：**论文协议复现通过**（±1e-3，实际全部 ≤1e-4）。差异来源均为生成阶段 batch/浮点抖动（见 §3），非协议字段错误。

### 已知阻塞（与协议无关）

`utils/run_evaluation.py::generate_summary_json` 在最后一步引用未定义变量 `eval_protocol`（约 L5100），`summary.json` 会在图片和 `metrics.csv` 写完之后报 `NameError`。复现值取自 `metrics.csv` + `utils/compute_dino_metrics.py`。修复属于评测侧一行改动，尚未落地（避免与 git 提交线同时写同一文件）。

---

## 3. batch 依赖：根因、数据与修复

### 3.1 根因（实测定位，非推断）

| 组件 | 现象（batch N vs batch 1） | 结论 |
|------|---------------------------|------|
| **VAE encode（fp16）** | latent `max\|Δ\|` = 0.047（bs2）/ 0.127（bs4）/ 0.125（bs8）/ 0.142（bs16）；bs2 vs bs32 = **0.265** | **主因**。fp16 卷积在不同 batch 下选到不同 cuDNN kernel，逐样本结果随之改变 |
| VAE decode（fp16） | 像素 `max\|Δ\|` = **0.0117（≈3/255）** | 次要来源，同为 batch 相关 kernel 选择 |
| 生成网络（bf16 autocast） | latent `max\|Δ\|` = 0.0025–0.0033 | 小幅浮点抖动 |
| AdaIN / GroupNorm / LayerNorm | 统计量均按 `dim=[2,3]` **逐样本**计算，不跨 batch | **不是**原因（已核对 `model.py` L1433–1439；GroupNorm(1,C) 为逐样本归一化） |
| 逐样本 RNG（`randn_like`） | `flow.py` L349 仅当 `bridge_sigma>0` 才注入噪声；本协议 `bridge_sigma=0` | 本协议下**不是**原因 |
| VAE encode 的 `latent_dist.sample()` | 使用全局 RNG；但换成 `mean` 后 batch 差异依旧 | 证明差异来自卷积数值路径，不是 RNG |

端到端实测（fp16 VAE + bf16 生成）：batch 2 vs batch 32 像素 `max|Δ| = 0.0508`，即 **12.95/255**——与训练线记录的 37/255 同源（量级随 batch 跨度与内容放大）。

### 3.2 修复方案（已定，落在评测侧）

1. **VAE 全流程 fp32**（encode + decode）。fp32 下 encode bs8-vs-1 残差从 0.125 降到 0.0057，decode 从 3/255 降到 0.2/255。
2. **源图逐样本编码**（VAE 编码固定 batch=1，与 `--batch` 无关）。150 张源图约几秒，且已有磁盘缓存时可完全跳过。
3. **生成网络 fp32**（关闭 bf16 autocast）用于 parity 档；bf16 仅作为“指标级 ≤1e-4”的快速档，单独标注。
4. 评测 batch 固定为 **32**（论文原始为 16，重跑基线两档都记录），`results.json` 写入 `eval_batch`。
5. 任何提速档位都必须先过“指标 parity ≤1e-4”门禁，再允许用于汇报。

### 3.3 修复前后对照（同一输入，端到端像素/潜变量）

**修复前**（fp16 VAE + bf16 生成）：

| batch | encode latent max\|Δ\| vs 1 | gen latent max\|Δ\| vs 1 | decode pixel max\|Δ\| vs 1 | 端到端 pixel max\|Δ\| |
|-------|-----------------------------|--------------------------|----------------------------|------------------------|
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0.047 | 0.0030 | — | — |
| 8 | 0.125 | 0.0033 | 0.0117 (3/255) | — |
| 32 | 0.142 | 0.0026 | — | 0.0508 (12.95/255, vs bs2) |

**修复后**（VAE fp32 + 生成 fp32 + 逐样本编码）：

| batch | gen latent max\|Δ\| vs 1 | 端到端 pixel max\|Δ\| vs 1 | mean\|Δ\| |
|-------|--------------------------|----------------------------|-----------|
| 1 | 0 | 0 | 0 |
| 2 | 2.75e-4 | **0.000516（0.132/255）** | 2.6e-5 |
| 8 | 2.98e-4 | **0.000868（0.221/255）** | 3.9e-5 |
| 32 | 3.25e-4 | **0.000621（0.158/255）** | 3.9e-5 |
| 64 | 3.25e-4 | **0.000971（0.247/255）** | 3.9e-5 |

> 说明：CUDA 卷积在 fp32 下仍非逐 batch 严格可复现，像素 `max|Δ|` 的理论下限约 1e-3（≈0.25/255，集中在极少数边缘像素），均值差 ~1e-5。**像素级“绝对为 0”只有逐样本串行解码才能保证，会显著拖慢评测**；本协议的判据是**指标逐字段 ≤1e-4**，在该量级下指标变化 ~1e-6。bf16 快速档约为上表的 4–5 倍（~0.8/255 max），仍远低于 1/255 可见阈值。

### 3.4 对历史数字的影响

| 产物目录 | batch | 是否可作论文口径 | 处理 |
|----------|-------|------------------|------|
| `archive/runs_submission/canonical_oriented_epoch4/paper_eval/epoch_0004` | **16** | ✅ 是（论文行原件） | 保留为权威基线，标注 `eval_batch=16` |
| `archive/runs_submission/canonical_oriented_epoch4/r5_eval/epoch_0004` | 2 | ❌ `only_lpips_clip_style=false`，口径不同 | 仅存档，不作对照 |
| `exp/710_baseline_weave_paper_reval` | 8 | ⚠️ reuse-only、`save_summary_grid=false` | 不作论文对照 |
| 所有 CLIP-L/14 + 600 张的旧评测 | 2–16 | ❌ **旧口径（已废弃）** | 报告中必须标注 batch 与“已废弃” |
| 本轮复现 `_tmp_evalperf/repro_paper` | 16 | ✅ | 复现证据，标注 `eval_batch=16` |

规则：**任何历史结果引用时必须带 `eval_batch`**；跨 batch 比较不得用于“提速收益”，除非指标已过 ≤1e-4 parity。

---

## 4. 给用户出计划用的接口

### 4.1 一条可直接复制的评测命令（论文协议，750 张）

```bash
cd /mnt/i/Latent_Style/WEAVE && python3 utils/run_evaluation.py \
  --checkpoint archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt \
  --config_override inference.json \
  --output runs/eval/<model_name>_paper750 \
  --batch_size 16 --generation_batch_size 16 --metric_batch_size 16 \
  --vae_decode_batch_size 16 --image_save_workers 4 \
  --max_src_samples 30 --max_ref_cache 16
```

DINO 指标单独跑（与 `run_evaluation.py` 共用同一张图集）：

```bash
python3 utils/compute_dino_metrics.py \
  --eval_dir runs/eval/<model_name>_paper750 \
  --dino_model_name facebook/dinov2-small \
  --cache_dir /mnt/i/Latent_Style/eval_cache/hf \
  --max_refs_per_style 30
```

### 4.2 新模型必须达到的四条阈值（同一 paper 协议、750 all-pairs）

| 指标 | 阈值 | 判据 |
|------|------|------|
| DINO-S | **≥ 0.4916** | 低于即不达论文基线 |
| CLIP-S | **≥ 0.7128** | 同上 |
| DINO-C | **≥ 0.8103** | 同上 |
| LPIPS | **≤ 0.2593** | 越低越好 |

（若按需求原文的 `0.4918 / 0.7128 / 0.8103 / 0.4916`，其中 0.4916 请按上表理解为 LPIPS=0.2593；`0.4918` 与 config 常量 0.4916 相差 2e-4，属论文排版级别差异，以 `config.json` 为准。）

### 4.3 输出 JSON 必含字段

`results.json` / `eval_meta.json` 必须写入：`protocol="paper"`、`backbone`、`clip_backbone`、`dino_backbone`、`split="all-pairs-750"`、`ref_policy`、`steps=8`、`adain=2.0`、`eval_batch`、`image_size=256`、`ckpt`。
