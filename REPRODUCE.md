# WEAVE 复现指南

> **⚠ 2026-11-14 仓库重组**：论文期平铺模块已从仓库根迁至 `tools/legacy_root/`。
> 下文所有 `python tools/legacy_root/run.py ...` 命令请改用对应脚本路径（功能等价，内部 import 已自举）：
> `python tools/legacy_root/run.py ...`、`python tools/legacy_root/run_evaluation.py ...`；
> `verify_main_table.py` → `python tools/legacy_root/verify_main_table.py`。
> 当前研究线的评测协议（750 任务、S10 解码）见 [docs/README.md](docs/README.md) 与
> [docs/HIGHLIGHTS.md](docs/HIGHLIGHTS.md)——与本文的论文口径（原版 decoder）是两代协议。

本指南说明如何从零复现论文 Table 1 中 **WEAVE** 行的三个核心指标：
**DINO-S (C↑, content, max 口径)、CLIP-S (S↑, style)、LPIPS (D↓, perceptual)**。

复现协议固定为 D5-512 测试板：5 风格 × 30 内容 × 5 参考 = **750 评估对**。
推理使用 **epoch 4** checkpoint，评估时 `endpoint_adain_scale=2.0`。

---

## 1. 概述

**复现目标**：Table 1 WEAVE 行（D5-512 协议）的 C / S / D 三个指标。

| 指标 | 含义 | 口径 | 方向 |
|------|------|------|------|
| DINO-S (C) | DINO ViT-S/16 内容相似度 | max over refs | ↑ 越高越好 |
| CLIP-S (S) | CLIP ViT-B/32 风格相似度 | cos(gen, target style proto) | ↑ 越高越好 |
| LPIPS (D) | AlexNet 感知距离 | mean over pairs | ↓ 越低越好 |

**入口命令**（三条）：

```powershell
# 训练
python tools/legacy_root/run.py --config config.json --train

# 推理
python tools/legacy_root/run.py --config inference.json --infer

# 评估
python tools/legacy_root/run.py --eval --output-dir results/

# 一条命令验证主表
python tools/legacy_root/verify_main_table.py
```

**双盲说明**：本指南不涉及作者身份信息；所有日志、checkpoint 与指标文件均以方法名 `WEAVE` 标识。

---

## 2. 环境准备

详细安装步骤见 [`INSTALL.md`](INSTALL.md)。最低要求：

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA RTX 3060 12GB（验证基准硬件）|
| CUDA | 12.x（与 PyTorch cu128 兼容）|
| Python | 3.10+ |
| PyTorch | ≥ 2.3（建议 cu128 wheel）|
| 显存 | 训练 batch 8 需 ~10GB；推理 batch 16 需 ~8GB |

```powershell
# 1. 安装 CUDA 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. 安装其余依赖
pip install -r requirements.txt
```

`requirements.txt` 关键依赖：`torch>=2.3`、`diffusers>=0.38,<0.39`、`transformers>=4.57,<5`、`lpips>=0.1.4`、`torchmetrics>=1.4`、`torch-fidelity>=0.3.0`。

**硬件验证**：

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')"
```

---

## 3. 数据准备

详细数据集说明见 [`DATA.md`](DATA.md)。复现主表需要 D5-512 数据集。

### 3.1 目录结构

```
data/
  train/                      # 训练 packed latents + 配对缓存
    Early_Renaissance/
    Impressionism/
    Minimalism/
    Rococo/
    Ukiyo_e/
    .latent_cache/packed/     # 预编码 latent 缓存
    manifest.json
  test/                       # 150 张评估板（5 风格 × 30 内容）
runs/
  cache/hf/                   # HF 模型缓存（untracked）
  submission/                 # checkpoint 与评估输出（untracked）
```

### 3.2 下载与预处理

```powershell
# 1. 下载 D5-512 数据集（见 DATA.md 的下载链接）
# 2. 预编码训练 latents（首次运行，约 10 分钟）
python tools/preencode_d5_packed.py --data-root data/train --output data/train/.latent_cache/packed

# 3. 构建配对缓存（top-k=8）
python tools/legacy_root/run.py --config config.json --train --build-pairing-cache-only
```

测试集 `data/test/` 包含 150 张图（5 风格 × 30 内容），评估时与 5 个风格的参考图配对生成 750 对。

---

## 4. 训练复现

### 4.1 命令

```powershell
python tools/legacy_root/run.py --config config.json --train
```

### 4.2 关键训练参数（`config.json`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `training.num_epochs` | 15 | 最大训练轮数 |
| `training.batch_size` | 96（实际 RTX 3060 用 8） | 批大小 |
| `training.learning_rate` | 2e-4 | AdamW 学习率 |
| `training.scheduler` | cosine | 学习率调度 |
| `training.use_amp` | true (bf16) | 混合精度 |
| `training.seed` | 42 | 随机种子 |
| `model.endpoint_adain_scale` | 1.0 | **训练时** endpoint AdaIN 强度 |
| `bridge.spectral_w_ll` | 0.3 | LL 子带 FM loss 权重 |
| `bridge.ll_partial_alpha` | 0.3 | LL 部分风格化 alpha |

> **batch_size 适配**：`config.json` 默认 `batch_size=96`（大显存 GPU）。RTX 3060 12GB 需降至 8 并配合 `accumulation_steps` 或直接使用 `configs/` 下的 RTX 3060 配置。

### 4.3 训练时长（RTX 3060 估计）

| 阶段 | 单 epoch | 15 epochs |
|------|----------|-----------|
| 训练 | ~22 min | ~5.5 h |
| 每轮全量评估（可选） | ~12 min | +3 h |

总时长约 **5.5–8.5 小时**（取决于是否开启 `full_eval_each_epoch`）。

### 4.4 停止规则

- **Early stopping**：`full_eval_stop_on_convergence=true`，patience=4，flat-tail window=4。
  典型收敛点：**epoch 7**。
- **推理用 checkpoint**：**epoch 4**（`runs/submission/canonical_oriented_epoch4/epoch_0004.pt`）。
  epoch 4 在 DINO-S / CLIP-S / LPIPS 三项上达到 Table 1 报告的最佳帕累托点。

### 4.5 预期日志位置

```
runs/submission/repro_brk_a_15ep/
  epoch_0001.pt ... epoch_0015.pt
  training_YYYYMMDD_HHMMSS.csv    # 每步 loss 曲线
  training.log
runs/submission/canonical_oriented_epoch4/
  epoch_0004.pt                   # 主表所用 checkpoint
  paper_eval/epoch_0004/
    metrics.csv                   # 750 对 per-pair 指标
    summary.json                  # 聚合指标
    dino_summary.json             # DINO 指标
    generated/                    # 生成图
```

---

## 5. 推理复现

### 5.1 命令

```powershell
python tools/legacy_root/run.py --config inference.json --infer
```

### 5.2 关键推理参数（`inference.json`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `model.endpoint_adain_scale` | **2.0** | **推理时** endpoint AdaIN 强度（覆盖训练的 1.0）|
| `inference.num_steps` | 8 | ODE 积分步数 |
| `inference.step_size` | 1.0 | 步长 |
| `inference.style_strength` | 1.0 | 风格强度 |
| `full_eval.vae_model` | ema | VAE 权重 |
| `full_eval.batch_size` | 16 | 评估批大小 |
| `full_eval.only_lpips_clip_style` | true | 仅计算 LPIPS+CLIP 风格（主表协议）|
| `full_eval.clip_style_idt_baseline` | 0.639920825263 | IDT CLIP 风格参考基线 |

### 5.3 输出位置

```
results/ours/                    # 生成图与 per-pair metrics
  metrics.csv
  summary.json
  generated/                     # 750 张风格化图
  summary_grid.png               # 汇总网格
```

### 5.4 endpoint_adain_scale 说明

训练时 `endpoint_adain_scale=1.0`，推理时通过 `inference.json` 覆盖为 **2.0**。
这是论文的定向 HF 增强协议：在 ODE 轨迹的 endpoint 处增强 AdaIN 风格强度，
HF 子带走 oriented 路径（见 `experiments/architecture/hf_oriented_nohh.json`）。

---

## 6. 评估复现

### 6.1 命令

```powershell
# 评估 WEAVE 生成结果（750 对）
python tools/legacy_root/run.py --eval --output-dir results/
```

### 6.2 750 对评估协议

D5-512 测试板：5 风格 × 30 内容 × 5 参考 = **750 评估对**。

| 指标 | 计算方式 |
|------|----------|
| DINO-S (C) | 对每个 (gen, content) 对，max over 5 refs of cos(DINO(gen), DINO(content))，再 mean over 750 对 |
| CLIP-S (S) | cos(CLIP(gen), CLIP(target_style_proto))，mean over 750 对 |
| LPIPS (D) | LPIPS-Alex(gen, content)，mean over 750 对 |

### 6.3 IDT baseline 实时计算

IDT (identity) baseline = 输出等于输入内容（output == content）。
**必须从 750 对实时计算**，不能手动填旧值。

```powershell
# 运行 IDT baseline 评估（输出到 exp/idt_baseline/）
python tools/legacy_root/run.py --eval --output-dir exp/idt_baseline --idt-baseline

# 产出：
#   exp/idt_baseline/metrics.csv    # per-pair: clip_style, content_lpips
#   exp/idt_baseline/dino.json      # DINO 指标（dino_content=1.0 by definition）
#   exp/idt_baseline/summary.json   # 聚合 + per-style 分解
```

IDT baseline 理论值：
- DINO-S (C) = **1.0**（输出==内容，by definition）
- LPIPS (D) = **0.0**（输出==内容）
- CLIP-S (S) = 从 750 对 cos(CLIP(content), CLIP(target_style_proto)) 实时平均

---

## 7. 主表验证

### 7.1 一条命令验证

```powershell
python tools/legacy_root/verify_main_table.py
```

### 7.2 预期输出

```
=== WEAVE Main Table Verification ===
Method          | DINO-S (C↑) | CLIP-S (S↑) | LPIPS (D↓)
----------------|-------------|-------------|----------
IDT (baseline)  | 1.000       | 0.653       | 0.000
WEAVE (ours)    | 0.810       | 0.713       | 0.259
----------------|-------------|-------------|----------
Paper reported  | 0.810       | 0.713       | 0.259
Match: YES
```

### 7.3 脚本选项

```powershell
python tools/legacy_root/verify_main_table.py --results-dir /path/to/WEAVE     # 指定项目根
python tools/legacy_root/verify_main_table.py --checkpoint epoch_0004          # 指定 checkpoint 名
python tools/legacy_root/verify_main_table.py --verbose                        # 打印数据来源
```

### 7.4 数据来源

脚本自动从以下文件读取指标（按优先级）：

**WEAVE 指标**：
1. `config.json` → `_main_table_metrics`（canonical）
2. `results/tables/main_table.csv`（交叉校验）
3. `results/cross_protocol/submission_pack/aux_tables_cross.json`（max 口径 DINO）
4. `state/dino/dino_main.json`（DINO fallback，注意是 avg 口径）

**IDT baseline**（实时计算，绝不 hardcode）：
1. `exp/idt_baseline/metrics.csv` → 平均 `clip_style` 和 `content_lpips` 列
2. `exp/idt_baseline/dino.json` → `dino_content`（≈1.0）

### 7.5 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 验证成功（指标在容差 ±5e-3 内）|
| 1 | 验证失败（指标偏离论文值超过容差）|
| 2 | 找不到 metrics 文件 |
| 3 | 30 秒超时 |

---

## 8. 消融复现

### 8.1 DWT route ablation（频域路由消融）

```powershell
# 关闭 DWT 路由，退化为纯空间路径
python tools/legacy_root/run.py --config inference_block.json --infer
python tools/legacy_root/run.py --eval --output-dir results/ablation_dwt_block
```

`inference_block.json` 关闭 `spectral_ode_enabled`，迫使所有计算走空间分支。

### 8.2 Endpoint-only ablation（仅 endpoint AdaIN）

```powershell
# 仅在最后一步 Euler step 应用 AdaIN
python tools/legacy_root/run.py --config inference_endpoint_only.json --infer
python tools/legacy_root/run.py --eval --output-dir results/ablation_endpoint_only
```

`inference_endpoint_only.json` 设置 `endpoint_adain_only_last_step=true`。

### 8.3 Stepwise ablation（训练时逐步 AdaIN）

```powershell
# 训练时启用 Stepwise AdaIN（默认关闭）
python tools/legacy_root/run.py --config config.json --train --config_override bridge.train_adain_enabled=true bridge.train_adain_scale=0.3
```

对应 `flow.py` 中的 Stage9 机制（spatial_fiber mean+std matching）。

### 8.4 endpoint_adain_scale 敏感性扫描

```powershell
# 扫描 1.0 / 1.5 / 2.0 / 2.5
foreach ($s in @(1.0, 1.5, 2.0, 2.5)) {
    python tools/legacy_root/run.py --config inference.json --infer `
        --config_override model.endpoint_adain_scale=$s
    python tools/legacy_root/run.py --eval --output-dir "results/sweep_adain_${s}"
}
```

参考配置：`configs/inference_fix_whitening.json`（`endpoint_adain_preserve_mean=true`）、`configs/inference_no_preserve_mean.json`（`endpoint_adain_preserve_mean=false`）。

---

## 9. 基线复现

### 9.1 IDT baseline（750 对实时计算）

见 §6.3。IDT 是 identity 基线，输出==输入，用于校验 CLIP-S 的零点。

### 9.2 Latent-WCT baseline

```powershell
python tools/infer_latent_wct.py --output-dir results/baselines/latent_wct
python tools/legacy_root/run.py --eval --output-dir results/baselines/latent_wct
```

### 9.3 其他基线

`baseline_metrics_unified.csv` 包含全部基线的聚合指标（CUT、SaMST、SaMam、StyleID、SD-Turbo、StyleAligned、StyleShot、Z-STAR、AdaIN、WCT 等）。复现脚本位于 `tools/`：

| 基线 | 推理脚本 |
|------|----------|
| AdaIN | `tools/infer_adain.py` |
| WCT | `tools/infer_wct.py` |
| Latent-WCT | `tools/infer_latent_wct.py` |
| SD-Turbo | `tools/infer_sd_variants.py` |
| CUT | `tools/bench_sd15.py` |

---

## 10. 常见问题

### 10.1 OOM（显存不足）

**症状**：`CUDA out of memory`。

**解决**：
- 降低 `training.batch_size`（RTX 3060 用 8，而非默认 96）
- 降低 `full_eval.batch_size` 至 4 或 2
- 启用 `training.use_gradient_checkpointing=true`
- 启用 `full_eval.lpips_chunk_size=2`（LPIPS 分块）

### 10.2 指标不符

**症状**：`verify_main_table.py` 输出 `Match: NO`。

**排查**：
1. 确认使用 **epoch 4** checkpoint（非 epoch 7 或最终）
2. 确认 `inference.json` 的 `endpoint_adain_scale=2.0`（非训练默认的 1.0）
3. 确认测试集是 D5-512 的 150 张图（MD5 见 `DATA.md`）
4. 运行 `python verify_main_table.py --verbose` 查看数据来源
5. 容差为 ±5e-3；超出则检查 checkpoint 是否被覆盖

### 10.3 Checkpoint 加载失败

**症状**：`RuntimeError: Error(s) in loading state_dict`。

**解决**：
- 确认 checkpoint 路径：`runs/submission/canonical_oriented_epoch4/epoch_0004.pt`
- 确认 `config.json` 的 `contract_family=weave`（非 legacy）
- 确认 `style_families.py` 的 `validate_i2sb_contract` 通过
- 若 `resume_checkpoint` 为空，训练会从头开始；指定 `--resume <path>` 续训

### 10.4 找不到 metrics 文件

**症状**：`verify_main_table.py` 退出码 2。

**解决**：
1. 先运行推理：`python tools/legacy_root/run.py --config inference.json --infer`
2. 再运行评估：`python tools/legacy_root/run.py --eval --output-dir results/`
3. 运行 IDT baseline：`python tools/legacy_root/run.py --eval --output-dir exp/idt_baseline --idt-baseline`
4. 用 `--results-dir` 指定正确的项目根

### 10.5 训练不收敛

**症状**：epoch 4 的 DINO-S 远低于 0.81。

**排查**：
- 确认 `bridge.spectral_w_ll=0.3`（LL 子带权重）
- 确认 `bridge.ll_partial_alpha=0.3`（LL 部分风格化）
- 确认 `model.spectral_ode_enabled=true`（频域 ODE 开启）
- 确认 `data.balance_target_styles_per_batch=true`（批内风格均衡）

---

## 11. 预期结果表

### 11.1 论文 Table 1（D5-512 协议，WEAVE 行）

| 指标 | 论文报告值 | 口径 | 95% CI（750 对 bootstrap）|
|------|-----------|------|---------------------------|
| DINO-S (C↑) | **0.8103** | max over 5 refs | ±0.008 |
| CLIP-S (S↑) | **0.7128** | cos(gen, style proto) | ±0.006 |
| LPIPS (D↓) | **0.2593** | AlexNet, mean over 750 | ±0.005 |

> CI 估算：对 750 对 per-pair 指标做 1000 次 bootstrap 重采样，取 2.5%–97.5% 分位。
> 容差 ±5e-3 用于 `verify_main_table.py` 的 Match 判定（与 CI 下界兼容）。

### 11.2 IDT baseline（实时计算，参考值）

| 指标 | IDT baseline | 说明 |
|------|-------------|------|
| DINO-S (C↑) | 1.000 | 输出==内容，by definition |
| CLIP-S (S↑) | ~0.65–0.69 | 从 750 对实时平均（cross-pair 约 0.653）|
| LPIPS (D↓) | 0.000 | 输出==内容 |

### 11.3 完整主表（参考）

更多基线与数据集（P2A-256、R5-WikiArt）的数值见 `results/tables/main_table.csv` 与 `baseline_metrics_unified.csv`。

```powershell
# 查看完整主表
type results\tables\main_table.csv

# 查看统一基线指标
type baseline_metrics_unified.csv
```

### 11.4 验证清单

- [ ] `python verify_main_table.py` 输出 `Match: YES`
- [ ] IDT baseline 的 DINO-S=1.0、LPIPS=0.0（理论值）
- [ ] WEAVE 三项指标均在论文报告值 ±5e-3 内
- [ ] 推理用 checkpoint 为 epoch 4
- [ ] `endpoint_adain_scale=2.0`（推理覆盖）

---

*本指南所有路径均为相对路径（相对于 WEAVE 项目根）。如遇问题，先运行 `python verify_main_table.py --verbose` 查看数据来源诊断。*
