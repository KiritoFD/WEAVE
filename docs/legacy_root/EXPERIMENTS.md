> **已归档（2026-11-14 仓库重组）**：论文快照期/阶段性文档，现状见 [docs/README.md](../README.md)。

# EXPERIMENTS

> WEAVE 实验索引与复现指南（双盲匿名版本）。
> 所有路径均为项目相对路径；检查点与训练配置的对应关系详见 `CHECKPOINT.md`。
> 术语统一：本文档使用 **Channel-wise AdaIN** 指代 endpoint 子带的对角协方差匹配（mean+std），对应代码字段 `endpoint_adain_scale`。

---

## 1. 概述

### 1.1 实验总览表

| # | 实验名 | 位置 | 论文章节 | 状态 | 说明 |
|---|--------|------|----------|------|------|
| 1 | Main checkpoint (4-epoch) | `exp/dino_s_break/brk_a_ll03_10ep/` | §5 Main Table | ✅ | 主表 WEAVE 行（推理用 epoch 4） |
| 2 | DWT route ablation | `exp/dwt_route_distinct5/` | §5 Ablation | ✅ | 小波路由消融 |
| 3 | Endpoint-only ablation | `runs/ablation_endpoint_only/`（配置 `inference_endpoint_only.json`） | §5 Ablation | ✅ | 仅端点 Channel-wise AdaIN |
| 4 | Stepwise ablation | `runs/ablation_stepwise/`（配置 `inference_block.json`） | §5 Ablation | ✅ | 逐步 Channel-wise AdaIN |
| 5 | endpoint_adain_scale 敏感性 | `runs/endpoint_adain_*.csv` | §5 Ablation (T1/T2/T7/T8/T12/T14) | ✅ | 1.5 / 2.0 / 2.5 扫描 |
| 6 | B2 Qualitative | `exp/b2_qualitative/` | §5 Qualitative | ✅ | 定性对比（8 pairs × 9 methods） |
| 7 | B3 Failure cases | `exp/b3_failure_cases/` | §5 Analysis | ✅ | 失败案例（15 cases） |
| 8 | B5 Compute cost | `exp/b5_compute_cost/` | §5 Efficiency | ✅ | 计算开销 |
| 9 | IDT baseline | `exp/idt_baseline/` | §5 Baselines | ✅ | IDT 基线（750 pairs 实时计算） |
| 10 | Latent-WCT baseline | `exp/latent_wct_baseline/` | §S9 | ⚠️ | Latent-WCT 基线（目录为空，结果在 `results/ours/`） |
| 11 | SD1.5 plugin | `exp/plugin_sd15/` | Frozen SD1.5 | ✅ | SD1.5 插件对比 |
| 12 | DINO avg vs max | `exp/dino_avg_vs_max/` | §5 Discussion | ✅ | DINO 计算口径对比 |
| 13 | Rebuttal | `exp/rebuttal/` | Reviewer response | ✅ | 审稿回应实验 |

### 1.2 数据集与评估协议

- **D5-512**: 5 风格 × 30 内容 = 150 测试图像；风格迁移评估共 750 pairs（含对角 identity）。
- **R5-512**: R5 调谐集，用于选择最佳推理 epoch（tuning split）。
- **DINO 口径**: DINO-S 用 **max**（all-pairs，对每个生成图取与所有参考图 CLS 余弦最大值），非 avg。
- **CLIP-S**: CLIP 风格相似度；**LPIPS**: 内容保持；**DINO-C**: DINO 内容保持。
- 主表指标 C/D/L = DINO-C / DINO-S / LPIPS。

---

## 2. 主表（Table 1）

### 2.1 WEAVE 在 D5-512 上的指标

| 指标 | 值 | 说明 |
|------|-----|------|
| C (DINO-C) | 0.8103 | 内容保持 |
| D (DINO-S) | 0.4916 | 风格相似度（max 口径） |
| L (LPIPS) | 0.2593 | 内容保持（越低越好） |
| CLIP-S | 0.7128 | CLIP 风格相似度（参考） |

- **Checkpoint epoch**: 4（推理用 4-epoch 检查点）
- **训练源**: `exp/dino_s_break/brk_a_ll03_10ep/`（10-epoch 训练，命名 `_10ep`）
- **推理检查点引用**: `runs/submission/canonical_oriented_epoch4/epoch_0004.pt`（配置 `_main_table_metrics.checkpoint`）
- **训练 endpoint_adain_scale**: 1.0
- **评估 endpoint_adain_scale**: 2.0（由 `inference.json` 覆盖）

> **约束**: 主表指标必须从唯一 checkpoint 重新生成。上述 C/D/L 来自 `config.json` 的 `_main_table_metrics` 节，对应唯一 4-epoch 检查点。

### 2.2 复现命令

```powershell
# 1. 训练（10 epochs，brk_a_ll03_10ep 配置）
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1

# 2. 仅评估已有检查点（复用 epoch 检查点，评估缺失 epoch）
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1 -EvalOnly

# 3. 单检查点评估（主表指标复现）
python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference.json
```

---

## 3. 消融实验

### 3.1 消融实验总览

| 消融 ID | 消融轴 | 位置 / 配置 | 论文位置 |
|---------|--------|-------------|----------|
| T1 | DWT route（小波路由） | `exp/dwt_route_distinct5/` | §5 Ablation |
| T2 | Endpoint-only（仅端点 Channel-wise AdaIN） | `inference_endpoint_only.json` → `runs/ablation_endpoint_only/` | §5 Ablation |
| T7 | Stepwise（逐步 Channel-wise AdaIN） | `inference_block.json` → `runs/ablation_stepwise/` | §5 Ablation |
| T8 | endpoint_adain_scale 敏感性 | `runs/endpoint_adain_1p5.csv` / `endpoint_adain_2p0.csv` / `endpoint_adain_2p5.csv` | §5 Ablation |
| T12 | LL 子带部分风格化 (brk_a ll03) | `exp/dino_s_break/brk_a_ll03_10ep/` | §5 Ablation |
| T14 | HF WCT 替换 (brk_w / brk_z) | `exp/dino_s_break/brk_w_hf_wct/` / `brk_z_ll_wct/` | §5 Ablation |

### 3.2 DWT route 消融（T1）

- **位置**: `exp/dwt_route_distinct5/`
- **Checkpoint**: `epoch_0001.pt` ~ `epoch_0010.pt`（10 epoch 全量保存，各 10.26 MB）
- **Config**: `exp/dwt_route_distinct5/config.json`（24 KB）
- **Metrics**: `exp/dwt_route_distinct5/full_eval/epoch_0010/metrics.csv`（4.5 MB）+ `summary.json`（828 KB）
- **复现命令**:
  ```powershell
  python -m WEAVE.run --config exp/dwt_route_distinct5/config.json --inference inference.json
  python run_evaluation.py --checkpoint exp/dwt_route_distinct5/epoch_0010.pt --config exp/dwt_route_distinct5/config.json
  ```

### 3.3 Endpoint-only 消融（T2）

- **配置**: `inference_endpoint_only.json`（2.54 KB，覆盖 `config.json` 推理字段，仅启用端点 Channel-wise AdaIN）
- **输出位置**: `runs/ablation_endpoint_only/`（结果汇总于 `results/ours/d5_750_endpoint_only/`）
- **DINO 指标参考**（`results/ours/d5_750_endpoint_only/dino_summary.json`）:
  - all_dino_s = 0.4836, all_dino_c = 0.6660, all_lpips = 0.3954
- **复现命令**:
  ```powershell
  python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference_endpoint_only.json
  ```

### 3.4 Stepwise 消融（T7）

- **配置**: `inference_block.json`（141 B，逐步 Channel-wise AdaIN）+ `inference_twolevel.json`（93 B，两级）
- **输出位置**: `runs/ablation_stepwise/`
- **复现命令**:
  ```powershell
  python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference_block.json
  ```

### 3.5 endpoint_adain_scale 敏感性（T8）

- **结果文件**: `runs/endpoint_adain_1p5.csv` / `runs/endpoint_adain_2p0.csv` / `runs/endpoint_adain_2p5.csv`
- **配置覆盖**: `results/ours/experiments/endpoint_adain/adain_1p5.json` / `adain_2p0.json` / `adain_2p5.json`
- **关键约束**: **`endpoint_adain_scale=2.0` 是最优值；>2.0（如 2.5）会导致模型崩溃。**
- **复现命令**:
  ```powershell
  # scale=2.0（最优）
  python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference results/ours/experiments/endpoint_adain/adain_2p0.json
  ```

| scale | 结果文件 | 状态 |
|-------|----------|------|
| 1.5 | `runs/endpoint_adain_1p5.csv` | 次优 |
| **2.0** | `runs/endpoint_adain_2p0.csv` | **最优（主表使用）** |
| 2.5 | `runs/endpoint_adain_2p5.csv` | 模型崩溃 |

### 3.6 LL 子带部分风格化（T12）与 HF WCT（T14）

这两组消融由 `exp/dino_s_break/` 下的 brk_* 子目录承载，各子目录训练 10 epoch 后保存 `epoch_0010.pt`：

| 子目录 | 消融内容 | DINO-S | DINO-C |
|--------|----------|--------|--------|
| `brk_a_ll03_10ep` | LL alpha=0.3（主表基准） | 0.4832 | 0.7999 |
| `brk_b_baseline_10ep` | 基线（无 LL 部分风格化） | — | — |
| `brk_d_wll10_10ep` | w_LL=0.1 | — | — |
| `brk_e_wll30_10ep` | w_LL=0.3 | — | — |
| `brk_f_ll04_10ep` | LL alpha=0.4 | — | — |
| `brk_g_ll05_10ep` | LL alpha=0.5 | — | — |
| `brk_h_ll03_15ep` | LL alpha=0.3, 15 epoch | — | — |
| `brk_w_hf_wct` | HF 子带 WCT 替换 | — | — |
| `brk_z_ll_wct` | LL 子带 WCT 替换 | — | — |

DINO 指标文件位于 `exp/dino_s_break/_dino/brk_*.json`（各 ~0.2 KB）。

---

## 4. 定性实验

### 4.1 B2 定性对比

- **位置**: `exp/b2_qualitative/`
- **输出**: `exp/b2_qualitative/grid_8x9.png`（6.24 MB，8 pairs × 9 methods 网格）
- **元数据**: `exp/b2_qualitative/metadata.json`（1.47 KB，含 8 个 src→tgt 风格对）
- **复现命令**:
  ```powershell
  python tools/batch_eval_wfi.py --config config.json --inference inference.json --output exp/b2_qualitative/
  ```

### 4.2 B3 失败案例

- **位置**: `exp/b3_failure_cases/`
- **输出**: `exp/b3_failure_cases/grid_15x4.png`（4.81 MB，15 cases × 4 列）+ `failure_cases.csv`（4.22 KB）+ `stats.json`（6.25 KB）
- **统计**（`stats.json`）: 15 cases，每指标 5 个失败样本；失败类型分布：高 LPIPS(5)、低 CLIP-S(5)、低 DINO-S(5)
- **复现命令**:
  ```powershell
  python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference.json --failure-analysis exp/b3_failure_cases/
  ```

---

## 5. 效率实验

### 5.1 B5 计算开销

- **位置**: `exp/b5_compute_cost/`
- **Metrics**: `exp/b5_compute_cost/summary.json`（6.65 KB）
- **结果汇总**（`results/ours/b5_compute_cost/summary.json`）:

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Encoder (VAE encode) | 14 s | 750 images → latent |
| Decoder (VAE decode) | 58.98 s | latent → 750 images |
| **总计** | **73 s** | 750 images on RTX 3060 |

- **测量配置**: `batch_size=1`, `n_iter=100`, `n_warmup=10`, `input_shape=[1,4,64,64]`
- **Checkpoint**: `runs/submission/canonical_oriented_epoch4/epoch_0004.pt`
- **复现命令**:
  ```powershell
  python tools/benchmark_timing.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference.json
  ```

---

## 6. 基线

### 6.1 IDT 基线（750 pairs 实时计算）

- **位置**: `exp/idt_baseline/`
- **Metrics**: `exp/idt_baseline/summary.json`（614 KB）+ `metrics.csv`（3.55 MB）+ `dino.json`（189 B）
- **关键约束**: **IDT baseline 必须从固定 750 pairs 实时计算，不能手动填旧值。**
- **输出图像**: 603 个（`exp/idt_baseline/images/`）
- **复现命令**:
  ```powershell
  python tools/infer_adain.py --mode idt --output exp/idt_baseline/
  python utils/compute_dino_metrics.py --input exp/idt_baseline/images/ --output exp/idt_baseline/dino.json
  ```

### 6.2 Latent-WCT 基线

- **位置**: `exp/latent_wct_baseline/`（**目录为空**）
- **推理工具**: `tools/infer_latent_wct.py`
- **结果存放**: `results/ours/` 与 `state/dino/P2A-256__wct.json` / `D5-512__wct.json`
- **复现命令**:
  ```powershell
  python tools/infer_latent_wct.py --output exp/latent_wct_baseline/
  ```
- **注意**: 当前 `exp/latent_wct_baseline/` 为空目录，需重新运行上述命令生成结果。

---

## 7. DINO 口径（avg vs max）

### 7.1 实验说明

- **位置**: `exp/dino_avg_vs_max/`
- **Metrics**: `exp/dino_avg_vs_max/comparison.csv`（3.21 KB）+ `ranking_change.txt`（2.44 KB）
- **结论**: **DINO-S 用 max（all-pairs）口径，不是 avg。max 比 avg 高 0.11~0.29。**
- **口径定义**（`results/ours/710_baseline_weave_paper_reval/dino_summary.json`）:
  - `dino_s_definition`: max cosine CLS(gen), CLS(target-style reference)
  - `dino_c_definition`: cosine CLS(gen), CLS(source)
- **avg 口径结果**: `state/dino_avg/D5-512__cut.json` / `D5-512__wct.json` / `_test_weave.json`
- **max 口径结果**: `state/dino/D5-512__weave.json` / `state/dino/dino_main.json`
- **复现命令**:
  ```powershell
  # max 口径（论文使用）
  python utils/compute_dino_metrics.py --input <generated_images> --mode max --output state/dino/D5-512__weave.json
  # avg 口径（对比）
  python utils/compute_dino_metrics.py --input <generated_images> --mode avg --output state/dino_avg/_test_weave.json
  ```

---

## 8. 审稿回应实验

### 8.1 实验说明

- **位置**: `exp/rebuttal/`
- **子任务**:

| 子任务 | 位置 | 内容 |
|--------|------|------|
| task1_generalization | `exp/rebuttal/task1_generalization/` | 泛化性图表 + `task1_generalization_results.json`（4 KB） |
| task1_haar2level | `exp/rebuttal/task1_haar2level/` | 2 级 Haar DWT，`epoch_0015/metrics.csv`（4.5 MB）+ `summary.json`（32.5 KB） |
| task1_sdxl_retrain | `exp/rebuttal/task1_sdxl_retrain/` | SDXL 重训练，`epoch_0013/` + `epoch_0015/` |
| task2_ode_truncation | `exp/rebuttal/task2_ode_truncation/` | ODE 截断分析，`task2_results.json`（11.2 KB）+ `task2_pure_ode_partial.json`（5.6 KB） |
| task4_failure_boundary | `exp/rebuttal/task4_failure_boundary/` | 失败边界，`task4_grid_results.json`（28 KB）+ `task4b_grid_results.json`（24.3 KB） |

- **配置文件**: `configs/exp_rebuttal_hh_head_p256.json` / `exp_rebuttal_w03_p256.json` / `exp_rebuttal_w10_p256.json`
- **复现命令**:
  ```powershell
  python -m WEAVE.run --config configs/exp_rebuttal_hh_head_p256.json --inference inference.json
  ```

---

## 9. 每个实验的详细卡片

### 卡片 1: Main checkpoint (4-epoch)

| 字段 | 值 |
|------|-----|
| 实验名 | Main checkpoint (4-epoch) |
| 位置 | `exp/dino_s_break/brk_a_ll03_10ep/` |
| 论文章节 | §5 Main Table |
| 配置文件 | `config.json`（`_main_table_metrics.checkpoint = runs/submission/canonical_oriented_epoch4/epoch_0004.pt`） |
| 推理配置 | `inference.json`（`endpoint_adain_scale=2.0`） |
| Checkpoint | 训练源 `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt`（10.15 MB）；推理用 4-epoch |
| Metrics | `results/ours/710_baseline_weave_paper_reval/dino_summary.json` + `results/ours/dino_s_break/_dino/brk_a_ll03_10ep_d5.json` |
| 复现命令 | `python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference.json` |
| 注意事项 | 推理用 4-epoch（best on R5-512 tuning split）；训练 10 epoch；`endpoint_adain_scale` 训练 1.0 / 评估 2.0 |

### 卡片 2: DWT route ablation

| 字段 | 值 |
|------|-----|
| 实验名 | DWT route ablation |
| 位置 | `exp/dwt_route_distinct5/` |
| 论文章节 | §5 Ablation |
| 配置文件 | `exp/dwt_route_distinct5/config.json`（24.03 KB） |
| Checkpoint | `exp/dwt_route_distinct5/epoch_0001.pt` ~ `epoch_0010.pt`（各 10.26 MB） |
| Metrics | `exp/dwt_route_distinct5/full_eval/epoch_0010/metrics.csv` + `summary.json` |
| 复现命令 | `python -m WEAVE.run --config exp/dwt_route_distinct5/config.json` |
| 注意事项 | 10 epoch 全量保存；含 `src/` 源码快照 |

### 卡片 3: Endpoint-only ablation

| 字段 | 值 |
|------|-----|
| 实验名 | Endpoint-only ablation |
| 位置 | `runs/ablation_endpoint_only/`（结果在 `results/ours/d5_750_endpoint_only/`） |
| 论文章节 | §5 Ablation |
| 配置文件 | `inference_endpoint_only.json`（2.54 KB） |
| Checkpoint | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| Metrics | `results/ours/d5_750_endpoint_only/dino_summary.json` + `summary.json`（838 KB） |
| 复现命令 | `python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference_endpoint_only.json` |
| 注意事项 | 仅启用端点 Channel-wise AdaIN，关闭中间步风格化 |

### 卡片 4: Stepwise ablation

| 字段 | 值 |
|------|-----|
| 实验名 | Stepwise ablation |
| 位置 | `runs/ablation_stepwise/` |
| 论文章节 | §5 Ablation |
| 配置文件 | `inference_block.json`（141 B）+ `inference_twolevel.json`（93 B） |
| Checkpoint | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| Metrics | `runs/endpoint_adain_*.csv`（汇总） |
| 复现命令 | `python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference_block.json` |
| 注意事项 | 逐步 Channel-wise AdaIN（每步应用），对应代码 `flow.py` 中 Stepwise AdaIN 机制 |

### 卡片 5: endpoint_adain_scale 敏感性

| 字段 | 值 |
|------|-----|
| 实验名 | endpoint_adain_scale 敏感性（T8） |
| 位置 | `runs/endpoint_adain_*.csv` |
| 论文章节 | §5 Ablation |
| 配置文件 | `results/ours/experiments/endpoint_adain/adain_1p5.json` / `adain_2p0.json` / `adain_2p5.json` |
| Checkpoint | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| Metrics | `runs/endpoint_adain_1p5.csv` / `endpoint_adain_2p0.csv` / `endpoint_adain_2p5.csv` |
| 复现命令 | `python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference results/ours/experiments/endpoint_adain/adain_2p0.json` |
| 注意事项 | **`endpoint_adain_scale=2.0` 是最优值；>2.0 会模型崩溃** |

### 卡片 6: B2 Qualitative

| 字段 | 值 |
|------|-----|
| 实验名 | B2 Qualitative |
| 位置 | `exp/b2_qualitative/` |
| 论文章节 | §5 Qualitative |
| 配置文件 | `config.json` + `inference.json` |
| Checkpoint | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| Metrics | `exp/b2_qualitative/metadata.json`（1.47 KB，8 pairs） |
| 复现命令 | `python tools/batch_eval_wfi.py --config config.json --inference inference.json --output exp/b2_qualitative/` |
| 注意事项 | 输出 `grid_8x9.png`（6.24 MB，8 pairs × 9 methods） |

### 卡片 7: B3 Failure cases

| 字段 | 值 |
|------|-----|
| 实验名 | B3 Failure cases |
| 位置 | `exp/b3_failure_cases/` |
| 论文章节 | §5 Analysis |
| 配置文件 | `config.json` + `inference.json` |
| Checkpoint | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| Metrics | `exp/b3_failure_cases/stats.json`（6.25 KB）+ `failure_cases.csv`（4.22 KB） |
| 复现命令 | `python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference.json --failure-analysis exp/b3_failure_cases/` |
| 注意事项 | 15 cases，每指标 5 个失败样本；Minimalism 是最常见源风格(8) |

### 卡片 8: B5 Compute cost

| 字段 | 值 |
|------|-----|
| 实验名 | B5 Compute cost |
| 位置 | `exp/b5_compute_cost/` |
| 论文章节 | §5 Efficiency |
| 配置文件 | `config.json` + `inference.json` |
| Checkpoint | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| Metrics | `exp/b5_compute_cost/summary.json`（6.65 KB） |
| 复现命令 | `python tools/benchmark_timing.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference.json` |
| 注意事项 | encoder 14s + decoder 58.98s = 73s for 750 images on RTX 3060 |

### 卡片 9: IDT baseline

| 字段 | 值 |
|------|-----|
| 实验名 | IDT baseline |
| 位置 | `exp/idt_baseline/` |
| 论文章节 | §5 Baselines |
| 配置文件 | — （IDT 无需训练配置） |
| Checkpoint | — （IDT 为解析方法，无模型权重） |
| Metrics | `exp/idt_baseline/summary.json`（614 KB）+ `metrics.csv`（3.55 MB）+ `dino.json`（189 B） |
| 复现命令 | `python tools/infer_adain.py --mode idt --output exp/idt_baseline/` |
| 注意事项 | **必须从固定 750 pairs 实时计算，不能手动填旧值**；输出 603 张图像 |

### 卡片 10: Latent-WCT baseline

| 字段 | 值 |
|------|-----|
| 实验名 | Latent-WCT baseline |
| 位置 | `exp/latent_wct_baseline/`（**空目录**） |
| 论文章节 | §S9 |
| 配置文件 | — |
| Checkpoint | — （Latent-WCT 为解析方法） |
| Metrics | `state/dino/D5-512__wct.json` / `P2A-256__wct.json`（结果在 state/） |
| 复现命令 | `python tools/infer_latent_wct.py --output exp/latent_wct_baseline/` |
| 注意事项 | 当前 `exp/latent_wct_baseline/` 为空，需重新运行生成结果 |

### 卡片 11: SD1.5 plugin

| 字段 | 值 |
|------|-----|
| 实验名 | SD1.5 plugin |
| 位置 | `exp/plugin_sd15/` |
| 论文章节 | Frozen SD1.5 |
| 配置文件 | — |
| Checkpoint | — （使用冻结 SD1.5） |
| Metrics | `exp/plugin_sd15/sd15_vanilla/eval/metrics.csv`（173 KB）+ `summary.json`（781 B）+ `timing.json`（58.5 KB） |
| 复现命令 | `python tools/bench_sd15.py --output exp/plugin_sd15/` |
| 注意事项 | 1507 张输出图像；含 timing.json 计时数据 |

### 卡片 12: DINO avg vs max

| 字段 | 值 |
|------|-----|
| 实验名 | DINO avg vs max |
| 位置 | `exp/dino_avg_vs_max/` |
| 论文章节 | §5 Discussion |
| 配置文件 | — |
| Checkpoint | — （口径对比，非训练实验） |
| Metrics | `exp/dino_avg_vs_max/comparison.csv`（3.21 KB）+ `ranking_change.txt`（2.44 KB） |
| 复现命令 | `python utils/compute_dino_metrics.py --input <images> --mode max --output state/dino/` + `--mode avg --output state/dino_avg/` |
| 注意事项 | **DINO-S 用 max（all-pairs）口径；max 比 avg 高 0.11~0.29** |

### 卡片 13: Rebuttal

| 字段 | 值 |
|------|-----|
| 实验名 | Rebuttal |
| 位置 | `exp/rebuttal/` |
| 论文章节 | Reviewer response |
| 配置文件 | `configs/exp_rebuttal_hh_head_p256.json` / `exp_rebuttal_w03_p256.json` / `exp_rebuttal_w10_p256.json` |
| Checkpoint | `exp/rebuttal/task1_haar2level/epoch_0015/` + `task1_sdxl_retrain/epoch_0013/` + `epoch_0015/` |
| Metrics | `exp/rebuttal/task1_generalization/task1_generalization_results.json` + `task2_ode_truncation/task2_results.json` + `task4_failure_boundary/task4_grid_results.json` |
| 复现命令 | `python -m WEAVE.run --config configs/exp_rebuttal_hh_head_p256.json --inference inference.json` |
| 注意事项 | 含 5 个子任务（task1_generalization / task1_haar2level / task1_sdxl_retrain / task2_ode_truncation / task4_failure_boundary） |

---

## 10. 术语与约束备忘

| 术语 / 约束 | 说明 |
|-------------|------|
| Channel-wise AdaIN | endpoint 子带对角协方差匹配（mean+std），对应代码字段 `endpoint_adain_scale` |
| `endpoint_adain_scale` | 训练 1.0 / 评估 2.0（`inference.json` 覆盖）；>2.0 模型崩溃 |
| DINO-S 口径 | max（all-pairs），非 avg |
| IDT baseline | 必须从固定 750 pairs 实时计算 |
| 主表 checkpoint | 推理用 4-epoch（best on R5-512 tuning split） |
| Stepwise AdaIN | 训练时可选机制（`flow.py`，默认关闭）；推理时通过 `inference_block.json` 启用 |
| 双盲匿名 | 本文档不含作者、机构、GitHub 用户名等身份信息 |
