# Qwen-Image-2.1-Turbo（D5 750 全量 Baseline）

- **模型权重**：`Qwen/Qwen-Image-2.1-Turbo`（bitsandbytes 4-bit NF4 `text_encoder` + `transformer`，BF16 `vae`，`512x512`，8 步 `sample_sigmas`，`true_cfg_scale=1.0`，`use_kv_cache=True`）
- **任务覆盖**：完整 `tasks_750`（`Cross600` + `Same150` = **750 张**，均采用 `sg750_manifest.csv` 冻结单参考图）
- **生成与评测节点**：
  - 生成：RTX 3060 12GB Windows（横向统一速度口径沿用标准在线 TE 端到端实测 **`16,070.78 ms/img`** = `0.0622 img/s`；`Same150` 补充生成阶段为加速出图采用离线 TE 预编码缓存，不纳入横向速度对照）
  - 指标评测：`src.eval.measure`（本地 RTX 4070 Laptop GPU 补全 `Same150`，与既有 `Cross600` 合并并通过 `require_full=True` 校验）
- **资产路径**：
  - 750 完整评估明细：[eval/eval.json](eval/eval.json)（[Cross600](eval/eval_cross600.json) · [Same150](eval/eval_same150.json)）
  - 推理耗时：[timing_cross600.json](timing_cross600.json) · [timing_same150.json](timing_same150.json)
  - 图像目录：`images/`（750 张 PNG，共 `339.4 MB`，按 `.gitignore` 规则不入 git）

## 实测指标汇总（`paper_rgb_v1` 同协议裁判）

| Scope | `n` | `DINO-C` ↑ | `DINO-S` ↑ | `CSD Top-1` ↑ | `CLIP-S` ↑ | `Style Affinity` ↑ | `LPIPS` ↓ | `Target Margin` | `pred_eq_src` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Cross600`** | 600 | `0.7016` | `0.4660` | `42.17%` | `0.6975` | `0.5817` | `0.6256` | `-0.0446` | `56.83%` |
| **`Same150`** | 150 | `0.8508` | `0.6100` | `99.33%` | `0.8231` | `0.7166` | `0.4940` | `+0.3798` | `99.33%` |
| **`All750`** | 750 | `0.7314` | `0.4948` | `53.60%` | `0.7226` | `0.6087` | `0.5993` | `+0.0402` | `65.33%` |

- `Target-Switch Contrast I`（150 源跨目标响应）：`0.0848`
