# V25 C2nt No-Teacher Reference-Regularized One-Shot (C2nt_sw6_lc0.8.pt)

> **定位**：零教师伪目标依赖的新 One-Shot 强迁移候选 (参考灵敏度 +0.0732)｜**状态**：`active (V25 C2nt breakthrough)`
> **图像目录**：`images_750/` (750 PNGs, 源 `C2nt_sw6_lc0.8_t55`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `C2nt_sw6_lc0.8.pt` | `runs/v25/oneshot_val/C2nt_sw6_lc0.8.pt` | 85.09 MB | `5580d76158b1af31...` |

## 2. 架构与无教师训练配方

- **架构**：SLFUNetV24OneShot (22.23M, Q/K=gray DINO-16, V=RGB ConvNeXt-s2, global 795D fiber + confidence gate) + RefAdapter (0.066M)
- **训练配方**：零教师包；自配对 flow-MSE 锚定内容 + VGG-16 (`relu3_3+relu4_3`) `(1−cos)` 逐样本参考风格正则（`λ_sw=6.0, λ_c=0.8`, `1200 updates`）
- **4 条件参考敏感度（`n=48`, 同内容水位 `dino_c≈0.587`）**：
  - `correct_ref`：`dino_c=0.5873`, **`dino_s_correct=0.3964`**
  - `wrong_ref`：`dino_c=0.5866`, `dino_s_correct=0.3232`（**参考特异性 `+0.0732`**）
  - `same_style_alt_ref`：`dino_c=0.5962`, `dino_s_correct=0.3765`
  - `diff_style_ref`：`dino_c=0.4928`, `dino_s_correct=0.2497`（**差 `+0.1467`**）

## 3. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `C2nt_sw6_lc0.8_t45` | 0.5448 | **0.5235** | 0.5205 | 0.6221 | **42.17%** | 96.7% |
| `C2nt_sw6_lc0.8_t55` | **0.6647** | **0.5027** | **0.3992** | 0.6049 | **28.33%** | 96.7% |

> **诚实边界**：`Early_Renaissance`/`Rococo` 目标类 Top-1 偏弱（5.8%/0.0% @t45），`Impressionism`/`Minimalism` 达 93.3%/90.8%；后续可加 200-pair 平衡参考采样或第二独立种子复验。
