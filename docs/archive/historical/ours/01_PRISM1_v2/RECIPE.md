# PRISM-1 v2 (V19 极速高内容单步基座)

> **定位**：极速单步基座 / Hybrid 混合分量｜**状态**：`active`
> **图像目录**：`images_750/` (750 PNGs, 源 `M0_prism1_l100`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `s2_prism1_v2.pt` | `runs/v19/students/s2_prism1_v2.pt` | 4.91 MB | `a15f5928fc70a664...` |

## 2. 架构与训练配方

- **架构**：SpatiallyRoutedChordNet(width=96, chord_dim=128, spatial_router=True), ~1.28M net (+0.31M StyleFiberEncoder)
- **训练配置**：
```json
{
  "script": "exp/v19/routed_student.py",
  "teacher": "V19 Anisotropic S10-VGG+Oklab T4",
  "cond": "StyleFiberEncoder(128D chord = bary[tid] - u_src) + DINO-16 + c_64 + latent luma",
  "loss": "latent MSE + Haar DWT high-freq (0.35) + S10 VGG+Oklab moment (0.12, every 2 steps)",
  "steps": 3600,
  "optimizer": "AdamW"
}
```

## 3. 推理配置

```json
{
  "nfe": 1,
  "anchor": "latent chroma-shift (alpha = min(0.45, 0.20*lam))",
  "strength_knob": "lambda in [0.75, 1.30], default 1.0",
  "decoder": "S10 (decode_s10, scaling_factor=0.18215)",
  "same_task_bypass": true
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `M0_prism1_l075` | 0.7715 | 0.4371 | 0.4849 | 0.5611 | 37.67% | 96.7% |
| `M0_prism1_l100` | 0.7440 | 0.4430 | 0.5057 | 0.5655 | 44.33% | 96.7% |
| `A_p1_l130` | 0.7032 | 0.4504 | 0.5339 | 0.5705 | 50.17% | 96.7% |

## 5. 说明

4090 协议批推理 17.0 ms/img；4090 完整 RGB->RGB (含 A0 encode + DINO + S10 decode) = 70.7 ms/img (b1) / 59.3 ms/img (b16)。
