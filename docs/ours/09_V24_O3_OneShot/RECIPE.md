# V24 O3 RGB-ConvNeXt Gated One-Shot (O3_rgb_cnx_gated.pt)

> **定位**：最强免类别 ID 单参考任意风格迁移模型 (PRISMRef 默认引擎)｜**状态**：`active`
> **图像目录**：`images_750/` (750 PNGs, 源 `O3_rgb_cnx_gated_t45`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `O3_rgb_cnx_gated.pt` | `runs/v24/oneshot_val/O3_rgb_cnx_gated.pt` | 85.09 MB | `0832c35f62e4b294...` |

## 2. 架构与训练配方

- **架构**：SLFUNetV24OneShot (22.23M, Q/K=gray DINO-16, V=RGB ConvNeXt-s2, global 795D fiber + confidence gate) + RefAdapter (0.066M)
- **训练配置**：
```json
{
  "script": "exp/v24/p0_o1_o3_value.py",
  "init": "runs/v22/attn_flow/prism_dual_branch.pt",
  "onehot": 0,
  "cross_pairs": "50% V23 cross pairs + 50% self pairs (srcA0 Anchor-Bridge tau0 in [0.35,0.60])",
  "updates": 1200,
  "micro": 12,
  "accum": 4,
  "seed": 20262700
}
```

## 3. 推理配置

```json
{
  "steps": 10,
  "nfe": "20 vel evals x2 CFG = 40 img-eq",
  "strength_knob": "tau0 in [0.40, 0.60] (t45 canonical, wj=2.2)",
  "encoder": "A0 encoder (runs/a0/a0_encoder_only.pt)",
  "decoder": "S10"
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `O3_rgb_cnx_gated_t40` | 0.5456 | 0.5304 | 0.4543 | 0.6293 | 40.83% | -% |
| `O3_rgb_cnx_gated_t45` | 0.5845 | 0.5245 | 0.4213 | 0.6234 | 35.33% | 96.7% |
| `O3_rgb_cnx_gated_t55` | 0.6783 | 0.5025 | 0.3494 | 0.6039 | 24.00% | 96.7% |
| `O3_rgb_cnx_gated_t60` | 0.7274 | 0.4883 | 0.3124 | 0.5915 | 20.33% | -% |

## 5. 说明

SG-25 OOD 125-pair 阶段诊断 DINO-C=0.5867, dS=+0.157; swap_dist=0.442; 4090 RGB->RGB 实测 304.6 ms (fresh) / 265.8 ms (ref-cached, 省 39ms)。
