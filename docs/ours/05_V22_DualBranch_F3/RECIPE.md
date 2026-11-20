# V22 DualBranch F3 (prism_dual_branch.pt)

> **定位**：V23 cs_j bug 修复后 20 NFE 高风格双分支 + One-Shot 底座｜**状态**：`frozen (also O3 initialization base)`
> **图像目录**：`images_750/` (750 PNGs, 源 `F3_dual_fixed_orig`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `prism_dual_branch.pt` | `runs/v22/attn_flow/prism_dual_branch.pt` | 83.76 MB | `ad25267445bf46bc...` |

## 2. 架构与训练配方

- **架构**：SLFUNet(base_dim=144, u_dim=128), 21.95M
- **训练配置**：
```json
{
  "script": "exp/v22/train_dual.py",
  "note": "双分支 cs_j 专训底座; 修复 layout_scale 误清零 bug 后在 DINO-C=0.6096 达 DINO-S=0.5522"
}
```

## 3. 推理配置

```json
{
  "anchor": "T4",
  "branches": "cs_j (20 NFE)",
  "gate": "original",
  "solver": "D5_SOLVER",
  "decoder": "S10"
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `F3_dual_fixed_orig` | 0.6096 | 0.5522 | 0.5723 | 0.6419 | 49.50% | 96.7% |

## 5. 说明

全仓除 StyleGallery 外 DINO-S@DINO-C>=0.61 最高单模型之一, 同时是 V24 O1/O2/O3 One-Shot 的热启动底座。
