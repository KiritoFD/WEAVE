# V23 DualSWD Pure (slf_DualSWD_pure.pt)

> **定位**：均衡高内容双分支 (DINO-C 0.6164, DINO-S 0.5467)｜**状态**：`frozen`
> **图像目录**：`images_750/` (750 PNGs, 源 `DualSWD_pure_hiT_ws52`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `slf_DualSWD_pure.pt` | `runs/v23/dual_swd/slf_DualSWD_pure.pt` | 83.76 MB | `8437d7c6fc20985a...` |

## 2. 架构与训练配方

- **架构**：SLFUNet(base_dim=144, u_dim=128), 21.95M
- **训练配置**：
```json
{
  "script": "exp/v23/p1_dual_swd_train.py (pure cs_j + ConvNeXt/RGB Patch-SWD + mid-scale content loss)"
}
```

## 3. 推理配置

```json
{
  "anchor": "T4",
  "branches": "cs_j",
  "gate": "original",
  "solver": "hiT (tau_min=0.32, tau_hi=0.68, ws_fig=5.2)",
  "decoder": "S10"
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `DualSWD_pure_hiT_ws52` | 0.6164 | 0.5467 | 0.5674 | 0.6384 | 51.17% | 96.7% |
| `B_dswd_t058` | 0.6770 | 0.5156 | 0.5351 | 0.6153 | 51.83% | 96.7% |

## 5. 说明

视觉审计通过的超 StyleGallery 内容工作点 (0.6164 vs SG 0.6113); V25 R1 B_dswd_t058 为高内容备选。
