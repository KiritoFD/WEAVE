# V23 HiC Joint-Pk (slf_Joint_Pk_C1AB.pt)

> **定位**：均衡高内容 Top-1 冠军 (20 NFE, Top-1 58.83% @ DINO-C 0.6299)｜**状态**：`frozen`
> **图像目录**：`images_750/` (750 PNGs, 源 `HiC_JPk_csj65_t4`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `slf_Joint_Pk_C1AB.pt` | `runs/v23/joint_c1ab/slf_Joint_Pk_C1AB.pt` | 83.76 MB | `d10ae418004ec0f4...` |

## 2. 架构与训练配方

- **架构**：SLFUNet(base_dim=144, u_dim=128), 21.95M
- **训练配置**：
```json
{
  "script": "exp/v23/p1_joint_c1_ab.py (ConvNeXt C1 mid-freq content + Anchor-Bridge AB)"
}
```

## 3. 推理配置

```json
{
  "anchor": "T4",
  "branches": "cs_j (20 NFE)",
  "gate": "original",
  "solver": "csj65 (tau_hi=0.65)",
  "decoder": "S10"
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `HiC_JPk_csj65_t4` | 0.6299 | 0.5330 | 0.5649 | 0.6301 | 58.83% | 96.7% |

## 5. 说明

Our-Frontier 中段代表点: 同 DINO-C 下击败 V20 D5 (+3.7pp) 且 NFE 减半。
