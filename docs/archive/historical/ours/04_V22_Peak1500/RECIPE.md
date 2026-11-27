# V22 Peak1500 (T4S snap1500)

> **定位**：强风格 Pareto 冠军档 (DINO-S 0.5781, Top-1 76.83%)｜**状态**：`frozen`
> **图像目录**：`images_750/` (750 PNGs, 源 `T4Pg45_wc10`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `slf_T4Ssnap1500.pt` | `runs/v22/tmatrix/slf_T4Ssnap1500.pt` | 83.76 MB | `ef25c10bcb00310e...` |

## 2. 架构与训练配方

- **架构**：SLFUNet(base_dim=144, u_dim=128), 21.95M
- **训练配置**：
```json
{
  "script": "exp/v22/tmatrix.py (--arm T4S --updates 3600, peak at snap 1500)",
  "init": "runs/v20/slf_flow/slf_final.pt",
  "core_frac": 0.8,
  "drop_c_prob": 0.3,
  "layout_train": false,
  "struct_w": 0.1,
  "lr": "1.3e-4 -> 1.5e-5",
  "batch": 120,
  "ema": 0.996,
  "seed": 20261205
}
```

## 3. 推理配置

```json
{
  "anchor": "T4",
  "branches": "full",
  "gate": "fixed_g=0.45 (T4Pg45_wc10) / fixed_g=0.50 (T4Pk_wc15)",
  "solver": "D5_SOLVER variant (wc_fig=1.0 / 1.5, tau_hi=0.66)",
  "decoder": "S10"
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `T4Pg45_wc10` | 0.4779 | 0.5781 | 0.6268 | 0.6671 | 76.83% | -% |
| `T4Pk_wc15` | 0.5319 | 0.5610 | 0.6049 | 0.6530 | 69.00% | -% |

## 5. 说明

Our-Frontier 强风格端点代表点: 76.83% Top-1 / 0.5781 DINO-S (DINO-C=0.4779)。
