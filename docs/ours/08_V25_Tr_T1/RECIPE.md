# V25 Round 2 Winner Tr_T1 (slf_Tr_T1.pt)

> **定位**：V25 最简训练配方胜出者 (drop_c=30%, AB=0%, 两独立种子复验通过)｜**状态**：`active`
> **图像目录**：`images_750/` (750 PNGs, 源 `V25_Tr_T1_d6hcsj`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `slf_Tr_T1.pt` | `runs/v25/r2_tmatrix/slf_Tr_T1.pt` | 83.76 MB | `da883c8a6060c162...` |

## 2. 架构与训练配方

- **架构**：SLFUNet(base_dim=144, u_dim=128), 21.95M
- **训练配置**：
```json
{
  "script": "exp/v25/r2_train_ablation.py (--arms Tr_T1 --updates 1500)",
  "init": "runs/v20/slf_flow/slf_final.pt",
  "drop_c_prob": 0.3,
  "ab_prob": 0.0,
  "core_frac": 0.8,
  "layout": 1.0,
  "struct_w": 0.1,
  "lr": "1.2e-4 -> 1.5e-5",
  "batch": 120,
  "ema": 0.996,
  "seed": 20263500,
  "seed2_verified": 20263501
}
```

## 3. 推理配置

```json
{
  "anchor": "T4",
  "branches": "cs_j",
  "gate": "original",
  "solver": "D6H_SOLVER (tau in [0.48,0.78])",
  "decoder": "S10"
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `V25_Tr_T1_d6hcsj` | 0.6613 | 0.5162 | 0.5411 | 0.6168 | 56.67% | 96.7% |
| `V25_Tr_T1_d5csj` | 0.6086 | 0.5371 | 0.5668 | 0.6333 | 57.17% | 96.7% |
| `V25_Tr_T1_s2_d6hcsj` | 0.6613 | 0.5149 | 0.5415 | 0.6161 | 56.17% | 96.7% |

## 5. 说明

V25 Round 2 配对 CI 验证: Tr_T1@D6h 通过对 T0 非劣且 Top-1 +1.00pp (CI 不含零), 且 AB 不产生额外增益 -> 冻结为最简训练配方。
