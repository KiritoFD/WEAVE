# PRISM-HighContent-R12 (B4 锚 + D6 cs_j + PRISM-1 高内容前沿突破候选)

> **定位**：极高内容 & 极低 LPIPS 前沿主力 (DINO-C 0.73-0.775)｜**状态**：`active (promoted candidate)`
> **图像目录**：`images_750/` (750 PNGs, 源 `MechA_B4anc_A0_global_g55`), `images_750_g70/` (750 PNGs, 源 `HC_R12_g70`), `images_750_g85/` (750 PNGs, 源 `HC_R12_g85`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `checkpoints/P1_B4_distill50.pt` | `runs/v6/P1_B4_distill50/P1_B4_distill50.pt` | 0.87 MB | `2716436fe0ee9993...` |
| `checkpoints/s2_prism1_v2.pt` | `runs/v19/students/s2_prism1_v2.pt` | 4.91 MB | `a15f5928fc70a664...` |
| `checkpoints/slf_core_v2.pt` | `runs/v20/slf_flow/slf_core_v2.pt` | 83.76 MB | `85737c5331895032...` |

## 2. 架构与训练配方

- **架构**：线性混合 (1-gamma)*z_P1 + gamma*z_D6(B4-anchored SLFUNet cs_j D6H); g100 为单模型免 P1
- **训练配置**：
```json
{
  "script": "exp/v25/r12_anchor_reeval.py + exp/v25/r13_highcontent_r12.py",
  "note": "零新权重重训; 组合 P1_B4_distill50 (218K C3UNet 锚) + slf_core_v2 (21.95M) + s2_prism1_v2 (1.28M)",
  "eval_seed": 20261185
}
```

## 3. 推理配置

```json
{
  "d6_anchor": "B4_distill50(canonicalizer(z_src, mode='c1_gentle', augment=False))",
  "d6_branches": "cs_j (20 NFE = 8 RK2 steps x2 CFG)",
  "d6_solver": "D6H_SOLVER (tau_min=0.48, tau_hi=0.78, steps=8, ws_fig=4.4, rescale_phi=0.48)",
  "gate": "original sigmoid((dsd - mid)*550)",
  "p1_lambda": 1.0,
  "mix_gamma": "0.55 (0.7723/0.4564/0.4783) | 0.70 (0.7622/0.4669/0.4799) | 0.85 (0.7470/0.4774/0.4863) | 1.00 single-model (0.7286/0.4870/0.4980)",
  "decoder": "S10"
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `HC_R12_g40` | 0.7748 | 0.4467 | 0.4807 | 0.5688 | 41.67% | 96.7% |
| `MechA_B4anc_A0_global_g55` | 0.7723 | 0.4564 | 0.4783 | 0.5752 | 42.50% | 96.7% |
| `HC_R12_g70` | 0.7622 | 0.4669 | 0.4799 | 0.5820 | 43.33% | 96.7% |
| `HC_R12_g85` | 0.7470 | 0.4774 | 0.4863 | 0.5885 | 43.33% | 96.7% |
| `HC_R12_g100` | 0.7286 | 0.4870 | 0.4980 | 0.5946 | 42.00% | 96.7% |

## 5. 说明

V25 最重要正向发现 (R13 严格三因子解耦与配对 CI 确认): g55 四项严格支配 PRISM-1 l075 (+0.0193 D-S, -0.0065 LPIPS, +4.83pp Top-1); g70/g85 在内容/风格/LPIPS 三维严格支配旧 Hybrid g25/g40/g55 (代价 CSD Top-1 -4~-9pp); g100 单模型直达 DINO-C=0.7286 打破 T4 下 0.7055 天花板; 4090 RGB->RGB 82.9 ms/img (b16)。
