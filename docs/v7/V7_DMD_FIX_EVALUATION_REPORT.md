# WEAVE V7 修正 DMD 符号后 750 全量评测报告

> 修复：DMD target = y - alpha*(eps_real-eps_fake)/sigma；
tanh 平滑残差；训练/推理完全统一；删除无效 Identity Loss。

## 600 跨风格主表

| 模型 | NFE | CSD ΔT↑ | Target Top-1↑ | Target Margin↑ | DINO-C↑ | LPIPS↓ | 5类改善 | Ukiyo Top-1 | Ukiyo ΔT |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| **V7_Fix_dmd_content** | **1** | **+0.0145** | **1.8%** | **-0.4053** | 0.9575 | 0.1660 | 3/5 | 1.7% | +0.0226 |
| **V7_Fix_pure_dmd** | **1** | **-0.0177** | **4.0%** | **-0.2642** | 0.8297 | 0.3259 | 1/5 | 0.0% | -0.0306 |

## 150 同风格身份保持

| 模型 | CSD ΔS | DINO-C | LPIPS | 潜空间位移 L2 |
|---|---:|---:|---:|---:|
| **V7_Fix_dmd_content** | +0.0262 | 0.9803 | 0.0745 | 0.0000 |
| **V7_Fix_pure_dmd** | +0.0262 | 0.9803 | 0.0745 | 0.0000 |

## 结论

报告仅陈述修正实现后的实测结果；是否替代 Calib_Balanced_s16 仍需按 Top-1/ΔT/DINO-C/Ukiyo-e 四项共同判断。
