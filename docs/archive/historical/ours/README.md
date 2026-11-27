# WEAVE / PRISM 核心模型资产库（`docs/ours/`）

> **状态复核**：2026-11-21｜**解码协议**：统一 `S10` (`10.66M`) × `test750` 四裁判（`cross600` 读数）
> 每个子目录以 **`RECIPE.md` + `config.json`** 固化模型权重来源、推理协议、锚点和评测工作点。大体积 **checkpoint（`.pt`）和生成图（`.png`）不随 Git 克隆自动获得**：表中“750 PNGs”表示实验归档数量，不保证仓库内可直接下载。

## 模型一览

| 目录 | 状态 | 代表工作点 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Top-1 ↑ | 750 图集 |
|---|---|---|---:|---:|---:|---:|---|
| [`01_PRISM1_v2`](01_PRISM1_v2/RECIPE.md) | active | `M0_prism1_l100` | 0.7440 | 0.4430 | 0.5057 | 44.33% | 750 PNGs |
| [`02_PRISM_HighContent_R12`](02_PRISM_HighContent_R12/RECIPE.md) | active (promoted candidate) | `MechA_B4anc_A0_global_g55` | 0.7723 | 0.4564 | 0.4783 | 42.50% | 2250 PNGs |
| [`03_V20_SLFUNet_D6`](03_V20_SLFUNet_D6/RECIPE.md) | active | `M0_v20_d6` | 0.6688 | 0.5107 | 0.5374 | 55.67% | 750 PNGs |
| [`04_V22_Peak1500`](04_V22_Peak1500/RECIPE.md) | frozen | `T4Pg45_wc10` | 0.4779 | 0.5781 | 0.6268 | 76.83% | 750 PNGs |
| [`05_V22_DualBranch_F3`](05_V22_DualBranch_F3/RECIPE.md) | frozen (also O3 initialization base) | `F3_dual_fixed_orig` | 0.6096 | 0.5522 | 0.5723 | 49.50% | 750 PNGs |
| [`06_V23_DualSWD_Pure`](06_V23_DualSWD_Pure/RECIPE.md) | frozen | `DualSWD_pure_hiT_ws52` | 0.6164 | 0.5467 | 0.5674 | 51.17% | 750 PNGs |
| [`07_V23_HiC_Joint_Pk`](07_V23_HiC_Joint_Pk/RECIPE.md) | frozen | `HiC_JPk_csj65_t4` | 0.6299 | 0.5330 | 0.5649 | 58.83% | 750 PNGs |
| [`08_V25_Tr_T1`](08_V25_Tr_T1/RECIPE.md) | active | `V25_Tr_T1_d6hcsj` | 0.6613 | 0.5162 | 0.5411 | 56.67% | 750 PNGs |
| [`09_V24_O3_OneShot`](09_V24_O3_OneShot/RECIPE.md) | 1-Shot 实验基线 | `O3_rgb_cnx_gated_t45` | 0.5845 | 0.5245 | 0.4213 | 35.33% | 750 PNGs |
| [`10_V25_C2nt_OneShot`](10_V25_C2nt_OneShot/RECIPE.md) | **原训练配方冻结；仅诊断** | `C2nt_sw6_lc0.8_t55` | 0.6647 | 0.5027 | 0.3992 | 28.33% | 750 PNGs |

## 按研究任务选择（V26 状态复核）

1. **极速高内容基座**：[`01_PRISM1_v2/`](01_PRISM1_v2/RECIPE.md)（1.28M，单步 `17.0 ms` 协议批 / `71.1 ms` 完整 RGB→RGB）
2. **极高内容前沿主选**：[`02_PRISM_HighContent_R12/`](02_PRISM_HighContent_R12/RECIPE.md)（`B4 + D6H cs_j + PRISM-1`，`DINO-C 0.7286–0.7748`, `LPIPS 0.478–0.498`，4090 `batch=16` `84.0 ms/img`）
3. **高 CSD Top-1 Hybrid / 强风格对照**：[`03_V20_SLFUNet_D6/`](03_V20_SLFUNet_D6/RECIPE.md)、[`04_V22_Peak1500/`](04_V22_Peak1500/RECIPE.md)（76.83%）、[`07_V23_HiC_Joint_Pk/`](07_V23_HiC_Joint_Pk/RECIPE.md)（58.83%）
4. **One-Shot 研究基线（不是当前生产主线）**：[`09_V24_O3_OneShot/`](09_V24_O3_OneShot/RECIPE.md)（V23 教师基线）与 **[`10_V25_C2nt_OneShot/`](10_V25_C2nt_OneShot/RECIPE.md)**（零教师突破：`t45 42.17% / 0.5235`，同内容参考特异性 `+0.0732`）

## V26 后模型状态与比较边界

- **C2nt** 的 t45 强风格工作点为 **Top-1 42.17% / DINO-S 0.5235 / DINO-C 0.5448**；表中 t55 是高内容档 **28.33% / 0.5027 / 0.6647**，不能只读 t55 推断全部能力。原训练配方已冻结，仅保留参考响应与 [K1/Exp1 诊断](../v26/LOCALITY_UOT_FIBER_BUNDLE.md)。[V26 裁决](../1120/V26_RESULTS_SUMMARY.md) · [降级决定](../1120/07_c2nt_demotion_decision.md)。
- **闭集原型 ≠ 任意参考 1-Shot**：PRISM-Flow/D6/R12 的优秀工作点并不自动代表任意参考超过 StyleGallery。部署时按 [根 README](../../README.md) 中的协议口径选择配置。
- **权重身份**：新的锚点、CFG 或强度档不一定意味着新 checkpoint；参阅 [V25 模型血缘](../v25/MODEL_LINEAGE.md)。
