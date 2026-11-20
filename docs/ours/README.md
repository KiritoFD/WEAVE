# WEAVE / PRISM 核心模型资产库（`docs/ours/`）

> **整理日期**：2026-11-20｜**解码协议**：统一 `S10` (`10.66M`) × `test750` 四裁判（`cross600` 读数）
> 每个子目录独立包含：**checkpoint (`.pt`)**、**750 张评测输出图 (`images_750/`)**、**`config.json`** 与 **`RECIPE.md` 配方说明**。（注：`.pt` 与 `.png` 受仓库 `.gitignore` 保护不上库；配置与配方 Markdown 受 Git 跟踪与链接守卫校验。）

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
| [`09_V24_O3_OneShot`](09_V24_O3_OneShot/RECIPE.md) | active | `O3_rgb_cnx_gated_t45` | 0.5845 | 0.5245 | 0.4213 | 35.33% | 750 PNGs |

## 推荐使用族（V25 收束定案）

1. **极速高内容基座**：[`01_PRISM1_v2/`](01_PRISM1_v2/RECIPE.md)（1.28M，单步 `17.0 ms` 协议批 / `70.7 ms` 完整 RGB→RGB）
2. **极高内容前沿主选**：[`02_PRISM_HighContent_R12/`](02_PRISM_HighContent_R12/RECIPE.md)（`B4 + D6H cs_j + PRISM-1`，`DINO-C 0.7286–0.7748`, `LPIPS 0.478–0.498`，4090 `batch=16` `82.9 ms/img`）
3. **高 CSD Top-1 Hybrid / 强风格对照**：[`03_V20_SLFUNet_D6/`](03_V20_SLFUNet_D6/RECIPE.md)、[`04_V22_Peak1500/`](04_V22_Peak1500/RECIPE.md)（76.83%）、[`07_V23_HiC_Joint_Pk/`](07_V23_HiC_Joint_Pk/RECIPE.md)（58.83%）
4. **One-Shot 任意参考引擎**：[`09_V24_O3_OneShot/`](09_V24_O3_OneShot/RECIPE.md)（免类别标签，SG-25 `ΔS=+0.157`）
