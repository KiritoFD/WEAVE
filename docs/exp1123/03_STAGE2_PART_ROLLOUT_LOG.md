# 阶段 2 & 3 记录：4 步可微 Rollout × `Part-Bank Spatial AdaGN` × `Part-Bary-SWD` 闭环

> **执行环境**：RTX 4090 24GB（`cu121`）｜ **脚本**：`exp/v27_mainline/rollout_refine.py`

---

## 一、受控消融阶梯设计（禁止跳过 `2a` 直奔 `2c`）

| 阶段 | 实验臂 | 架构 | 监督损失 | 验证核心问题 |
|---|---|---|---|---|
| **2a** | `P2_Rollout_Glob` | `SLFUNetGamma`（无部件银行） | 全局 3 层 VGG-SWD + Haar 能量比 + 灰度自相似 + P1-Reg | 纯 4 步可微 Rollout + 全局分布精修的独立收益上限 |
| **2b** | `P2_Part_Glob` | `SLFUNetPart`（`Part-Bank Spatial AdaGN`） | 同 `2a` 全局损失 | 单变量隔离：部件银行架构在仅全局损失下是否有增量？ |
| **2c** | **`P2_Part_Closed`** | `SLFUNetPart`（`Part-Bank Spatial AdaGN`） | 全局 SWD + **`Part-Bary-SWD`** + Haar + 灰度自相似 + P1-Reg | 完整闭环：目标同时达成 `DINO-C >= 0.65`, `DINO-S >= 0.57`, `Top-1 >= 62%`, `LPIPS <= 0.53` 且部件纹理分化度 $R^2_{\text{part}}$ 显著提升 |
| **3** | `P3_1Shot_PartBank` | 零训练复用 `2c` 权重 | `lookup_exemplar(kappa=8.0)` | 经验贝叶斯部件银行在 SG-5 与 Holdout 单图下的零训练 1-Shot 外推 |

---

## 二、50 步梯度范数校准与分段解冻规范

- **冻结范围**：`dino_proj`（保护 $Q_{32}$ 商空间纯度）、`in_conv`、`enc64`、`down32`、`enc32`。
- **分段解冻**：`1–500` 步仅训 `part_proj + sp_film1/2`（`lr = 2e-4`）；`501–2500` 步联合微调 `mid + dec32/dec64`（`lr = 2.5e-5`）与部件分支（`lr = 1e-4`），`EMA = 0.995`，每 `500` 步存盘全量评测。
- **2c 初始梯度范数锁定**：`||∇Global-SWD|| : ||∇Part-Bary-SWD|| : ||∇Haar|| : ||∇Struct|| : ||∇Reg|| = 0.70 : 0.60 : 0.25 : 0.60 : 0.35`。

---

## 三、实时执行日志（`4090`，`micro=3 × accum=6 = 18`，BF16 + `checkpoint`）

### 1. `v1` 探针诊断（`P2_Rollout_Glob` `s500/s1000`）
- `50` 步梯度校准：`calib_norms = [17.2547, 18.0019, 0.4745, 0.5788]`，权重 `[1.0000, 0.2396, 21.8173, 11.9250]`，峰值显存 **`6.90 GB`**。
- **`s500`**：仅 500 步全可微 Rollout 即把 `Top-1` 从 `D1v2` 的 `53.17%` 推至 **`55.83%`**、`DINO-C` 从 `0.5666` 推至 **`0.6038`**、`Imp→Min` 冲上 **`100.0%`**（`ER: 9.2%→19.2%`, `Uki: 1.7%→5.8%`, `R^2_part = 0.2884`）。
- **`s1000` 过平滑回落诊断**（见视觉对比图 [qual_s2a.png](data/qual_s2a.png)）：`Top-1` 回落至 `53.67%`、`DINO-S` 降至 `0.5111`。直读 `qual_s2a.png` 发现两个根因：
  1. 仅用 `relu1_2..relu3_3` 全局 SWD 缺少深层语义纹理与颜色分布锚定（未含 `relu4_3`、深层 Gram/一二阶矩及 Oklab 27D），使 `s500/s1000` 整体画风变粉/偏素、缺少 `Impressionism/Rococo` 色彩饱和与深层笔触；
  2. `Struct:Reg = 0.60:0.40` 过高（`w_struct=21.82`），强压过久把高频笔触磨光。

### 2. `v2` 修正重跑（`P2_Rollout_Glob → P2_Part_Glob → P2_Part_Closed → P3_1Shot`）
在保留 4 步可微 Rollout 与裁判 100% 隔离的前提下做两项修正：
1. **全局风格监督补齐深层语义与色盘**：扩展 `FrozenVGG16Rollout` 至 `relu4_3`，在 `l_glob` 中加入 `0.35 * vgg_moment_and_gram_loss(relu3_3, relu4_3) + 4.0 * Oklab27_smooth_l1`；
2. **调正梯度的结构/风格比**：将目标梯度比从 `[1.0, 0.25, 0.60, 0.40]` 调整为 `[1.00, 0.20, 0.30, 0.20]`（`2c` 为 `[0.85, 0.55, 0.20, 0.30, 0.20]`）。
- **`v2` 梯度校准实测**：`calib_norms = [25.007, 18.0018, 0.4745, 0.6636]` → `weights = [1.0, 0.2778, 15.809, 7.5371]`，峰值显存 **`7.09 GB`**。当前 4090 正在连续跑 `2a (1500 upd) → 2b (1500 upd) → 2c (2000 upd) → 3`。



