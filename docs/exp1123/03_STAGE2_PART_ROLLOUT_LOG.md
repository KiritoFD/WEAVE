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

- **`2a` (`P2_Rollout_Glob`) 50 步梯度校准实测**：
  - `calib_norms = [17.2547 (SWD), 18.0019 (Haar), 0.4745 (Struct), 0.5788 (Reg)]`
  - 自动校准权重：`weights = [1.0000, 0.2396, 21.8173, 11.9250]`
  - 4 步可微 Rollout + `S10-BF16` 解码 + VGG 反向传播峰值显存：**`6.90 GB`**（严格低于 `< 10.5 GB` 预算红线）。
- 当前 4090 正在依次执行：`P2_Rollout_Glob (2000 upd)` → `P2_Part_Glob (2000 upd)` → `P2_Part_Closed (2500 upd)` → `P3_1Shot`，每 500 步存盘全量评测。

