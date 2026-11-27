# 主线实验总规划（2026-11-23）：无锚点 4 步 Q32-Reflow × Part-Bank Spatial AdaGN × 部件级 Rollout 闭环

> **定位**：ICME 2027 论文终稿（`main@79f47a9`）已冻结提交；所有新实验统一落在独立目录 `exp/v27_mainline/` 与 `runs/v27_mainline/`，零污染投稿代码。本文档记录闭环方案设计、硬门控判据与实时实验进度。

---

## 一、核心诊断与算法闭环

1. **延迟拆解真相（RTX 3060 12GB, `b=16`）**：
   - `Q32 6k` 旗舰 `6 步 cs_j`：`enc (71.4 ms) + dino (57.9 ms) + flow (98.6 ms) + dec (74.2 ms) = 302.1 ms`，但实测全流程 E2E 为 `539.7 ms`（`e2full` 为 `571.9 ms`）——**中间 `~237.6 ms` 全被外部 `z_t4` 锚点（2 步 S10-BF16 反传）吞掉**。
2. **为什么先做方向 1，再做方向 2 + 方案 A**：
   - 在 `6 步 RK2 + e2full`（20–24 次前向）上跑全可微 Rollout 显存与耗时过高；
   - 先由 **阶段 1（`P1_Reflow`）** 拔掉 `z_t4` 拐杖并把 CFG 蒸馏进**单模型、单分支、4 步 Euler**（仅 4 次 U-Net 前向，`<= 245 ms` E2E @ 3060）；
   - 再在阶段 2/3 将共享同一套 `Q32` 通用语义部件词典（$M=32$）的 **推理端 `Part-Bank Spatial AdaGN`** 与 **训练端 `Part-Bary-SWD`** 闭环接入 4 步无截断可微 Rollout（“识别内容 $\to$ 精准施加对应部件风格”），零触碰 `mid16` 商流形（严格遵守 V26 Exp 1 双向瓶颈纯净原则）。

---

## 二、受控阶梯实验矩阵与硬门控

| 阶段 | 实验臂 | 核心配方 | 硬门控 / 验证问题 | 状态 |
|---|---|---|---|---|
| **0** | `P1_A0` + `Audit_Part` | 修复 `Q32_wrong` 锚点替换；零训练替 `z_t4`（`A_p1/A_b4/A_chroma`）；$M=32$ `Q32` 聚类画派熵审计 | `Q32_wrong <= 3%`；部件词典平均归一化熵 `>= 0.85` 且单画派占比 `<= 55%` | **已完成 `P1_A0`，`Audit_Part` 运行中** |
| **1** | **`P1_Reflow`** | 解析桥起点 $\tilde{z}_0 = 0.70\epsilon + 0.30 z_0^\perp$；`SLFUNetGamma`（零初始化 $\gamma$ 列）；Pseudo-Huber + 0.35 Haar-HF + 动能正则，3,500 upd | 4 步 Euler 单分支：**Top-1 >= 56.0%**, **Imp→Min >= 88%**, **Same150 >= 95%**, **错标签 <= 3.0%**, 3060 E2E **<= 245 ms** | 待跑（Stage 0 后立刻启动） |
| **2a** | `P2_Rollout_Glob` | `P1_Reflow` + 4 步可微 Rollout + 全局 3 层 VGG-SWD + Haar 能量比 + 灰度自相似 + P1-Reg | 测量纯 4 步 Rollout + 全局监督的独立增益上限 | 待跑 |
| **2b** | `P2_Part_Glob` | `P1_Reflow` + `Part-Bank Spatial AdaGN` + 同 `2a` 全局损失 | 单变量隔离：部件银行架构本身是否带来增益？ | 待跑 |
| **2c** | **`P2_Part_Closed`** | `P1_Reflow` + `Part-Bank Spatial AdaGN` + 全局 SWD + **`Part-Bary-SWD`** + Haar + 灰度自相似 | **DINO-C >= 0.65, DINO-S >= 0.57, Top-1 >= 62%, LPIPS <= 0.53**，部件间纹理分化度显著高于 `2a`，并补第 2 种子配对 CI | 待跑 |
| **3** | `P2_1Shot_Bonus` | 直接复用 `2c` 权重零训练调用 `lookup_exemplar` 测 SG-5 / Holdout 单图 | 检验经验贝叶斯部件银行零训练 1-Shot 外推能力 | 待跑 |
