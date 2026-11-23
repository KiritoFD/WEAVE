# 2026-11-23 主线实验文档索引 (`docs/exp1123`)

本目录记录 ICME 2027 终稿冻结（`main@79f47a9`）之后，在独立工程包 `exp/v27_mainline/` 下开展的闭环主线实验：

- [00_MASTER_PLAN.md](00_MASTER_PLAN.md)：总体路线（阶段 0 → 阶段 1 `P1_Reflow` → 阶段 2a/2b/2c `Part-Bank` 阶梯 → 阶段 3 `1-Shot` 外推）、硬门控与防踩坑规范。
- [01_STAGE0_LOG.md](01_STAGE0_LOG.md)：阶段 0（6k `Q32` 错标签修复、`P1_A0` 零训练锚点替换、`Audit_Part` $M=32$ 部件熵审计与 102D 部件银行构建）。
- `02_STAGE1_REFLOW_LOG.md`：阶段 1（无锚点 4 步单分支 `Q32-Reflow` 训练、$\gamma\in[0,1]$ 滑块曲线与 RTX 3060 实机 E2E 测速，完成后更新）。
- `03_STAGE2_PART_ROLLOUT_LOG.md`：阶段 2a/2b/2c（4 步可微 Rollout + `Part-Bank Spatial AdaGN` + `Part-Bary-SWD` 消融阶梯、部件纹理分化度、第 2 种子配对 CI 与阶段 3 `1-Shot` 外推，完成后更新）。
- `data/`：各阶段拉回本地的原始 JSON 测量结果与图表。
