# V26：统一任意参考 PRISM — 调整版总纲（`docs/1120/`，2026-11-21 C2nt 降级修订）

> **基线锚点**：`main @ 632d94c`→V26 提交｜**主线（North Star）**：**让 PRISM-Flow 的强迁移能力获得可靠的任意参考条件**——而不是救活 C2nt。
> **C2nt 状态（2026-11-21 用户决策）**：**从主线降级为探索分支并冻结原训练配方**；保留为零训练诊断载体（SR-CFG / RefAttn 因果 / 题材偏置 / SG-25 OOD）与**单轮** SWD 受控试验对象。完整决策记录见 [`07_c2nt_demotion_decision.md`](07_c2nt_demotion_decision.md)。
> **工作目录**：`exp/v26/{gpu3060,gpu4090}/`、`tests/v26/`（严格隔离，**不修改 V25 已验证实现与权重**）
> **算力口径（72 GPU·h 双卡）**：**RTX 4090** = C2nt 零训练诊断（SR-CFG / K1 RefAttn 因果 / K2 题材偏置）+ Q32 审计；**RTX 3060** = 少步扫描 + SG-25 OOD + C2nt 最后一轮 SWD 受控试验（≤4–6h，含停机条件）+ 后续转向 Flow 少步化 / Q32 / 解码器；**4090 产物完结即拉回**；**所有正式延迟/吞吐口径一律以 RTX 3060 12GB 实测为准**。

---

## 文档索引

| 文档 | 核心内容 |
|---|---|
| [`01_diagnostics_and_srcfg.md`](01_diagnostics_and_srcfg.md) | **P0 零训练诊断与 SR-CFG**（H1/H2/H0 判别门槛与实验矩阵；实现于 `exp/v26/sr_cfg.py` 等） |
| [`02_swd_and_mechanisms.md`](02_swd_and_mechanisms.md) | **调整版：单轮 SWD 受控试验（S1fix-VGG vs S1fix-SWD）+ 冻结 C2nt 后的转向路线** |
| [`03_schedule_and_dual_gpu.md`](03_schedule_and_dual_gpu.md) | **双卡队列排程（已实现版）、4090 及时拉回红线、失败不断链路由表** |
| [`04_engineering_and_tests.md`](04_engineering_and_tests.md) | **`exp/v26/` 模块契约、`tests/v26/` 13 契约 / 19 测试验收** |
| [`05_validation_and_deliverables.md`](05_validation_and_deliverables.md) | **收束验证、三态判定准则、交付清单与终极问题（调整版）** |
| [`06_4090_diagnostic_queue.md`](06_4090_diagnostic_queue.md) | **V26-4090 独立诊断队列（K0 / SR-CFG / K1 / K2）**：Q32 审计、RefAttn 因果干预 R0–R6、SWD/题材偏置 |
| [`07_c2nt_demotion_decision.md`](07_c2nt_demotion_decision.md) | **C2nt 降级决策记录**：性能证据、目标冲突机理、保留/冻结/转向路线表 |

---

## 一、核心研究目标（调整后）

**不再把"救活 C2nt"当目标。** 保留 C2nt 的科学结论（免教师即可让模型真实响应参考：正确/错误参考 `DINO-S` 差 `+0.0732`），把它作为**探索分支的诊断探针**；研究主线转为：

> **如何把 PRISM-Flow（`60.50%` Top-1 / `0.5672` DINO-S / `0.6584` Affinity，超越 StyleGallery）的闭集强迁移，转化为可靠的任意参考能力。**

| 路线 | 决策（2026-11-21） |
|---|---|
| **PRISM-1 / PRISM-HC / PRISM-Flow** | **核心模型**，保留已建立的 Pareto 优势 |
| **C2nt 原始训练配方** | **冻结**，不再长时间直接续训（仅保留 ≤4–6h 收尾预算） |
| Source-Repulsive CFG | **保留**零训练验证（4090，`SR0–SR7`） |
| RefAttn 门控与 Value 消融 | **保留**，查明参考瓶颈（4090 `K1`，`R0–R6`） |
| 动态 SWD | **只保留一轮受控试验**（3060，`S1fix-VGG vs S1fix-SWD`，1200 updates） |
| C2nt 多尺度参考 / 完整 Rollout / Reflow | **暂缓**，不同时投入 |
| 统一任意参考 Flow | **长期目标**（North Star） |
| Q32 消融 / 解码器研究 / Flow 少步化 | C2nt 判停后的 GPU 余量去向 |

## 二、五大研究假设（信息增益排序，调整版）

1. **H1 — Source-Repulsive CFG**：源参考条件速度场差分能否抵消源风格偏置（零训练，4090；同 NFE 公平对照）。
2. **H2 — Reference Attention Bottleneck**：`RefAttnV24` 门控/Value 是否真被利用（4090 `K1` 因果干预）→ 直接指导 PRISM-Flow 任意参考化的条件通路设计。
3. **H3 — Distributional Style Supervision（单轮试验）**：动态多尺度 SWD 是否补足 VGG 均值损失的纹理盲区（3060 一次性受控试验，硬停机条件）。
4. **H0 — Q32 必要性审计**：`Q32` 谱/泄漏/结构保持二维权衡（4090 `K0`）→ 决定是否进入匹配训练消融。
5. **H5（新增）— PRISM-Flow 任意参考化路径**：综合 `K1`/`K2`/SR-CFG 证据，设计把闭集 Flow 的强迁移转化为任意参考的最小机制（后续立项）。

## 三、三项必须严防的方法论误区（不变）

1. `CSD Top-1` 不是风格迁移唯一正确性定义（`K2-C` 题材偏置审计直接量化此混杂）。
2. 不能把 V25 `C2nt` 宣称为"完全不使用风格标签训练"（推理免标签，训练参考采样用风格元数据）。
3. 不能将多模型 Pareto 拼接成单模型能力（最终报告须展示同一 checkpoint 随同一 strength 参数的真实曲线）。
