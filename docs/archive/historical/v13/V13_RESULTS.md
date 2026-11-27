# V13 结果：LocalH2-GCL —— P1 Sim 门槛判负，路线停止

> 预注册 [WEAVE_V13_LOCALH2_GCL_PLAN.md](WEAVE_V13_LOCALH2_GCL_PLAN.md) · 2026-11-14
> 状态：**P1 门槛未通过 → 按预注册判停，不训 Adapter。**

## TL;DR

对比学习 E_s 编码器（内容分层对比 + VICReg）在 H2 DWT token 上训练 4000 步后，
画作级 Sim 门槛 **未通过**：painting_acc 0.425（< 0.5），correct-vs-wrong margin
**−0.003**（反向）——**同画派的 H2 token 嵌入不比内容匹配的跨画派 token 更接近。**

这意味着：DWT 域的局部笔触 token（9 维方向 + 3 维低频 + 1 维边缘）经对比学习后
仍无法有效分离五个画派——**H2 系数域的风格信号极弱或不存在**（与 V10.2 手工特征、
V11 密度方向形成三重闭合）。

## P1 数字（holdout 200 幅，painting 级）

| 指标 | 值 | 门槛 | 判定 |
|---|---:|---|---|
| painting_acc | **0.425** | ≥0.5 | ❌ |
| correct_sim | 0.9775 | — | |
| wrong_sim | 0.9803 | — | |
| **margin** | **−0.0028** | >0 | ❌ 反向 |
| Early_Renaissance | 0.474 | | |
| Impressionism | 0.375 | | |
| Minimalism | **0.610** | | 最强（低纹理画派反而最可分）|
| Rococo | 0.233 | | 最弱 |
| Ukiyo_e | 0.447 | | |

## 训练细节

- E_s：3 层 Conv（13→64→64→64）+ 1×1 head，D=64；VICReg 变差/协方差正则
- 输入：H2_ds(9) + LL2_ds(3) + Sobel(1) = 13ch 32×32 token
- 对比：同画派跨画作正样本 + **内容匹配跨画派硬负样本**（13 维原始 token
  L2 距离选最近邻）；VICReg 权重 0.5/0.05
- 4000 步 × 512 token/步（16 画 × 32 token）；loss 9.4→9.3（VICReg 主导，
  对比分量持平——**没有学到有效分离**）

## 分析

1. **DWT 域的画派信号不在逐 token 统计里**：H2 的 9 个方向子带 token 是
   局部频率/方向统计，不编码"笔触如何组织"这种空间关系。V10 的手工
   特征已经证明统计不够；对比学习在同样的 token 表示上也无法提取更多。
2. **Rococo 最差（0.233 ≈ 随机）**：Rococo 的笔触特征（装饰性细节、复杂层次）
   在 DWT 域可能与 Impressionism 混淆——纹理域的画派边界不清。
3. **Minimalism 最好（0.61）**：恰好是最"平坦"的画派——低纹理本身就是一个
   强判别信号（但这是"缺乏笔触"而非"有笔触"的信号）。

## 三线实验的全景结论（v10.2 → v11 → v12 → v13）

| 阶段 | 尝试 | 结论 |
|---|---|---|
| V10.2 | 手工 latent 统计 Score | 恢复对齐全通用，无画派特异性 |
| V11 P0 | 四表示审计 | **DINO patch token 是唯一有画派边际的表示**（+0.658）|
| V11 P2-A | DINO 空间 KDE Score | 特异性增益 = 0（密度≠判别方向）|
| V12-GCL | 内容 Query 闸门 ×5 | 全部不 binding（风格⊥内容 Query）|
| **V13 P1** | **DWT 域对比学习 E_s** | **DWT token 无画派可分性（margin 反向）**|

**总结论**：在 WEAVE 的 4×64×64 latent / DWT 域内，无论手工特征、密度模型、
还是对比学习，都无法提取出可靠的画派特异局部表示。**画派信息存在于
RGB/decoder 域的更大感受野中**（DINO 的 16×16 patch = 224px 原图），而非
latent 的 8×8 patch（= 64px 原图）。

**方法论第四条**：`DWT Token ≠ Style Carrier` —— DWT 系数的逐 patch 统计
（无论手工还是学习）都不携带足够的画派信号。

## 对后续的决策含义

1. **关闭 DWT 域风格路线**（V10/V11/V12/V13 全链关闭）。
2. **如果继续风格研究**，只有两条有证据支撑的路线：
   - **decoded-RGB 域**：DINO patch token（V11 P0 唯一通过的表示）+
     判别式梯度（非密度）——但需要 through-S10 反传（成本高）；
   - **放弃 latent-only 约束**：允许 WEAVE 输出加一个轻量 RGB 残差分支
     （v11 文档 §五 预留的路线），在 RGB 域做画派特异性编辑。
3. **LD4 / E4 仍为当前可交付最优**。风格研究需要全新思路，不是当前管线的增量改进。

## 产物

- `exp/v13/{stroke_encoder, memory, run_p1}.py` + `sample_fixed.py, train_fixed.py`（8 bug 修复）
- `runs/v13/{stroke_encoder/es_encoder.pt, stroke_memory.pt, p1_gate.json}`
- 远程 `I:/Latent_Style/WEAVE/v13_run/` + `runs/v13_remote/`（LD4 重建）
- 训练基础设施：`exp/v13/rebuild_remote.py`（3060 远程全链重建，148MB→0 上传）
