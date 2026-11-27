# WEAVE V14：风格表示与梯度可执行性诊断

**预注册 · 2026-11-14 · RTX 4070 Laptop · 0 新训练参数**
**用户修正（v13 post-mortem）**：V13 的"DWT Token ≠ Style Carrier"结论过强——DWT 可逆
所以信息存在，失败在于**局部 patch 提取 + 当前训练方案**，不在表示本身。同时
"感受野不足"未隔离验证，DINO 的全局 self-attention 使 16×16 token 的感受野 ≠
16×16 像素。本轮用两个诊断实验回答三个假说（H1/H2/H3），不训练新生成器。

## 三个假说

| 假说 | 内容 | 已有证据 | 判定实验 |
|---|---|---|---|
| H1：局部上下文太小 | patch 级无法稳定辨认画派 | DINO 正边际 vs V13 E_s 未达标 | **实验一（尺度扫描）** |
| H2：DWT/latent 编码器未学到风格表征 | 手工 + 对比学习均失败 | V10.2 + V13 | 实验一（RGB 对照） |
| H3：生成空间不能执行画派变化 | 风格梯度可能落在不可编辑的频带 | H2-only 收益、R1 权衡 | **实验二（频带分解）** |

## 实验一：风格信息的尺度扫描（诊断，不训练生成器）

对**同一套**真实画作（train/holdout 划分不变），比较以下表示的五画派判别能力：

| 编号 | 表示 | 输入 | 提取方式 |
|---|---|---|---|
| S0 | V10 手工 latent 统计 | 4×64×64 latent | f34（V10 基线）|
| S1 | DWT token 8×8 | latent 64×64 | V13 E_s 输入（9+3+1=13ch）|
| S2 | DWT token 大 crop | RGB 128×128 → DWT | 13ch 32×32 |
| S3 | RGB patch 64px | RGB 512→64×64 crop | CNN 编码 |
| S4 | RGB patch 128px | RGB 128×128 crop | CNN 编码 |
| S5 | RGB patch 224px | RGB 224×224 | CNN 编码 |
| S6 | **冻结 DINOv2-small** | 224×224 | patch tokens 16×16 |
| S7 | **冻结 CSD-ViT-L** | 224×224 | CLS 特征（参考上限）|
| S8 | 多尺度 RGB+DWT | 512 全图 + 128 crop | 全局+局部拼接 |

训练：线性探针（Logistic Regression / NCM）在 train split 上训练，
holdout split 上评测。每表示报告：5-way Top-1（画作级）、correct-vs-wrong
检索 margin（含 painting-clustered bootstrap CI）、**灰度化后的保留率**（颜色捷径检查）。

**判据**：
- H1 确认：同表示在大 crop 下 acc 显著上升（≥10pp）；或 RGB patch > DWT 同尺寸
- H2 确认：所有 DWT 变体 < RGB 变体（表示处理问题而非感受野问题）
- 上限参考：DINO/CSD 的准确率 = 可达水平的上界

## 实验二：风格梯度的频带可执行性

从已有 DINO/CSD 风格目标出发，对**解码 RGB** 求梯度，然后分解到 Haar 子空间：

```
x = S10(z)                          # 解码 RGB (3,512,512)
g_x = ∇_x L_style(x)               # DINO 或 CSD 风格损失对 RGB 的梯度
(LL2_g, H2_g, H1_g) = DWT(g_x)     # 梯度的频带分解
```

度量四个子空间：
1. **梯度能量占比**：`‖g_band‖² / ‖g_x‖²` per band
2. **可执行性**：对每个 band 单独施加更新（其余 band 固定为 z_src），经 S10 解码后
   测量 ΔDINO-S（风格改善）和 ΔLPIPS（内容损伤）
3. **画派特异性**：正确画派 vs 错误画派的梯度方向在子空间的余弦——
   如果正确/错误的 H2 分量高度相似，说明 H2 的风格信号弱

额外检查：`z_S10` 的梯度通过 S10 编码器反传（如果可达），与直接 RGB 梯度比较。

## 预注册结论判定

| 结果 | 含义 | 下一步 |
|---|---|---|
| 尺度扫描：大 crop >> 小 patch | H1 确认（感受野是主因）| P1 编码器改用大 crop 重训 |
| 尺度扫描：RGB >> DWT 同尺寸 | H2 确认（DWT 表示处理不行）| 表示改为 RGB 域 |
| 尺度扫描：DINO >> 所有自训 | 需要预训练表征（自训不够）| 蒸馏或冻结特征路线 |
| 频带分解：H2 占梯度能量 >50% | H3 部分否定（H2 可执行）| H2 Flow 路线继续 |
| 频带分解：LL2 占 >50% | H3 确认（H2-only 有结构上限）| 需要多频带/全带编辑 |
| 频带分解：正/错画派梯度无差异 | 风格信号不在梯度里 | 回到表示问题 |

## 12h 预算

| 阶段 | 预算 | 内容 |
|---|---|---|
| P0 | 0.5h | 图像/latent 加载 + DWT 基建验证 |
| P1 | 3h | 实验一：全表示尺度扫描 |
| P2 | 2h | 实验二：频带能量分解 + 可执行性 |
| P3 | 1h | 汇总 + H1/H2/H3 判定 |
| 剩余 | 5.5h | 结果分析 / 后续预注册 |

## 代码

```
exp/v14/
  __init__.py
  features.py       # 各表示提取器（统一接口）
  scale_scan.py     # 实验一
  band_energy.py    # 实验二
  bootstrap.py      # 配对统计
runs/v14/
  scale_scan.json
  band_energy.json
  v14_report.json
docs/v14/V14_RESULTS.md
```
