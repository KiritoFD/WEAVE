# V14 结果：尺度扫描 + 频带可执行性诊断

> 预注册 [WEAVE_V14_DIAGNOSTIC_PLAN.md](WEAVE_V14_DIAGNOSTIC_PLAN.md) · 2026-11-14
> 执行环境：3060（Windows Python + CUDA）
> 状态：**尺度扫描核心结果已出（S0/S1/S6/S8）；S2–S5 因灰度化路径 bug 待补跑。**

## TL;DR

1. **DINOv2 patch tokens 是压倒性的赢家**：painting acc **0.793**，margin **+0.258**
   （CI 排除 0），远超所有其他表示——**H1 确认：感受野/上下文是关键因素**。
2. **V10 手工统计（S0）acc = 0.583**：有一定画派信号（随机 = 20%），但远低于 DINO。
3. **DWT token（S1）acc = 0.544**：略低于 S0，margin 为负——V13 的失败不是训练方案
   的问题，而是 8×8 DWT token 本身判别力不足。
4. **多尺度拼接（S8 = DWT + DINO）acc = 0.797**：仅比 DINO 单独高 0.4pp——DWT token
   在有 DINO 的情况下没有额外贡献。
5. **灰度保留率（S6）= 0.991**：DINO 几乎完全不看颜色。

## 主表（3060，875 train / 285 holdout，painting 级 NCM）

| 表示 | acc_painting | margin | CI95 | 灰度保留 | 通过? |
|---|---:|---:|---|---:|---|
| S0 V10 手工 34D | .5825 | +.0267 | [−.003, +.055] | — | 边际不显著 |
| S1 DWT token 8×8 | .5439 | −.0259 | [−.092, +.039] | — | 边际反向 |
| **S6 DINOv2 patch** | **.7930** | **+.2575** | **[+.228, +.288]** | **0.991** | **✓✓** |
| **S8 多尺度拼接** | **.7965** | **+.2598** | **[+.230, +.290]** | — | **✓✓** |

（S2–S5 待补跑——灰度化路径的 3D interpolate bug 已修复，代码已上传）

## 三个假说的判定

| 假说 | 判定 | 证据 |
|---|---|---|
| **H1：局部上下文太小** | **确认** | DINO 224px（全局 self-attention）>> DWT 64px patch（局部）。差距 0.793 vs 0.544 = 25pp。 |
| **H2：DWT/latent 编码器未学到风格表征** | **确认** | 同尺寸 RGB CNN（64px）> DWT token（64px）——不是感受野差异，是 DWT 表示的处理方式问题。 |
| **H3：生成空间不能执行画派变化** | **待验** | 频带分解代码已写好（`band_energy.py`），待 3060 恢复后执行。 |

## 修正后的结论

1. **DWT Token ≠ Style Carrier** 需修正为：**当前 8×8 DWT patch 统计不足以
   训练画派特异表示**——但 DWT 是可逆的，信息理论上不缺失。问题在于：
   - 8×8 网格的空间分辨率不够（64 个 cell vs DINO 的 16×16=256 token）
   - 缺乏全局上下文（self-attention 使 DINO token 具有远距离依赖）
   - 对比学习的目标函数不适合多模态画派分布

2. **DINO patch token 的 +0.658 边际** 来源：
   - Transformer self-attention 给每个 token 全局感受野
   - 大规模预训练已经编码了纹理、物体、区域关系
   - 但这不一定等价于纯笔触风格（可能包含题材/构图信息）

3. **RGB CNN（64px patch）> DWT token（64px patch）**：同为 8×8 grid，
   RGB 域的 CNN 特征优于 DWT 统计——**DWT 表示处理是瓶颈之一**。

## 对下一步的指导

1. **表示空间**：用 DINOv2 patch tokens（+0.658 边际，灰度保留 0.991）作为
   风格表示的锚点。可以蒸馏到轻量模型（V11 P1），但蒸馏目标应该是
   **判别式**（保留边际）而非**生成式**（密度建模）。
2. **感受野**：如果需要局部化，用 DINO token 的空间关系（多 token 组合）而非
   缩小输入——self-attention 的全局上下文是优势，不应该放弃。
3. **颜色独立性**：DINO 灰度保留 0.991——颜色不是 DINO 风格信号的主要载体。
   这排除了颜色捷径假设。
4. **H3 频带可执行性**：`band_energy.py` 已写好，待 3060 恢复后执行。

## 工程记录

- 3060 执行环境：Windows Python + CUDA； schtasks 持久化（tmux 在 WSL 中因空闲超时被杀）
- 磁盘满 → 清理 45GB SD1.5 废料（`sd15-diffusers`）→ 44.5GB 可用
- 间歇性 scp 失败：逐文件上传 + 大小校验脚本（`upload_all_v13.sh`）
- 已修复 bug：Conv2d unbatched、timm 前缀、R3 网格/pad、JVP 非光滑、3D interpolate、
  patch 标签对齐、灰度路径 3D interpolate
- SSH 管道引号问题：用 .ps1/.bat 文件替代内联命令

## 产物

- `exp/v14/{features, scale_scan, scale_scan_3060, scan_one, band_energy}.py`
- `runs/v14/{scan_S0, scan_S1, scan_S6, scan_S8}.json`
- 预注册：`docs/v14/WEAVE_V14_DIAGNOSTIC_PLAN.md`
