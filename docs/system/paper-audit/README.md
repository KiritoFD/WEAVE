# Paper Final Audit｜论文最终实验审计与证据补全（F0–F8）

**原则**：主线权重与 Encoder 架构全面冻结；**不新增可学习参数、不启动性能提升型训练、不追求更高 `test750` 分数**；仅补齐泛化边界、独立视觉评价、机理响应分析与终版性能核查。总 GPU 预算硬上限 **12 GPU·h**。

## 1. 论文核心科学叙事

> **More capacity can preserve the wrong information. Less capacity can destroy the right information.**  
> 风格迁移的关键不是整体增减容量，而是在正确的位置保留正确的信息、去除错误泄漏的统计/约束。

| 系统部件 | 实证操作 | 应保留的信息 | 应去除的信息 / 结论 |
|---|---|---|---|
| **Encoder (`A0`)** | `34.16M → 19.23M (enc20)` 内容退化；原 `A0 + TRT FP16` 无损 `1.36×` | 精确空间与细节信息 | 不主动压缩；工程加速采用无损 FP16 |
| **Conditioning** | `Stiefel Q32` 消除二阶 Gram 泄漏 | 对象语义与布局 | 源画法二阶统计泄漏 |
| **E-Core** | `11.13M` 单步高保真编辑 | 高保真结构与编辑方向 | 源风格吸引域 |
| **P-Core** | `22.29M → 44.54M` 强风格能力显著提升 | 目标画法生成与重绘能力 | 过细源结构约束；`CFG` 离线蒸馏在大引导区失真（停止） |
| **Decoder (`S10B`)** | `49.49M → 10.66M` (-78.5%), Top-1 反升 +12.17pp | 潜变量中有效编辑方向 | 阻碍风格表达的部分解码纹理响应 |

## 2. 冻结配置定义（唯一权威口径）

| 模型名称 | 组成配置 | 定位边界 |
|---|---|---|
| **`CLEAVE-E`** | `11.13M` E-Core + 原 `A0` + `S10B` | 闭集高保真路径（不加载 P-Core） |
| **`CLEAVE-P/E`** | `11.13M` E + `44.54M` P + `A3` + `A0` + `S10B` | 闭集全谱双核主通路（`7×4B + 1×E = 29` Flow NFE） |
| **`CLEAVE-One`** | `V36 P-Core (45.22M)` + `Reference-A3` (`6 AB2/Euler steps = 18 CFG evals`) | 严格未见 Core-5 的开放域单参考基线；不混同闭集或 Oracle |

*(已结题不再重复的实验：Stiefel 同构消融、Q32 语义置零/打乱、E-Core 草稿 2×2、A3 消融、P-Core 扩容、Pool C、Unified-58M 对照、Decoder 质量对比、Encoder 蒸馏、One-shot V35–V37 训练、主干超参搜索。)*

## 3. F0–F8 审计进度矩阵

| 阶段 | 内容 | 预算 | 优先级 | 状态 | 产物 |
|---|---|---:|---|---|---|
| **F0** | 最终模型、资产、蒸馏数据同步核验与评测协议审计 | 0.5h | 必做 | **✅ 已完成** | [F0_PROTOCOL_AUDIT.md](F0_PROTOCOL_AUDIT.md) · [DISTILLATION_RESULTS.md](../optimization/DISTILLATION_RESULTS.md) |
| **F1** | `Artist-Holdout-750` 真正未见画家泛化与重叠审计 | 1.5h | 必做 | 待确认执行 | — |
| **F2** | 独立 VLM 盲测视觉评价（内容/画法/艺术性/伪影/泄漏） | 2.0h | 必做 | 待选定 VLM 后端 | — |
| **F3** | One-shot `V36` 实例级风格控制汇总（Swap / Color-Matched / SG-25） | 1.5h | 必做 | 待执行（复用 V37 反事实） | — |
| **F4** | Decoder/Encoder 非对称性与 `Texture Retraction` 编辑响应分析 | 1.0h | 必做 | 待执行（资产已核验就位） | — |
| **F5** | E/P 交界连续性（`H0` 硬切换 vs `H1` 插值 vs `H2` 联合） | 0.75h | 必做 | 待执行 | — |
| **F6** | Photo-to-Art (`P2A`) 跨域泛化边界评测 | 1.5h | 推荐 | 待执行（`P2A-256` 已核验就位） | — |
| **F7** | 最终性能定型（A0 TRT / S10B TRT / E→P TRT / Stiefel 误差） | 1.25h | 必做 | 待执行（`A0 TRT` 已完成） | — |
| **F8** | 统一总账、LaTeX 宏与自动化 Pareto 图汇总 | 2.0h | 机动 | 随 F1–F7 汇总 | — |
