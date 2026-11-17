# exp/ — 实验代码包总索引（V8 → V18）

> **仓库规范**：`exp/` 仅存放模块化实验代码（均以 `python -m exp.<pkg>.<mod>` 从仓库根运行，统一经 `paths.py` / 相对路径寻址，通过 `tests/test_no_absolute_paths.py` 防硬编码护栏校验）。运行时权重、缓存与评测输出一律写入 `runs/`（gitignore 隔离），跨代统一指标汇总于 [`docs/performance/`](../docs/performance/)。

## 实验包一览

| 包路径 | 状态 | 研究主线与核心结论 | 主要入口 | 对应文档 |
|---|---|---|---|---|
| `exp/v8/` | 已归档收尾 | 受控风格控制（A 层次控制 / B SPD 预处理梯度 / C 实画支持集 / N2 有约束优化），产出 **`N2C`** 前馈基底（Top-1 `32.83%`） | `python -m exp.v8.train` / `run_arm` | [`docs/v8/`](../docs/v8/) |
| `exp/v9_local/` | **谱系 A 基石** | 在 `N2C` 潜空间上叠加 SPD 预处理 `R1` + Haar 局部风格能量 + Top-25% 风险门控掩码，产出 **`LD4`（Top-1 `35.17%` / LPIPS `0.4872`）** | `python -m exp.v9_local.run` | [`docs/v9/V9_LD_RESULTS.md`](../docs/v9/V9_LD_RESULTS.md) |
| `exp/v10/` | 已收尾 | 真实画作内容条件闭式 KDE 密度梯度，产出 **`E4`（DINO-S `0.4415` / Affinity `0.5616`）**；证伪检索目标（V10.1）并量化统计空间 5% 特异性上限（V10.2） | `python -m exp.v10.run_v10` | [`docs/v10/V10_RESULTS.md`](../docs/v10/V10_RESULTS.md) |
| `exp/v11b/` | 已收尾 | RGB 正交 Haar 小波子带条件 Flow（H2/H1 频带建模与 `bands.py` 工具层，供后续 H2 实验复用） | `python -m exp.v11b.run_v11b_750` | [`docs/v11/`](../docs/v11/) |
| `exp/v12_gcl/` | 已判负关闭 | 内容查询门控（Content-Query GCL ×5），证实风格梯度与内容查询近正交、门控不触发 | `python -m exp.v12_gcl.run_gcl` | [`docs/v12/V12_GCL_RESULTS.md`](../docs/v12/V12_GCL_RESULTS.md) |
| `exp/v13/` | 已判负关闭 | DWT 小波域对比学习编码器（修复 `sigma` $\sqrt{2}$ 计数与平坦区 NaN），证实局部 DWT token 无法承载画派语义 | `python -m exp.v13.run_p1` | [`docs/v13/V13_RESULTS.md`](../docs/v13/V13_RESULTS.md) |
| `exp/v14/` | 诊断定论 | 多尺度表征扫描（S0–S8）：确立 **224px DINOv2 16×16 patch（NCM `79.3%` >> DWT `54.4%`）** 为最优语义风格/布局载体 | `python -m exp.v14.scale_scan` | [`docs/v14/V14_RESULTS.md`](../docs/v14/V14_RESULTS.md) |
| `exp/v15/` | 已收尾 | 16 臂容量×注意力×Where 矩阵 + Scaling（`dim96→384`, 34.2M）+ Style-Bridge 后训练：证实自配对容量提升 $\tau_0=0$ 生成（`49.3%`）但加剧 $\tau_0=0.7$ 保源锁定（`6.67%`） | `python -m exp.v15.train` / `eval_arms` | [`docs/v15/`](../docs/v15/) |
| `exp/v16/` | 已收尾 | 严格留出容量诊断（`A_strict`）、V15 锚点局部教师蒸馏（`B`）、RGB-H2 有限步 S10 反演（`C`，吸收率 `16.7%`） | `python -m exp.v16.run` | [`docs/v16/RESULTS.md`](../docs/v16/RESULTS.md) |
| `exp/v17/` | 已收尾 | 条件探针/CFG/五标签互换（`P0`，揭示 ER/Uki 先验坍缩与 59% 源锁定）、DINO 上下文 `u_tgt−u_src` 学生（`P2_S4` 达 `13.33%` 反超教师）、Signed RGB-H2（`P3`）、均衡 FM 残差（`P4`） | `python -m exp.v17.run` | [`docs/v17/V17_RESULTS_EVAL.md`](../docs/v17/V17_RESULTS_EVAL.md) |
| **`exp/v18/`** | **当前主线** | **双轨架构突破**：① `Gate 0` 零色相亮度对比度结构 + 空间白化灰度 DINO 布局（全局风格泄漏降至 `22.3%` 随机水平）；② **Track A** 校准 `LD4+` 与 `B4` 纯单步 `FastDINOStudent` 摊销；③ **Track B (`CCF-UNet`)** 从 $\tau_0=0.0$ 纯高斯噪声起步的 SPADE+SubMode 结构条件生成流 | `python -m exp.v18.run_v18` | [`docs/v18/WEAVE_V18_BREAKTHROUGH_PLAN.md`](../docs/v18/WEAVE_V18_BREAKTHROUGH_PLAN.md) |

## 依赖约定

- 门面接口统一从 `src/` 导入（`src.utils.load_s10_decoder` / `decode_s10`、`src.eval.measure`、`src.model.*`、`src.infer.*`）。
- 切勿移动或重构 `tools/` 下的钉住子包（清单见 [`tools/README.md`](../tools/README.md)）。
- 单卡 12GB 评测铁律：生成落盘 `raw/<arm>/` 与 `src.eval.measure` 裁判必须分属两个独立子进程，严禁同进程叠加驻留。
