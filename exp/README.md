# exp/ — 实验代码谱系（V8 → V26）

> **仓库规范**：`exp/` 仅存放模块化实验代码（均以 `python -m exp.<pkg>.<mod>` 从仓库根运行，统一经 `paths.py` / 相对路径寻址，通过 `tests/test_no_absolute_paths.py` 防硬编码护栏校验）。运行时权重、缓存与评测输出一律写入 `runs/`（gitignore 隔离），跨代统一指标汇总于 [`docs/performance/`](../docs/performance/)。

## 实验包一览

| 包路径 | 状态 | 研究主线与核心结论 | 主要入口 | 对应文档 |
|---|---|---|---|---|
| `exp/v8/` | 已归档收尾 | 受控风格控制（A 层次控制 / B SPD 预处理梯度 / C 实画支持集 / N2 有约束优化），产出 **`N2C`** 前馈基底（Top-1 `32.83%`） | `python -m exp.v8.train` / `run_arm` | [`docs/v8/`](../docs/v8/) |
| `exp/v9_local/` | **谱系 A 基石** | 在 `N2C` 潜空间上叠加 SPD 预处理 `R1` + Haar 局部风格能量 + Top-25% 风险门控掩码，产出 **`LD4`（Top-1 `35.17%` / LPIPS `0.4872`）** | `python -m exp.v9_local.run` | [`docs/v9/V9_LD_RESULTS.md`](../docs/v9/V9_LD_RESULTS.md) |
| `exp/v10/` | 已收尾 | 真实画作内容条件闭式 KDE 密度梯度，产出 **`E4`（DINO-S `0.4415` / Affinity `0.5616`）**；证伪检索目标（V10.1）并量化统计空间 5% 特异性上限（V10.2） | `python -m exp.v10.run_v10` | [`docs/v10/V10_RESULTS.md`](../docs/v10/V10_RESULTS.md) |
| `exp/v11b/` | 已收尾 | RGB 正交 Haar 小波子带条件 Flow（H2/H1 频带建模与 `bands.py` 工具层，供后续 H2 实验复用） | `python -m exp.v11b.run_v11b_750` | [`docs/v11b/V11B_RESULTS.md`](../docs/v11b/V11B_RESULTS.md) |
| `exp/v12_gcl/` | 已判负关闭 | 内容查询门控（Content-Query GCL ×5），证实风格梯度与内容查询近正交、门控不触发 | `python -m exp.v12_gcl.run_gcl` | [`docs/v12/V12_GCL_RESULTS.md`](../docs/v12/V12_GCL_RESULTS.md) |
| `exp/v13/` | 已判负关闭 | DWT 小波域对比学习编码器（修复 `sigma` $\sqrt{2}$ 计数与平坦区 NaN），证实局部 DWT token 无法承载画派语义 | `python -m exp.v13.run_p1` | [`docs/v13/V13_RESULTS.md`](../docs/v13/V13_RESULTS.md) |
| `exp/v14/` | 诊断定论 | 多尺度表征扫描（S0–S8）：确立 **224px DINOv2 16×16 patch（NCM `79.3%` >> DWT `54.4%`）** 为最优语义风格/布局载体 | `python -m exp.v14.scale_scan` | [`docs/v14/V14_RESULTS.md`](../docs/v14/V14_RESULTS.md) |
| `exp/v15/` | 已收尾 | 16 臂容量×注意力×Where 矩阵 + Scaling（`dim96→384`, 34.2M）+ Style-Bridge 后训练：证实自配对容量提升 $\tau_0=0$ 生成（`49.3%`）但加剧 $\tau_0=0.7$ 保源锁定（`6.67%`） | `python -m exp.v15.train` / `eval_arms` | [`docs/v15/`](../docs/v15/) |
| `exp/v16/` | 已收尾 | 严格留出容量诊断（`A_strict`）、V15 锚点局部教师蒸馏（`B`）、RGB-H2 有限步 S10 反演（`C`，吸收率 `16.7%`） | `python -m exp.v16.run` | [`docs/v16/RESULTS.md`](../docs/v16/RESULTS.md) |
| `exp/v17/` | 已收尾 | 条件探针/CFG/五标签互换（`P0`，揭示 ER/Uki 先验坍缩与 59% 源锁定）、DINO 上下文 `u_tgt−u_src` 学生（`P2_S4` 达 `13.33%` 反超教师）、Signed RGB-H2（`P3`）、均衡 FM 残差（`P4`） | `python -m exp.v17.run` | [`docs/v17/V17_RESULTS_EVAL.md`](../docs/v17/V17_RESULTS_EVAL.md) |
| `exp/v18/` | 已完成突破 | ① `Gate 0` 零色相亮度对比度 + 空间白化灰度 DINO（泄漏降至 `18.3%`）；② **Track A** 校准 `LD4+`（`38.0%–42.2%`）与纯单步 `FastDINOStudent`（`33.8%–38.0%`）；③ **Track B (`CCF`)** 确立 $\tau_0=0$ 自配对控制流的 `18.67%` 笔触几何天花板 | `python -m exp.v18.run_v18` | [`docs/v18/V18_RESULTS.md`](../docs/v18/V18_RESULTS.md) |
| `exp/v19/` | 已收官 | **`PRISM v2` 正交商—纤维对偶 $(\Pi_\perp,\Pi_\parallel)$**：免源标签连续弦 $e(x_s\to\tau)$ + 潜色度正交分裂各向异性信任域 + 穿过可微 `S10` 的无归一化 `VGG-16+Oklab` 均衡教师（`T4` `59.00%` Top-1，救活 `Ukiyo-e 45%` / `ER 33.3%`）+ 语义路由单步网络（`PRISM-1` `44.33%–50.17%` @ `17.0 ms`） | `python -m exp.v19.run_v19` | [`docs/next1118/V19_RESULTS.md`](../docs/next1118/V19_RESULTS.md) |
| `exp/v20/` | 已收官 | **提高闭集原型条件下的 DINO-S / Pareto 工作点（非任意参考全指标支配）**：纯训练集 `kNN-30` 核心流形原型（剔除 WikiArt 3D 展厅离群点）+ 仅 $16\times 16$ 瓶颈层注入结构的 **Coarse-16 `SLFUNet` 分段视界流**（`DINO-S = 0.5789`, `Affinity = 0.6683`, `Top-1 = 64.00%`）+ 多尺度 `Patch-SWD`（`64.33%` Top-1）+ 零 DINO `PRISM-Flash` 消融 | `python -m exp.v20.run_v20` | [`docs/next1118/V20_RESULTS.md`](../docs/next1118/V20_RESULTS.md) |
| **`exp/v21/`** | 历史完成 | **Stiefel 极分解正交商 + 零类别表 1-Shot 跨注意力流**：消除对角白化残留的 `92.33%` 二阶 Gram 泄漏（降至 `20.00%` 随机水平），支持任意单图 1-Shot（同 `StyleGallery` 参考图达 `0.6006` DINO-S / `0.6761` Affinity @ `135.4 ms`）与 4-Shot 画廊模式（`0.6016` DINO-S） | `python -m exp.v21.run_v21` | [`docs/next1118/05_V21_STIEFEL_ONESHOT_THEORY.md`](../docs/next1118/05_V21_STIEFEL_ONESHOT_THEORY.md) |

## 后续实验代码（V22–V26）

这一部分专门修正旧索引停在 V21 的问题。**模块存在≠实验成功**；是否有效以所列 RESULTS/审计为准，尤其 V26-L 的排队任务。

| 实验包 | 状态 | 重点问题 | 关键入口 / 证据 |
|---|---|---|---|
| `exp/v22/` | 完成 | 固定布局强度、参考适配器、最佳 checkpoint 与初始化锚点 | [`t4.py`](v22/t4.py) · [V22 结果](../docs/v22/V22_P1_RESULTS.md) |
| `exp/v23/` | 完成 | CFG 条件清零 Bug、Anchor-Bridge、中尺度内容损失、One-Shot 跨风格监督 | [`p1_anchor_bridge.py`](v23/p1_anchor_bridge.py) · [V23 结果](../docs/v23/V23_RESULTS.md) |
| `exp/v24/` | 完成 | PRISM-1/Flow 组合、O1/O3 局部参考的 RGB Value 与全局门控 | [`p0_o1_o3_value.py`](v24/p0_o1_o3_value.py) · [V24 进度](../docs/v24/PROGRESS.md) |
| `exp/v25/` | 完成 / 部分模型冻结 | R12/R13 极高内容工作点、C2nt 参考条件、端到端成本核算 | [`r13_highcontent_r12.py`](v25/r13_highcontent_r12.py) · [V25 台账](../docs/v25/ABLATION_LEDGER.md) |
| `exp/v26/` | 诊断完成 | Q32、RefAttn、SR-CFG、局部性 Exp1/Exp2、跨画派教师、Full Decoder matched 与少步 | [`gpu4090/`](v26/gpu4090/) · [V26 文档索引](../docs/1120/README.md) |
| `exp/v33_semantic_control/` | 实测收官（P0–P2+CFG 探针，P3 按门控跳过） | P0 路由 CSLS 与 Y16 亮度解耦、P1 `Q64+` 正交补空间与部分白化泄漏审计、P2 `16×16+32×32` SemanticControlAdapter 与 CFG 重平衡瓶颈定位 | [`run_v33.py`](v33_semantic_control/run_v33.py) · [V33 报告](../docs/v33/V33_SEMANTIC_CONTROL_REPORT.md) |
| `exp/v34_frontier/` | 实测收官（A/B/C/D + Holdout-300） | Stage A 解析起点 `A3` ($L_{\rm low}+P_C$)、Stage B 连续条件 `417D`/`1101D` 响应与前沿消融、Stage C 2×2 `44M Q32` 容量解锁及独立留出集复验 | [`run_v34.py`](v34_frontier/run_v34.py) · [V34 报告](../docs/v34/V34_FRONTIER_REPORT.md) |
| `exp/v35_final/` | 实测定型收官（P0–P8 全栈 + GPU 饱和与批测） | Zero-DINO CLEAVE 全系统因果归因：`P0` 审计与 `A3`、`P1` `E-Core` 因果、`P2` 解码器 (`D0/D1/D2`)、`P3/P4` Stiefel+PoolC 双种子、`P5` 训练时长 bell curve、`P6` 无泄露超参校准、`P7` 大小流、`P8` 终测与耗时分布 | [`run_v35.py`](v35_final/run_v35.py) · [V35 终稿报告](../docs/v35/V35_FINAL_REPORT.md) |

**关键纠偏**：V25 主线多数使用 `dino_16`，不能将 V21 的硬 Q32 当作所有生成模型的统一输入；C2nt 旧局部 `V` 在 V26 K1/Exp1 被判为不值得原样继承。V26-L L1 的早期 **F2=11% 因教师目标条件反置已作废**。[当前执行记录](../docs/1120/PROGRESS.md)。

## 依赖约定

- 门面接口统一从 `src/` 导入（`src.utils.load_s10_decoder` / `decode_s10`、`src.eval.measure`、`src.model.*`、`src.infer.*`）。
- 切勿移动或重构 `tools/` 下的钉住子包（清单见 [`tools/README.md`](../tools/README.md)）。
- 单卡 12GB 评测铁律：生成落盘 `raw/<arm>/` 与 `src.eval.measure` 裁判必须分属两个独立子进程，严禁同进程叠加驻留。
