# WEAVE / PRISM Documentation Master Index

**Last updated:** 2026-11-18（统一单源权威索引；历史快照与中间进度日志均已归档至 `docs/archive/`）

---

## 一、核心必读入口（Start Here）

| 顺序 | 文档 | 定位与内容 |
|---|---|---|
| **1** | [`../AGENTS.md`](../AGENTS.md) | **工程与科学铁律协议**：一律使用 `S10` 解码、四裁判子进程隔离、12GB(3060)/24GB(4090) 显存规范、当前活跃主线 |
| **2** | [**`performance/MASTER_LEDGER.md`**](performance/MASTER_LEDGER.md) | **全仓唯一数据与结论总台账**：V6→V19 全部 750 任务实测数据（`metrics.csv` / `metrics.json`）、六条方法论铁律与帕累托前沿矢量图 |
| **3** | [**`icme2027/README.md`**](icme2027/README.md) · [`STORY.md`](icme2027/STORY.md) | **ICME 2027 独立论文工程（`PRISM`）**：6 页双栏 `IEEEtran` 完整手稿（`paper.tex` / `paper.pdf`）、可插拔宏 `macros.tex` 与全套矢量生成脚本 |
| **4** | [**`next1118/README.md`**](next1118/README.md) | **`PRISM v2 / V19` 正交商—纤维对偶与各向异性谱传输蓝图**：源码级病灶剖析、完整数学推导与 4090 执行方案 |
| **5** | [`HIGHLIGHTS.md`](HIGHLIGHTS.md) · [`vae/README.md`](vae/README.md) | **基石设施档案**：`S10` 剪枝解码器（`10.66M`，`2.86×` 加速，`+3.17pp` Top-1）与冻结 `tasks_750` 四裁判协议 |
| **6** | [`../exp/README.md`](../exp/README.md) | **实验代码包总索引**：`exp/v8` → `exp/v20` 模块入口、职责与文档映射 |

---

## 二、分代结论档案导航（Clean Per-Generation Index）

| 目录 / 文档 | 状态 | 核心结论摘要 |
|---|---|---|
| [`next1118/`](next1118/README.md) · [`04_V20`](next1118/04_V20_SUBMODE_SLF_THEORY.md) | **当前活跃 (`exp/v19`–`v20`)** | `V19` 建立 $(\Pi_\perp, \Pi_\parallel)$ 正交对偶与 `S10-VGG` 各向异性求解器（`59.0%` Top-1）；`V20` 引入鲜明子流派 `Patch-SWD` 与仅 $16\times 16$ 瓶颈注入结构的粗布局生成流 `SLFUNet` 冲刺超越 `StyleGallery` |
| [`v18/V18_RESULTS.md`](v18/V18_RESULTS.md) | 已完成突破 | Gate-0 去风格结构（泄漏 `72.3%→18.3%`）；Track A `LD4+`（`38.0%–42.2%`）与纯单步 `FastStudent`（`33.8%–38.0%`）；Track B 确立 $\tau_0=0$ 自配对控制流 `18.67%` 天花板 |
| [`v17/V17_RESULTS_EVAL.md`](v17/V17_RESULTS_EVAL.md) | 已收尾 | 揭示 SDEdit 59% 源锁定与 Impressionism 先验坍缩；`P2_S4`（DINO 上下文 + `u_tgt−u_src`）解决单步残差拟合 |
| [`v16/RESULTS.md`](v16/RESULTS.md) · [`v15/`](v15/WEAVE_V15_POSTSCALE_AND_POSTTRAIN_SUMMARY.md) | 已收尾 | 确立 **260× 自配对扩容悖论**（$0.13\text{M}\to 34.2\text{M}$ 纯生成升至 `49.3%`，$\tau_0=0.7$ 编辑退化至 `6.67%`） |
| [`v14/V14_RESULTS.md`](v14/V14_RESULTS.md) · [`v13/V13_RESULTS.md`](v13/V13_RESULTS.md) | 诊断定论 | 确立 224px DINOv2 16×16 patch（NCM `.793` >> DWT `.544`）为最优语义布局载体 |
| [`v12/V12_GCL_RESULTS.md`](v12/V12_GCL_RESULTS.md) · [`v11b/V11B_RESULTS.md`](v11b/V11B_RESULTS.md) | 已收尾 | 证明复杂内容 Query 门控与风格梯度近正交（不 binding），瓶颈在判别器域与表示 |
| [`v10/V10_RESULTS.md`](v10/V10_RESULTS.md) · [`v9/V9_LD_RESULTS.md`](v9/V9_LD_RESULTS.md) · [`v8/`](v8/README.md) | 早期基石 | 产出 `N2C`（`32.83%`）、`LD4`（`35.17%`）、`E4`（DINO-S `0.4415`），证伪检索目标与手工 KDE 特异性 |
| [`vae/`](vae/README.md) · [`v6/`](v6/Q2A_ORACLE_BASELINE_RANKING.md) | **协议权威** | `S10` 00–14 完整训练与测速卷宗；`Q2A+Oracle` 750 任务四裁判冻结基准 |
| [`baseline/`](baseline/README.md) · [`LEADERBOARD.md`](LEADERBOARD.md) | 外部对照 | 19 外部方法评测图档与复现记录（`LEADERBOARD.md` 为早期 32 行冻结快照，全量数据以 `performance/MASTER_LEDGER.md` 为准） |
| [`negative_results/`](negative_results/README.md) · [`1111/`](1111/README.md) | 证伪与回溯 | L1–L4 负结果分类账本、早期 54 路线回顾与纠错边界（`04_bugs_corrections_and_invalid_claims.md`） |
| [`legacy_root/`](legacy_root/) · [`archive/`](archive/README.md) | 历史冷存 | AAAI 投稿期文档与所有历史中间阶段计划、进度快照 |
