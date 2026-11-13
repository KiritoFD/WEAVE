# WEAVE Documentation Index

**Last updated:** 2026-11-14（本索引随 2026-11-14 仓库重组重写；2026-07 版本所指向的提交快照文档已归档至 `docs/legacy_root/` 与 `docs/713/`）

## Start Here（按顺序读）

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [AGENTS.md](<../AGENTS.md>)（仓库根） | **决策协议（2026-11-14）**：解码一律 S10 + 使用契约 + 证据；覆盖一切历史文档的相反约定 |
| 2 | [HIGHLIGHTS.md](<HIGHLIGHTS.md>) | **两大进展一页纸**：S10 剪枝 decoder、Q2A+Oracle 协议（数字+出处+链接） |
| 3 | [v9/V9_LD_RESULTS.md](<v9/V9_LD_RESULTS.md>) | 最新完成的工作：V9-LD latent 消融（LD4 唯一净胜臂） |
| 4 | [WEAVE_RESEARCH_SYNTHESIS_20261112.md](<WEAVE_RESEARCH_SYNTHESIS_20261112.md>) | 全研究线"什么存活下来了"的综合叙事 |
| 5 | [1111/README.md](<1111/README.md>) | WEAVE/Art-VAE 全路线回顾（时间线、协议族谱、54 路线索引、证伪账本） |

## 两大进展的权威文档

- **S10 剪枝 decoder**：`docs/vae/`（00–14 编号集，`vae/README.md` 为入口）——训练全记录、S10B 决策、TRT 速度、恢复式剪枝 postmortem。
- **Q2A + Oracle**：`docs/v6/Q2A_ORACLE_BASELINE_RANKING.md`（协议定义+32 方法校验）、`docs/v6/Q2A_ORACLE_DECODER_RANKING.md`（decoder 配对证据）；冻结证据 `deliver/q2a_oracle_scores_baseline_protocol.json`。
- **排行榜**：[LEADERBOARD.md](<LEADERBOARD.md>)（32 外部方法 + Ours 全维度）。

## 按阶段导航

| 目录 | 状态 | 内容 |
|---|---|---|
| `docs/v9/` | **当前** | V9-LD 预注册计划 + 进展 + 结果（RTX 4070 本地） |
| `docs/vae/` | **当前（S10 权威）** | S10/S10B 全套；与 docs/v6 的重复已去重（stub 指针） |
| `docs/v8/` | 当前（队列暂停） | V8 A/B/C/I 矩阵；注意其"标准 SD-VAE 口径"已被作废（README 有修订横幅） |
| `docs/v6/` | 混合 | Q2A 系列权威在 v6；S10 系列原件已 stub → vae；V6_24H/V6_NEXT 历史报告 |
| `docs/v7/` | 已关闭 | V7/V7-R，结论已被 1111 收编 |
| `docs/baseline/` | **当前** | 13 外部方法 × 3 数据集评测图档案 + UNIFIED_FULL_RANKING + 3060 复现记录 |
| `docs/maths/` | 历史 | 2026-05 理论文档（Schrodinger-bridge 时代） |
| `docs/negative_results/` | 当前参考 | 有约束的负结果精选（L1–L4 分类） |
| `docs/1111/` | **当前（回溯权威）** | 全路线回顾 + 证伪账本（`1111/comparison/04_bugs_corrections_and_invalid_claims.md`） |
| `docs/legacy_root/` | 已归档 | 原根目录的论文快照期文档（CHECKPOINT/DATA/EXPERIMENTS/Exp/Plan…） |
| `docs/113/`、`docs/713/`、`docs/reproduction/`、`docs/archive/`、`docs/exp/`、`docs/model_probe/` 等 | 历史归档 | 2026-06～07 提交快照期及更早 |

## 活跃契约（2026-11-14 现状）

- 分支 `main`；活跃代码 = `src/`（门面）+ `exp/v8`、`exp/v9_local`；论文期平铺代码在 `tools/legacy_root/`。
- 解码：一律 S10（`src.utils.load_s10_decoder` / `decode_s10`），见 AGENTS.md 契约。
- 评测：`eval_protocol/tasks_750.json`（Cross600+Same150，5 画派），referee 管线 `src/eval/measure.py`。
- 数据：本地 `G:\WEAVE\data|datasets`；3060 执行仓库 `/root/Workspace/WEAVE_VAE`（VAE 线）/ WSL `/mnt/i/Latent_Style/WEAVE`（v8 期口径，历史文档中的路径按当年机器解读）。
- 配置：训练 `config.json`、推理 `inference.json`（均在仓库根，可移植性由 tests 守卫）。
- 3060 当前被他人占用做 StyleGallery 复现，**不要碰**。

## Evidence Rule（不变）

DINO-S 是主风格指标、CLIP-S 次之；DINO-C 与 LPIPS 必须同时报告以防内容坍缩收益。
跨 decoder（S10 vs 原版）的绝对值不可混比——全仓已统一 S10 口径（协议级选择，证据见 HIGHLIGHTS）。
