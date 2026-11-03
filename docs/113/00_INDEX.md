# WEAVE 整理索引（docs/113）

> **报数必须先写口径**：`750 all-pairs` 还是 `600 跨风格`；DINO 主干是 `dinov2-small` 还是 `dinov2-base`。缺任一项的数字视为无效。以后每份结果表都遵守这条。

> 生成于 2026-11-03；所有数字来自 3060 实测或仓库内文件，引用均带路径/行号。**本目录是新的 canonical 入口**；旧文档保留在 `docs/` 与 `docs/archive/merged-20261103/`（原 `docs/713` 及 `archive/{72,710,628,630}`）。

## 推荐阅读顺序

1. [05_METHOD.md](05_METHOD.md) —— 先懂模型在做什么、为什么现在卡住。
2. [01_DATA.md](01_DATA.md) —— 5k/18k/评测集到底在哪、缺什么。
3. [03_EXPERIMENTS.md](03_EXPERIMENTS.md) —— 哪些路已经走过、别再重试。
4. [04_RESULTS.md](04_RESULTS.md) —— 数字属于哪套口径，哪些不能横比。
5. [02_CODE_MAP.md](02_CODE_MAP.md) —— 要改代码时看调用链与配置流。
6. [06_RUNBOOK.md](06_RUNBOOK.md) —— 上机、跑诊断、避坑。
7. [07_ISSUES_TODO.md](07_ISSUES_TODO.md) —— 当前 bug/债/待验证假设。
8. [08_REFACTOR_PLAN.md](08_REFACTOR_PLAN.md) —— 按风险排序的清理清单。

## 每篇一句话

| 文件 | 一句话作用 |
|---|---|
| [00_INDEX.md](00_INDEX.md) | 导航、阅读顺序、状态快照 |
| [01_DATA.md](01_DATA.md) | 5k（有 latent）、18k（只有原图）、150 评测集、600/150 协议与真实路径 |
| [02_CODE_MAP.md](02_CODE_MAP.md) | 入口→模型→训练→评测的模块职责、行号、配置生效链 |
| [03_EXPERIMENTS.md](03_EXPERIMENTS.md) | 2026-02→11 的实验时间线，每条"试了什么/结果/为何失败" |
| [04_RESULTS.md](04_RESULTS.md) | 四指标定义；750 all-pairs 与 600+150 两套口径的数字与来源 |
| [05_METHOD.md](05_METHOD.md) | 小波+FM 机制、五个瓶颈、下一代 A/B/C/D 设计与依赖 |
| [06_RUNBOOK.md](06_RUNBOOK.md) | 环境实测、离线权重、常用命令、8 个易踩坑 |
| [07_ISSUES_TODO.md](07_ISSUES_TODO.md) | 10 个代码问题、6 个文档失实、6 个数据/评测债、6 个待验证 |
| [08_REFACTOR_PLAN.md](08_REFACTOR_PLAN.md) | L/M/H 三级重构清单 + 回归测试 + 提交顺序 |

## 当前状态（2026-11-03）

| 项 | 状态 |
|---|---|
| 论文主表 | ckpt `runs/submission/canonical_oriented_epoch4`；750 all-pairs：DINO-S 0.4916 / CLIP-S 0.7128 / LPIPS 0.2593 / DINO-C 0.8103 |
| 新口径基线 | 600 跨：0.3977 / 0.6425 / 0.7939 / 0.3053；150 同：DINO-S 0.8263 |
| 路线 A | 机制打通（500/1000 步），无风格增益；扫描矩阵未完成 |
| 路线 C | 依赖 DINO cache（已有）与 18k latent（**缺**） |
| 路线 D | VAE 重建上限探针未跑 |
| 训练机 | RTX 3060 12G，已用 7.7G；固定 batch=1 |
| 远端协作 | 另一会话在跑训练/评测；本次只写 `docs/113/` |

## 本次整理发现的重复/矛盾（摘要）

1. `docs/79/main_table_v4_remote.csv` 的 WEAVE 行（0.715/0.382、903K）≠ `config.json` 主表（0.7128/0.2593、约 1.04M）——CSV 是旧快照。
2. `DATA.md` 把 18k 与 R5 都写成"与 D5 同 5 风格"——实测只有 Impressionism 重合。
3. `docs/NEXTGEN_RUNBOOK.md` 的 `data/train_d5_taesd` 与 `I:/datasets` 路径均不存在。
4. `model.py` forward docstring 说"HH removed"，实际 `enable_hh_head=True` 仍可返回 HH。
5. `style_condition_source="target_dino_patches"` 实际走 `style_memory`。
6. 本地 git 有 29 项未提交（下一代代码 + Plan/Exp/文档），远端却在跑这些代码。
7. `docs/archive/*` 有 1566 个未跟踪的 `node_modules` 文件与 57 个 drawio 缓存文件。
8. 根目录 16 个 `_mv_*.sh` 含 `rm -rf`，已归档到 `archives/legacy-root-oneoff-20261103/`。

详见 [07_ISSUES_TODO.md](07_ISSUES_TODO.md)。
