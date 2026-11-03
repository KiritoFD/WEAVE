# 08 重构清单

> 风险分级：**L**=不影响运行、只动文档/归档；**M**=改配置/文件布局，需要验证；**H**=动训练链路或远端，必须等训练停并由用户确认。
> 当前远端有训练/评测在跑：**M/H 一律不执行**。

## 0 已完成的低风险动作（本次）

| 动作 | 结果 |
|---|---|
| 根目录 16 个一次性脚本归档 | 移到 `archives/legacy-root-oneoff-20261103/`（去掉 `_` 前缀，commit `9e8e40f`） |
| docs/113 真相文档 | 本目录 9 份，作为新的 canonical 索引 |

## 1 立刻可做（L 级，无需等训练）

| # | 动作 | 具体做法 | 风险/注意 |
|---|---|---|---|
| L1 | **文档合并**：把 `docs/archive/{72,710,628,630}` 与 `docs/713` 的重叠内容收进 `docs/113/03_EXPERIMENTS.md`，旧文档移入 `docs/archive/merged-20261103/` | 只移动不删除；保留原始文件便于引用 | 低；不要改 `docs/archive` 以外的路径 |
| L2 | **配置去重**：`config_r5.json` 21KB、`config_taesd.json` 21KB 是 `config.json` 的整份拷贝（drift 风险高），改成 `{"_base":"config.json", "model":{...}, "data":{...}}` 只写差异 | 先 diff 三份 JSON 的每个 key | 低；但若另一会话正用这两个 config 启动，**先确认再用** |
| L3 | **仓库卫生**：`.gitignore` 增加 `docs/archive/**/node_modules/`、`**/.codeartsdoer/`、`**/.drawio_tmp/`；本地实测这些目录有 **1566+57** 个未跟踪文件（磁盘垃圾，会拖慢同步） | 追加规则，不删文件 | 极低 |
| L4 | **临时探针归位**：`tests/_smoke_t26.py` 等 `_` 前缀脚本移到 `archives/oneoff-tools-20261103/` | `git mv` | 低；先确认没有 runbook 引用 |
| L5 | **文档路径修正**：改 `docs/NEXTGEN_RUNBOOK.md` 的 `data/train_d5_taesd`→`data/train_taesd`、`I:/datasets`→`/mnt/i/Latent_Style/WEAVE/datasets` | 纯文档 | 低；但远端只允许写 docs/113，**等训练停再同步** |

## 2 训练停后立刻做（M 级）

| # | 动作 | 具体位置 | 收益 |
|---|---|---|---|
| M1 | **提交下一代代码**（最高优先级）：`features.py/swd.py/output_supervision.py/pseudo_targets.py/reference_encoder.py/texture_module.py` + `tests/test_nextgen_skeleton.py` + `Plan.md/Exp.md/docs/HANDOVER.md/docs/NEXTGEN_RUNBOOK.md` 目前**未跟踪**；`model.py/trainer.py/flow.py/config_schema.py/utils/dataset.py/run.py` 有未提交改动 | `git status` 29 项 | 消除"远端跑的不是 git 代码"风险 |
| M2 | **拆分 `model.py`（1590 行）**：`model/backbone.py`（DWT+ResBlock+cross-attn）、`model/heads.py`（ll/lh/hl/hh + 四类 delta）、`model/solver.py`（Euler/RK4/Heun + AdaIN） | [model.py#L432](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L432)、[L1195](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1195)、[L1442](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1442) | 可读性；行为不变（digest 回归测试必须通过） |
| M3 | **拆分 `trainer.py`（1933 行）**：`trainer/loop.py`（`train_epoch`）、`trainer/ckpt.py`、`trainer/diagnostics.py`（内部探针） | [trainer.py#L1434](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L1434)、[L1981](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L1981) | 同上 |
| M4 | **拆分 `config_schema.py`（1419 行）**：按 `Model/Bridge/Data/Training/Inference` 五块拆；`_split_known_fields` 的"未知键 rehydrate"改为显式白名单 | [L105-L129](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L105-L129) | 去掉隐式魔法；`enable_hh_head` 等字段必须显式声明 |
| M5 | **修 C3/C4/C5**：forward docstring 改回"可选 HH"；`style_condition_source` 改名 `condition_source=style_memory` 或补真实分支；`reference_encoder_in_dim` 默认 384→按权重探测 | 07_ISSUES C3-C5 | 消除误判 |
| M6 | **统一评测入口**：`tools/eval.py` 一次产出 600/150 + 四指标，结果 JSON 强制写 `protocol/backbone/reference_policy`；把 DINO 计算从 `scripts/_repro_dino_eval.py` 收进仓库 | `utils/run_evaluation.py`(4711 行) 已过载 | 解决 C7/C8/E5 |
| M7 | **`t_sampling` 去 legacy**：删 `t_sampling_legacy_compat`，新实验只认显式模式；旧 ckpt 复现用独立 `repro_legacy` flag | [config_schema.py#L884-L891](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L884-L891) | 名称与实际一致 |

## 3 需实验决策（H / 不急着改）

| # | 事项 | 阻塞 |
|---|---|---|
| H1 | 18k latent 离线编码（约 1.6G） | 路线 C；需确认风格对齐方案（5k/18k 仅 Impressionism 重合） |
| H2 | `data/test` 与 `datasets/d5/test` 合并为一份（另一份软链） | 评测脚本默认路径可能写死，需 grep 全部引用 |
| H3 | 数据/缓存目录迁移到 `I:\Latent_Style\datasets\` 顶层并按 `raw/latent/packed` 分层 | 远端有运行中任务，禁止移动 |
| H4 | 把 `runs/`（含 ckpt）纳入 git-lfs 或独立 artifact 仓 | 体积大，且远端非 git |
| H5 | `utils/run_evaluation.py` 拆成 metrics / io / viz / main 四模块 | 4711 行；改动面大，需逐函数快照测试 |

## 4 回归保护（改代码前必须先有）

1. `tests/test_digest.py`：默认配置 + 固定 seed 的前向/反向 digest 必须与改动前逐位一致（当前 `tests/test_nextgen_skeleton.py` 只测开关，不够）。
2. `tests/test_solver_equivalence.py`：HH 关闭时 Euler/RK4/Heun 与旧实现数值一致；HH 开启时 `v_hh` 必须进入积分（防 C1 回归）。
3. `tests/test_eval_split.py`：750→600/150 的计数与均值（`run_eval_split --smoke` 已有，补断言）。
4. `tests/test_config_fields.py`：所有 config JSON 的键都能被 schema 显式识别（防 C5 类隐式字段）。

## 5 建议的提交顺序

```
L3 .gitignore           → 1 commit
L1 docs 合并             → 1 commit（只移动）
L2 config 去重           → 1 commit（先 diff）
M1 提交现有代码           → 1 commit（用户确认）
M5 文档/命名修正          → 1 commit
M2/M3/M4 拆分            → 每模块 1 commit + digest 回归
M6 评测统一              → 1 commit + 600/150 复算对照
```
