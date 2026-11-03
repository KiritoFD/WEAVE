# 07 已知问题与技术债

> 标注规则：**已证**=本地代码/数据实测；**未验证**=只有文档说法。不要在训练结束前改远端代码。

## 1 代码问题

| # | 问题 | 状态/证据 | 影响 |
|---|---|---|---|
| C1 | Heun/RK4 曾把 `v_hh` 丢掉 | **已修**：[model.py#L1232](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1232)、[#L1256](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1256) 注释 "P1 fix ... silently dropped" | 历史 ckpt 的 HH 集成不可比；新跑无影响 |
| C2 | `t_sampling_mode="logit_normal"` 实际是 uniform | **已加诚实开关**：[config_schema.py#L884-L891](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L884-L891) `t_sampling_legacy_compat=True` 保持 uniform 并 warn | 论文"uniform"正确；配置字段名骗人，新实验须显式 `False` |
| C3 | HH 头"死参数"说法 | **部分失实**：forward docstring 写"HH removed"（[L762](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L762)、[L835](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L835)），但 `enable_hh_head=True` 时仍构建并返回 `out["hh"]`（[L1072-L1117](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1072-L1117)、[L1189-L1191](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1189-L1191)），且 solver 已集成 | 易误判"没有 HH 代码"；实际是默认关闭 |
| C4 | `style_condition_source="target_dino_patches"` 实际走 `style_memory` | [model.py#L446](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L446) 注释明说 | 配置语义误导，排查条件路径时容易走错 |
| C5 | `reference_encoder_in_dim` 默认 384（DINOv2-small），服务器 cache 是 dinov2-base 768d | [config_schema.py#L717-L721](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L717-L721)；cache 实测 768d | 开路线 C 必须显式改 768，否则形状错 |
| C6 | 训练 AdaIN=1.0、论文推理 AdaIN=2.0，但 config 里只有一个 `endpoint_adain_scale` | `config.json` + `inference.json` 覆盖 | 复现论文必须同时改；否则指标漂移 |
| C7 | DINO 指标不在 `utils/run_evaluation.py` 内 | 该文件无 DINO 代码；靠 `scripts/_repro_dino_eval.py` 调外部脚本 | 评测链路两段式，容易只跑一半 |
| C8 | `run_eval_split.py` 只做事后拆分，不强制参考图策略 | [tools/run_eval_split.py#L1-L25](file:///g:/GitHub/Latent_Style/WEAVE/tools/run_eval_split.py#L1-L25) | 不同 run 的参考图选择可能不同 → 指标漂移 |
| C9 | 原根目录 16 个 `_mv_*.sh/_chk*.sh` 含 `rm -rf`（**已归档**到 `archives/legacy-root-oneoff-20261103/`） | 实测 `mv_base2.sh#L3`：`find results/baselines -type d -name images -exec rm -rf {} +`；路径仍指向旧机 `/mnt/g/...` | 误运行会删结果图；保留仅作历史 |
| C10 | `tests/_smoke_t26.py` 等临时探针混在测试/工具目录 | 文件名 `_` 前缀；实测在 `tests/` | 新人误用 |

## 2 代码/文档一致性

| # | 文档说法 | 实际 | 证据 |
|---|---|---|---|
| D1 | `docs/NEXTGEN_RUNBOOK.md` 用 `data/train_d5_taesd/.latent_cache/packed` 当 18k cache | 目录不存在；18k 无任何 latent | [01_DATA.md](01_DATA.md) §5 |
| D2 | `DATA.md` 说 `wikiart512_5style` 是"D5-512 源 / 同 5 风格" | 18k 风格与 D5 只有 Impressionism 重合 | `docs/HANDOVER.md#L158-L163` |
| D3 | `DATA.md` 说 `r5` 与 D5 同风格 | R5 = Cubism/Expressionism/Pop_Art/Romanticism/Symbolism | 实测第 0 节表 |
| D4 | `docs/79/main_table_v4_remote.csv` 的 WEAVE 行 0.715/0.382、903K | `config.json` 主表 0.7128/0.2593/0.8103/0.4916、约 1.04M | [04_RESULTS.md](04_RESULTS.md) §6 |
| D5 | `docs/archive/merged-20261103/713/SUBMISSION_HANDOFF` 写远端 `I:\Github\Latent_Style\WEAVE` | 当前远端是 `I:\Latent_Style\WEAVE`；旧机另有一份 | handoff L64 vs 实测 |
| D6 | `NEXTGEN_RUNBOOK` 命令用 `I:/datasets/...` | 实际 `/mnt/i/Latent_Style/WEAVE/datasets/...` | 实测 |

## 3 数据/评测债

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| E1 | 18k 无 latent → "数据规模"路线阻塞 | 实测 | 先 VAE 编码 18k（约 1.6G）再谈 |
| E2 | 论文 DINO 是 dinov2-small，Nov 口径是 dinov2-base，迁移无记录 | legacy 脚本 L7；`Exp.md#L22` | 在结果文件里写死 backbone 字段 |
| E3 | `distinct5_512/test` 300 文件（含 `processed_data` 空目录），易误算 150 | `docs/HANDOVER.md#L151` | 只按 `*-style` 子目录计数 |
| E4 | `data/test` 与 `datasets/d5/test` 双份，内容同源 | 实测 150=150 | 保留一份，另一份软链 |
| E5 | 750/600/150 三口径并存，结果文件不记录协议 | [04_RESULTS.md](04_RESULTS.md) | 结果 JSON 强制写 `protocol` 字段 |
| E6 | 基线主表缺 DINO-S/DINO-C | `docs/79/main_table_v4_remote.csv` 只有 clip/lpips/musiq | 补测或论文只比 CLIP/LPIPS |
| E7 | **评测结果未强制落盘 + 未纳入 registry**：结果散落在 `runs/*/results.json`、`metrics.csv`、`dino_summary.json` 三种形态，只有部分 run 追加到 `registry.jsonl` | 实测 `/mnt/i/Latent_Style/runs/`：`nextgen_eval_quick/*` 只有 `metrics.csv`；`nextgen_eval_smoke/` 只有 `dino_summary.json`；`results.json` 只在 `nextgen_eval_full/*` 与 `routeA_scan_eval/*` 存在；registry 只记录 routeA 三条 | 每次评测必须落 `results.json`（含 `protocol`/`counts`/`reference_selection`/backbone）并追加一行到 `registry.jsonl`；**禁止只在 stdout 或 csv 里留数** |
| E8 | **论文主表 0.4916 冻结（won't fix，2026-11-04 决定）**：缺口仍属实——无原始 eval JSON、无具体 ckpt、无 750 请求清单 | `config.json#L3-L12` 只有常量；`config.json#L8` 指向 `runs/submission/canonical_oriented_epoch4/epoch_0004.pt`，该 ckpt 不在本地也不在远端仓库内；`/mnt/i/Latent_Style/runs/` 无 submission 评测产物 | **不再列为待办、不再追复算**。论文 0.4916 一族（含 0.4915/0.4886 等同族数字）仅作**历史标注**，不作为当前比较基准，**禁止与新口径横比**。若日后确需坐实，路径是：用该 ckpt + `inference.json`（`num_steps=8`, `endpoint_adain_scale=2.0`）重跑 750 全量，输出 `results.json` + `results_split.json` 并写 registry，600/150 与 `all_*` 同时对上才算复现 |

## 4 待验证假设（训练停后再做）

1. **VAE 重建上限**是否足以支撑 latent HH（`tools/vae_recon_probe.py`，未跑）。
2. **5k vs 18k** 的收益是数据量还是风格差异（需同协议、同风格子集）。
3. **参考图交换**输出是否真的跟随参考（`diag_reference_swap.py`，未跑）。
4. **路线 A** 在 >1000 步、lr≥1e-5 时是否越过平台期（当前两轮 loss 不降）。
5. **路线 C** 替换/并联 Style-ID 的参数量与 Identity 代价。
6. **batch=2 是否可在关掉 output_supervision 时恢复**（当前 OOM 只发生在 A 的 VGG 前向）。

## 5 仓库卫生

1. ~~**本地 git 脏**~~ **已处理（2026-11-03）**：29 项未提交改动已按 4 个主题提交（`4503372`/`622fc8f`/`504a243`/`8e992a7`），工作区干净；远端仍在跑同一份代码，故仍**未 push、未覆盖远端代码**。
2. `docs/archive/612-phase2/.codeartsdoer/node_modules`、`*.drawio_tmp` 是历史垃圾，占空间且拖慢同步（`.gitignore` 已忽略，文件仍留在磁盘）。
3. ~~`docs/` 下 `713/` 与 `archive/{72,710,628,630}` 内容高度重叠~~ **已完成**：已合并进 `docs/113/03_EXPERIMENTS.md`，原目录移到 `docs/archive/merged-20261103/`。
4. `Exp.md` 只追加不清理，`无效` 行与有效行混排；建议 `state/registry.jsonl` 机器可读化。
