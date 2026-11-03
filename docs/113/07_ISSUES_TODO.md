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

## 6 调查：内部门控早停（internal-dynamics early stop，2026-11-04）

**结论：仓库里确实存在，但"内部门控早停"不是 HF ratio / high-frequency energy ratio。** 它是 2026-07 为 oriented-HF 架构做的**训练动力学判据**，核心量是 **LL/HF 梯度范数比 + 目标 HF 门控均值的变化方向**。

### 6.1 是哪个量

| 量 | 定义 | 位置 |
|---|---|---|
| `internal_probe_gate_mean` | LH/HL 两个 `target_latent_hf_subband_delta` 模块的 `tanh(gate)` 均值 | [internal_dynamics.py#L82-L94](file:///g:/GitHub/Latent_Style/WEAVE/internal_dynamics.py#L82-L94)；模块构造 [model.py#L668-L684](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L668-L684) |
| `internal_probe_shared_ll_hf_grad_ratio` | 共享主干参数在 `loss_fm_spectral_ll` 下的梯度范数 / 在 `loss_fm_hf_total` 下的梯度范数（**梯度比，不是能量比**） | [internal_dynamics.py#L147-L159](file:///g:/GitHub/Latent_Style/WEAVE/internal_dynamics.py#L147-L159)；探针 [L97-L164](file:///g:/GitHub/Latent_Style/WEAVE/internal_dynamics.py#L97-L164) |
| `internal_probe_route_shared_hf_grad_ratio`、`..._route_hf_head_grad_ratio` | 目标 HF 分支梯度相对共享主干 / 相对 HF 头的比例，**仅记录、不参与判据** | [L160-L163](file:///g:/GitHub/Latent_Style/WEAVE/internal_dynamics.py#L160-L163) |

探针细节：固定 batch（默认 4 个 latent，`t=0.5`）、`eval` 模式、每个 epoch 额外两次 backward、`torch.random.fork_rng` 保 RNG，不更新优化器（[trainer.py#L436-L516](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L436-L516)）。

### 6.2 怎么触发

判据（[internal_dynamics.py#L185-L195](file:///g:/GitHub/Latent_Style/WEAVE/internal_dynamics.py#L185-L195)）：

```text
epoch >= min_epoch(默认 3)
且 rho_epoch / rho_prev <= 0.65      # 相对收缩，绝对阈值 rho<=1 已被废弃
且 gate_delta > 0                    # 门控从收缩转扩张
```

- **单 epoch 判定**，没有"连续 N 步"或 patience；首次满足即 `transition_epoch`。
- 触发后：`requested_stop=True`（[trainer.py#L513-L516](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L513-L516)）→ 当前 epoch 结束、**保存该轮 ckpt**（[run.py#L663](file:///g:/GitHub/Latent_Style/WEAVE/run.py#L663)）→ 跳出训练循环（[run.py#L694-L696](file:///g:/GitHub/Latent_Style/WEAVE/run.py#L694-L696)）。**不降 lr、不冻结门控、不回退 ckpt、不跳过样本。**
- 默认全关：[config_schema.py#L1075-L1078](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L1075-L1078) `internal_early_stop_enabled=False`、`min_epoch=3`、`gate_delta_threshold=0.0`、`ratio_drop=0.65`；且必须同时 `internal_probe_enabled=True`（默认 False，[L1071](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L1071)；否则 [trainer.py#L455-L456](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L455-L456) 直接 no-op）。

### 6.3 当前是否生效

- **当前训练：关。** 远端正在跑的是 `runs/p0_diag/cap32/config.json`（`ps` 实测：`python3 run.py --config /mnt/i/Latent_Style/runs/p0_diag/cap32/config.json`），该配置里 `internal` 相关键 **0 个**。
- 仓库 `config.json`、`config_r5.json` 均无该键 → 默认关。
- 只有 3 个 rebuttal 配置开了探针但不早停：`configs/exp_rebuttal_{w03,w10,hh_head}_p256.json` 只有 `internal_probe_enabled: true`，没有 `internal_early_stop_enabled`。
- **那次真正启用它的实验配置已不在仓库**：`configs/hf_oriented_internal_early_stop.json` 全仓库 Glob 无文件，`git log --all` 也查不到（从未被跟踪）；`experiments/architecture/` 目录同样不存在。但 `prepare_submission*.py`、`pack_*.py`、`scripts/bench_inference_latency.py`、`docs/reproduction/*` 仍在引用这两个路径 → 复现链断了（属 §2 D 类）。

### 6.4 实现可信度

- 有判读文档：[internal_dynamics_early_stop.md](file:///g:/GitHub/Latent_Style/WEAVE/docs/reproduction/internal_dynamics_early_stop.md)，含"为什么用这个量"、盲测曲线与三 seed 验证。
- 实测曲线 [internal_dynamics_early_stop.csv](file:///g:/GitHub/Latent_Style/WEAVE/docs/reproduction/internal_dynamics_early_stop.csv) 与文档一致：epoch 4 `gate_delta +0.00371`、`rho 0.1634`（上轮 1.0868），`stop_requested=1`。
- 三 seed 选中 epoch：42→4（DINO-S 0.4915）、7→4（0.4910，`internal_dynamics_seed7_eval.json`）、123→3（0.4862）。
- **已知问题（文档已披露）**：① 初版绝对阈值 `rho<=1` 对 seed 7 / probe batch 8 失效（起点已在 1 以下），故改成相对 0.65；② seed 123 提前一轮，选出的 DINO-S 比 seed 42 低 0.005；③ `remaining_ablation_plan_2026-07-16.md#L29-L33` 明确它只是**架构相关**的判据，不能当架构无关收敛定理。
- **实现与文档不一致（D 类）**：文档"Diagnostic Result"把 route/head 梯度比列为事件之一，但代码判据只用 gate_delta + shared ratio；route/head 只写日志。
- **"开而无效"风险**：若模型没建 `target_latent_hf_subband_delta_lh/hl`（当前 `config.json` 未启用 target-HF subband，[model.py#L682-L684](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L682-L684) 置 `None`），`target_hf_gate_mean` 恒返回 0（[L92-L94](file:///g:/GitHub/Latent_Style/WEAVE/internal_dynamics.py#L92-L94)）→ `gate_delta=0` → 判据永不成立，且不会报错。

### 6.5 与"记忆"最接近的其他三处（都不是早停）

1. **门控均值**：训练日志里的 `last_debug["target_latent_hf_subband_delta_gate_mean"]`（[model.py#L1168-L1175](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1168-L1175)）——和早停用的 gate_mean 同源，但不参与停止。
2. **能量/泄漏比**：`fiber_energy_ratio`、`low_freq_leak` 只作为 epoch 指标占位与记录（[trainer.py#L1981-L1982](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L1981-L1982)、[utils/training.py#L225-L226](file:///g:/GitHub/Latent_Style/WEAVE/utils/training.py#L225-L226)），**具体计算位置未定位（未验证）**，且不驱动停止。
3. **数据门禁**：`StyleTargetScreener` 的 `content_keep` / `reference_copy` 阈值是**伪目标准入过滤**（[b1_style_operator.py#L757-L810](file:///g:/GitHub/Latent_Style/WEAVE/b1_style_operator.py#L757-L810)），作用在样本级、不早停训练。
