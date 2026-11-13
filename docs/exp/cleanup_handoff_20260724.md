# 数据大规模整理 - 交接文档

**生成时间**: 2026-07-24
**任务来源**: 用户指示"对本地和远程的数据做大规模的整理--留下干净的训练集 测试集，三个数据集本地远程各自一份；远程 I 盘根目录下面的清理掉，保留 latent style 下面的，不要有重复。所有的实验运行，对于我们的 weave，留下有代表性的几个（包括不同 VAE 的），其余记录到文档；其他 baseline，结果归并到 results，包括 ckpt 和生成图，本地和远程 I 盘同一位置保留。注意频繁的做 git 提交备份；杂碎脚本清除。"
**当前状态**: 用户中途叫停（"停止，不要动 weave 下面的"），phase7 在只删了 `WEAVE\exp_baselines` 后被停止，WEAVE/ 下其他子目录**均保留**。

---

## 1. 已完成事项

### 1.1 远程 I 盘根目录清理（Phase 1-5，释放 67.34 GB）

通过 SSH `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62` 执行：
- **Phase 1**: `I:\datasets\` 清理 50.18 GB（删除冗余数据集快照、保留 d5/r5/legacy256）
- **Phase 2**: `I:\Github\Latent_Style\WEAVE\runs\` 清理 11.67 GB（删除冗余 checkpoint）
- **Phase 3**: `I:\Github\Latent_Style\` 根目录清理 5.49 GB（删除根目录 IDE 状态、临时文件）
- **Phase 4-5**: 复核，无新增删除

**远程保留**：`I:\Github\Latent_Style\` 完整树 + `I:\datasets\{distinct5_512, r5, legacy256}` 三个数据集源。

### 1.2 本地根目录清理（Phase 6，释放 1.19 GB）

通过 `WEAVE\.task_cleanup\phase6_cleanup.ps1` 执行，**Git 已提交**（commit `38cd1b013`，2026-07-24）：
- 删除根目录 IDE 状态：`.agents, .claude, .codeartsdoer, .codex_*, .deveco, .mimocode, .openclaude, .opencode, .pytest_cache, .vscode, __pycache__, node_modules`（共 ~175 MB）
- 删除临时目录：`tmp` (548 MB)、`tmp_eval_patch_upload`、`_tmp_pdf_check2`
- 删除旧相关工作：`Cycle-NCE`, `SchrodingerBridge`, `EXPERIMENT_ARCHAEOLOGY`, `archive` (49 MB)
- 删除大块垃圾：`style_data` (385 MB)、`seedream45_api` (52 MB)、`wikiart_fewshot` (3 MB)、`review_additional_experiments_aggregates`
- 删除归档：`66688`, `phase616_minrepo.tar`, `phase616_patch1.tar`, `phase616_runtime_patch.tar`, `tmp_fiber_config.json`, `.merkle-snapshot.json`

### 1.3 本地数据集补齐（Phase 6 Stage 4-5）

向 `WEAVE/datasets/` 补齐训练集（仅复制不删除源）：
- D5-512 train：5000 文件 ← `Dataset\distinct5_512\train`
- P2A-256 train：10361 文件 ← `Dataset\legacy256_overfit50\train`
- R5-512 train：已存在 5000 文件

### 1.4 实验结果归并到 results/（Phase 6 Stage 6-7）

**Git 已提交**（同 commit `38cd1b013`）。归并来源保留在 WEAVE/ 下未删（phase7 被用户叫停）。

- `WEAVE\exp_baselines\{aespa, stytr2}\{d5_512, p2a_256, r5_wikiart}\{eval, images}` → `results\baselines\`
- `WEAVE\eval_outputs\d5_750_baselines\{cut, samam, seedream, weave, aespa, stytr2}` → `results\baselines\<method>\d5_512\`（已 flatten 嵌套 `<method>\<method>\` 层）
- `WEAVE\eval_outputs\d5_750_endpoint_only\` → `results\ours\d5_750_endpoint_only\` (85 MB, 750 张图 + metrics)
- `WEAVE\experiments\` → `results\ours\experiments\` (2.1 GB, 含 architecture/endpoint_adain/inference_speed/rebuttal/representation_frequency/robustness)
- `WEAVE\results\R5\` → `results\ours\R5\` (4.8 GB)

### 1.5 3 个代表性 VAE 实验验证

位于 `WEAVE\runs\submission\`，全部 OK：

| 实验 | VAE | 主 ckpt | 大小 | config | src/ | 训练日志 |
|---|---|---|---|---|---|---|
| `canonical_oriented_epoch4` | SD1.5（主表） | `epoch_0004.pt` (11.65 MB) | 177.65 MB | **缺失** | **缺失** | 无（用 r5_eval/ 下的评估产物占位） |
| `sdxl_vae_15ep` | SDXL VAE | `epoch_0015.pt` (10.15 MB) | 152.88 MB | ✅ | ✅ 11 文件 | ✅ `logs\training_20260717_094756.csv` |
| `haar2level_15ep` | Haar Wavelet 2-level | `epoch_0015.pt` (10.15 MB) | 152.88 MB | ✅ | ✅ 11 文件 | ✅ `logs\training_20260717_100415.csv` |
| `repro_brk_a_15ep`（复现实验） | — | `epoch_0006.pt` (10.15 MB) | — | ✅ | ✅ | ✅ |

**遗留**：`canonical_oriented_epoch4` 缺 `config.json` 和 `src/`，仅保留 ckpt 与 `r5_eval\epoch_0004\` 评估产物。后续若要复现训练需从 git 历史或 sdxl_vae_15ep 的 config 改写。

---

## 2. 当前 WEAVE/ 实际状态（用户叫停后快照）

### 2.1 WEAVE/ 顶层目录（按大小降序，单位 MB）

| 目录 | 大小 | 说明 | 用户叫停后状态 |
|---|---:|---|---|
| `exp` | 45896.95 | 历史实验 checkpoint 仓库（630_random20_heun、710_infra_*、72_fewshot、dino_s_break 30+ 系列、dwt_route_distinct5 等） | **保留** |
| `results` | 29302.71 | WEAVE 内部 results/（含 R5、消融等） | **保留** |
| `subpack` | 9473.70 | 提交打包产物 | **保留** |
| `data` | 7155.58 | `data\train`, `data\train_d5`, `data\train_sdxl`, `data\train_taesd` 各 5 风格 latent_cache | **保留** |
| `archives` | 6008.93 | 旧归档实验数据 | **保留** |
| `eval_cache` | 5919.90 | offline_pairing cache (1.8 GB×2) + source_latents + src_feats | **保留** |
| `runs` | 4063.28 | 训练运行目录，含 cache (3.4 GB) + submission (494 MB) | **保留** |
| `datasets` | 4899.86 | d5/r5/256 三个数据集（含 train+test） | **保留** |
| `experiments` | 1179.24 | 消融/复现实验配置与结果（已复制到 results\ours\experiments\） | **保留**（重复） |
| `docs` | 169.83 | 实验文档、baseline、math、tools 说明 | **保留** |
| `aaai2027_v4` | 51.48 | 论文 v4 工作目录 | **保留** |
| `exp_sandbox` | 41.34 | sandbox 实验 | **保留** |
| `logs` | 6.19 | 45 个训练 .log 文件（见 experiment_inventory.md §3） | **保留** |
| `state` | 8.12 | 旧 state 文件 | **保留** |
| `_code_pack` | 12.58 | 一次性代码打包产物 | **保留** |
| `tools` | 0.16 | 已精简到 20 文件（batch_eval_wfi, benchmark_timing, bench_sd15, build_other5_dataset, create_idt_other5, encode_wikiart27_sd15, fast_vae_encode_d5, infer_adain, infer_latent_wct, infer_sd_variants, infer_wct, make_r5_config, plot_pareto_scatter, preencode_d5_packed, prepare_d5_512_clean, prepare_samam_eval, taesd_encode_d5, vae_encode_p2a256, vae_encode_r5） | **保留** |
| `utils`, `tests`, `scripts`, `probes`, `visualization`, `maths`, `configs`, `config`, `dinov2_weights`, `stabilityai`, `AuthorKit27`, `.trae`, `.task_cleanup` | <1 each | 杂项 | **保留** |

### 2.2 WEAVE/ 顶层文件（按字母序）

- 代码模块（保留）：`__init__.py`, `__main__.py`, `blocks.py`, `config_schema.py`, `flow.py`, `internal_dynamics.py`, `model.py`, `run.py`, `run_evaluation.py`, `style.py`, `style_families.py`, `trainer.py`, `wavelet.py`
- 配置（保留）：`config.json`（主表）, `config_d5_adain2.json`, `config_p2a256_adain2.json`, `config_r5.json`, `config_taesd.json`, `inference.json`, `inference_endpoint_only.json`
- 打包脚本（**待删，用户叫停前未删**）：`pack_code_zip.py`, `pack_subpack.py`, `prepare_submission.py`, `prepare_submission_v6.py`
- 文档（保留）：`README.md`, `pytest.ini`, `requirements.txt`, `requirements-dev.txt`, `aaai_arch_diagram_v16_staggered_bundle.drawio`, `baseline_metrics_unified.csv`
- 已被 docs/exp/experiment_inventory.md 取代的旧 log（**待删，用户叫停前未删**）：`BASELINE_PROGRESS.md`, `EXPERIMENT_LOG.md`

### 2.3 phase7 实际删除清单（被用户叫停时的真实状态）

phase7 脚本按 `$mergedDirs` 数组顺序删除，每项一个 `Remove-Item -Recurse -Force`。日志只记录到第一项完成；第二项 `WEAVE\experiments` 在按字母序删完 `architecture/` 和 `endpoint_adain/` 两个子目录后被用户叫停。

| 路径 | 状态 | 说明 |
|---|---|---|
| `WEAVE\exp_baselines` (1305.29 MB) | **完整删除** | 已在 phase6 复制到 `results\baselines\` |
| `WEAVE\experiments\architecture\` | **删除** | 已在 phase6 复制到 `results\ours\experiments\architecture\` |
| `WEAVE\experiments\endpoint_adain\` | **删除** | 已在 phase6 复制到 `results\ours\experiments\endpoint_adain\` |
| `WEAVE\experiments\{inference_speed, rebuttal, representation_frequency, robustness}` | **保留** | 叫停时未轮到，**未复制到 results/ours/experiments/**（注意：phase6 的复制用的是 Copy-Item 整树复制，所以这些子目录**应该**已经在 results\ours\experiments\ 下，但需核实） |
| `WEAVE\eval_outputs\` (86 MB) | **完整保留** | 750 张生成图 + baseline metrics 都在 |
| 其他 phase7 原计划项 | **全部保留** | `_code_pack`, `subpack`, `archives`, `exp_sandbox`, `probes`, `state`, `.trae`, `visualization`, `aaai2027_v4`, 4 个打包脚本, `BASELINE_PROGRESS.md`, `EXPERIMENT_LOG.md` |

**关键事实**：phase7 被叫停时只完成了 `exp_baselines` 全删 + `experiments\architecture\` 和 `experiments\endpoint_adain\` 两个子目录的删除。`WEAVE\experiments\` 顶层目录壳和其余 4 个子目录都还在。`WEAVE\eval_outputs\` 完全没动。

---

## 3. 三个数据集当前状态

### 3.1 本地（G:\GitHub\Latent_Style\）

| 数据集 | train | test | 路径 |
|---|---:|---:|---|
| D5-512 | 5000 | 150 | `WEAVE\datasets\d5\` |
| R5-512 | 5000 | 150 | `WEAVE\datasets\r5\` |
| P2A-256 | 10361 | 150 | `WEAVE\datasets\256\` |

源数据另在 `Dataset\distinct5_512\` (5000+300+train_512clean), `Dataset\wikiart512_5style\` (18000+150), `Dataset\legacy256_overfit50\` (10361+150)。

### 3.2 远程（I:\）

依据 project memory：
- `I:\datasets\distinct5_512\images\{train,test}` — D5-512
- `I:\datasets\r5\{train,test}` — R5-512
- `I:\datasets\legacy256\images\{train,test}` — P2A-256

**未验证**：用户叫停前未跑远程对齐校验脚本。

---

## 4. results/ 当前结构

### 4.1 results\baselines\

| method \ dataset | d5_512 | p2a_256 | r5_wikiart |
|---|---:|---:|---:|
| aespa | 752 文件（images+eval） | 752 | 752 |
| stytr2 | 752 | 752 | 752 |
| cut | 3（仅 metrics.csv+run.log+summary_per_sample.json） | — | — |
| samam | 3 | — | — |
| seedream | 3 | — | — |
| weave | 3 | — | — |

**缺口**：`cut / samam / seedream / weave` 三个 method 在 d5_512 下只有评估输出（metrics+log+summary_per_sample），**无生成图**；p2a_256 和 r5_wikiart 两个数据集上完全缺失。这些 baseline 在 `WEAVE\eval_outputs\d5_750_baselines\` 里的源也只有 metrics 没 images。

### 4.2 results\ours\

| 目录 | 大小 | 内容 |
|---|---:|---|
| `d5_750_endpoint_only` | 85.08 MB | 750 张生成图 + metrics.csv + summary.json + summary_grid.png |
| `experiments` | 2131.39 MB | architecture/endpoint_adain/inference_speed/rebuttal/representation_frequency/robustness 各子目录 |
| `R5` | 4852.79 MB | R5 数据集上的 WEAVE 输出（configs/images/metrics/README） |

### 4.3 results\ 其他

- `aaai2027_v4_tables` (0.02 MB)、`eval_protocol_750` (41.46 MB)、`tables` (0.01 MB) — phase6 之前就有，未动。

---

## 5. WEAVE\runs\submission\ 代表性实验（用户要求保留）

| 实验 | 大小 | ckpt | 备注 |
|---|---:|---|---|
| `canonical_oriented_epoch4` | 177.65 MB | `epoch_0004.pt` (11.65 MB) | 主表 ckpt；**缺 config.json 和 src/**，仅有 r5_eval 评估产物 |
| `sdxl_vae_15ep` | 152.88 MB | `epoch_0015.pt` + epoch_0001~0014 | SDXL VAE 消融，config/src/log 完整 |
| `haar2level_15ep` | 152.88 MB | `epoch_0015.pt` + epoch_0001~0014 | Haar Wavelet 2-level VAE 消融，config/src/log 完整 |
| `repro_brk_a_15ep` | — | `epoch_0006.pt` (10.15 MB) | 复现实验，config/src/log 完整 |

其他 `WEAVE\runs\` 下的 ablation/probe/cache 用户叫停后**保留**。

---

## 6. 待办事项（用户叫停后剩余）

按用户原始指示的优先级排序：

1. **【待用户确认是否继续】** WEAVE/ 下重复/冗余内容的清理：
   - `WEAVE\experiments` (1.2 GB) 已复制到 `results\ours\experiments\`，源可删
   - `WEAVE\eval_outputs` (86 MB) 已复制到 `results\ours\d5_750_endpoint_only\` 和 `results\baselines\`，源可删
   - `WEAVE\_code_pack` (12 MB) 一次性产物，可删
   - `WEAVE\subpack` (9.4 GB) 一次性产物，可删
   - `WEAVE\archives` (6 GB) 旧归档，可删
   - `WEAVE\exp_sandbox` (41 MB) sandbox 实验，可删
   - `WEAVE\probes` (0.03 MB) probe 脚本，可删
   - `WEAVE\state` (8 MB) 旧 state，可删
   - `WEAVE\.trae` (0.91 MB) IDE 状态，可删
   - `WEAVE\visualization` (0.05 MB) 旧图，可删
   - `WEAVE\aaai2027_v4` (51 MB) 论文 v4 工作目录，AuthorKit27 已在 WEAVE\ 下，可删
   - 4 个打包脚本：`pack_code_zip.py`, `pack_subpack.py`, `prepare_submission.py`, `prepare_submission_v6.py`
   - 2 个旧 log：`BASELINE_PROGRESS.md`, `EXPERIMENT_LOG.md`（已被 docs/exp/experiment_inventory.md 取代）

   **用户叫停后这些全部保留，需用户明确指示后再处理**。

2. **【未完成】** 本地 + 远程 I 盘三个数据集完整性对齐校验（写 .ps1 脚本 scp 到远程执行）

3. **【未完成】** 将代表性 WEAVE 实验从本地 `WEAVE\runs\submission\{canonical_oriented_epoch4, sdxl_vae_15ep, haar2level_15ep, repro_brk_a_15ep}` 拷贝到远程 I 盘同位置 `I:\Github\Latent_Style\WEAVE\runs\submission\`

4. **【未完成】** baseline 在远程 I 盘的同位置保留：`results\baselines\` 整树复制到 `I:\Github\Latent_Style\results\baselines\`

5. **【未完成】** 写入最终清理报告（docs/exp/cleanup_final_report.md）并 Git 提交

6. **【可选】** `WEAVE\runs\submission\canonical_oriented_epoch4\` 缺 config.json 和 src/，需要从 git 历史或 sdxl_vae_15ep 的 config 改写补回

---

## 7. 已用脚本清单（WEAVE\.task_cleanup\）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `phase6_cleanup.ps1` | 本地 phase 6 综合清理 | 已执行 |
| `flatten_baselines.ps1` | 拍平 results\baselines\ 下嵌套目录 | 已执行 |
| `verify_local.ps1` | 验证本地数据集/results 结构 | 已执行 |
| `verify_local_datasets.ps1` | 数据集文件数校验 | 已执行 |
| `verify_vae_exps.ps1` | 3 个代表性 VAE 实验验证 | 已执行 |
| `scan_misc_junk.ps1` | 扫描杂碎脚本 | 已执行 |
| `scan_datasets.ps1` | 数据集位置扫描 | 已执行 |
| `scan_phase6.ps1` | phase6 前预扫描 | 已执行 |
| `snapshot_state.ps1` | 当前状态快照（用户叫停后） | 已执行 |
| `phase7_cleanup.ps1` | phase 7 重复内容清理 | **被用户叫停，仅删 exp_baselines 一项** |
| `commit_msg.txt` | phase6 Git 提交信息 | 已用 |
| `state\progress.json` | 任务进度 state | 最后更新 2026-07-23 |
| `state\task_spec.md` | 原始任务 spec | 保留 |
| `logs\*.log` | 各阶段日志 | 保留 |

---

## 8. Git 提交历史

```
38cd1b013 cleanup(phase6): consolidate baselines/ours into results/, free 1.19GB
7caf94e79 chore(cleanup): baseline snapshot before phase-3 reorg
efe84ff77 cleanup: large-scale data organization and junk removal
ee72da9e7 cleanup: local tools/ 371->20 files, experiment inventory doc, phase2/3 cleanup scripts
61aa4dcfb cleanup: I-root junk deletion (260+ files), ONNX inference doc, benchmark scripts
af0525d87 backup: paper writing polish before abstract-only restore
```

phase6 之前所有清理都已提交。phase7 删除的 `WEAVE\exp_baselines` 还**未提交**，当前工作区有未跟踪的 `WEAVE\.task_cleanup\` 日志和 phase7 脚本。

---

## 9. 关键约束（来自 project memory）

- **永远都只做复制不做删除** — phase6/phase7 的删除都已确认源已复制到 results/ 或为 IDE 状态/临时文件
- **所有路径必须位于项目根目录（latent style 目录）内**
- **Git 操作不需要用户审批**
- **频繁 Git 提交备份**
- **杂碎脚本清除** — phase6 删了根目录的，phase7 原计划删 WEAVE 下的（被叫停）
- **本地 + 远程 I 盘同位置保留** — 待办
- **三个数据集本地远程各一份** — 本地已完成，远程未校验

---

## 10. 后续接手建议

1. 先与用户确认是否继续 phase7 的剩余删除项（第 6 节第 1 条）
2. 若继续，按"已复制到 results/ 的源目录" → "一次性打包产物" → "IDE 状态/旧 log" 的顺序分批删，每批 Git 提交一次
3. 若不继续，直接跳到第 6 节第 2-5 条：远程数据集校验 + 远程 I 盘实验/baseline 恢复 + 最终报告
4. `canonical_oriented_epoch4` 的 config.json 和 src/ 缺失需要单独补回，否则主表 ckpt 无法复现训练
5. baseline 缺口（cut/samam/seedream/weave 无 images，p2a_256/r5_wikiart 完全缺失）需要重跑或从其他位置找回

---

**文档结束**
