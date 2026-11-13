# WEAVE 实验清单总览

**生成时间**: 2026-07-23
**整理范围**: `G:\GitHub\Latent_Style\WEAVE\` 目录下全部与实验相关的文档、日志、配置与脚本
**信息来源**: `EXPERIMENT_LOG.md`、`BASELINE_PROGRESS.md`、`config*.json`、`logs/`、`tools/`、`docs/exp/`、`docs/baseline/`、`docs/archive/`
**目的**: 为 AAAI 2027 投稿（WEAVE / FC-SB spectral-bridge 风格迁移方法）汇总所有实验脉络、配置、checkpoint 路径与保留/删除建议。

> 说明：WEAVE 目录本地为代码与文档仓，绝大多数 checkpoint 与评估产物实际位于远程 `I:\Github\Latent_Style\` 或 `I:\Github\Latent_Style\SchrodingerBridge\exp\` 下；本地 `exp/` 与 `runs/` 子目录多为空占位或仅含元数据。本文档同时标注"配置声明的本地路径"与"文档中提及的远程实际路径"。

---

## 0. 项目主线速览

- **方法名**: WEAVE（曾用名 SchrodingerBridge / FC-SB），spectral-bridge family。
- **核心思想**: 在 Stable Diffusion VAE latent 空间上做 Haar DWT 三频带 (LL / LH-HL / HH) 分解，对低频保留内容、高频做 AdaIN 统计注入，配合 I2SB flow matching 速度场预测。
- **主表指标**（来自 `config.json` `_main_table_metrics`，D5-512 数据集）:
  - CLIP-S = 0.7128, LPIPS = 0.2593, DINO-C = 0.8103, DINO-S = 0.4916
  - canonical checkpoint = `runs/submission/canonical_oriented_epoch4/epoch_0004.pt`
  - eval protocol: train `endpoint_adain_scale=1.0`，eval 用 `inference.json` override 为 `endpoint_adain_scale=2.0`
- **训练成本**（论文 Table 1，RTX 3060 12GB）: 1.04M trainable + 84M frozen VAE，1.4 min/epoch × 15 epoch ≈ 21 min。
- **推理成本**: 750 张 D5-512 ≈ 126 s（8-step Euler ODE + VAE decode）。
- **本地实测**（RTX 4070 Laptop，B5 报告）: 0.874M trainable，0.535 GFLOPs/forward，81 ms/img（latent only）。
- **远程执行主机**: `ssh -p 2222 administrator@100.115.18.62`（Windows + RTX 3060 12GB），命令须用绝对 python 路径 + `schtasks` 启动长任务。

---

## 1. 数据集

| 简称 | 远程路径 / 本地路径 | 分辨率 | 风格 | 用途 |
|---|---|---|---|---|
| **D5 / distinct5_512** | `I:/datasets/wikiart_distinct5_samam_512_classview/test` | 512 | Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e | **论文主表** + B2/B3/B4/B5 + 信号级实验 |
| **P2A / Photo2Art-256** | `I:/datasets/legacy256_overfit50/test` | 256 | cezanne, Hayao, monet, photo, vangogh | P2A-256 主表对照 |
| **R5 / Random5 (hold-out)** | `I:/datasets/wikiarts20_512_test` | 512 | Cubism, Expressionism, Pop_Art, Romanticism, Symbolism | 未见风格泛化测试 |
| **wikiarts5 (历史 256)** | 远程 `exp_256_photo2art/*_256/images`（750 张/方法） | 256 | 同 P2A | 历史 SaMam 256 baseline |
| **D5 latents** | `G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema/wikiart_distinct5_samam_512_latents_ema/train` | latent | 5 风格 × 1000 = 5000 sample | 信号级实验 A/C/D |
| **本地训练数据** | `data/train/`, `data/train_d5_taesd/`, `data/train_p2a/`, `data/train_r5/` | 512 / 256 / 512 | 同上 | 各 config.json `data_root` |

测试集命名约定: `{{src_style}}__{{src_stem}}__to__{{tgt_style}}.png`（部分方法用 `<src>_<src>__<work>_to_<tgt>.png`）。

---

## 2. 配置文件清单

| 配置文件 | 用途 | 关键改动 | save_dir | ablation.name | 主表对应 |
|---|---|---|---|---|---|
| `config.json` | Canonical 训练 + 主表 | `endpoint_adain_scale=1.0`（train），eval=2.0；batch=96；num_epochs=15；full_eval_each_epoch=false | `runs/submission/repro_brk_a_15ep` | `brk_a_ll03_15ep` | ✅ WEAVE 主表 |
| `config_r5.json` | R5 hold-out 5 风格 retrain | batch=8；data_root=`data/train_r5`；style_subdirs=Cubism/Expressionism/Pop_Art/Romanticism/Symbolism；test_image_dir=`datasets/r5/test`；full_eval_each_epoch=true | `runs/r5/weave_r5` | `weave_r5_15ep` | R5 主表 |
| `config_taesd.json` | TAESD VAE 替换实验 | `latent_scale_factor=1.0`（不再用 0.18215）；`latent_cache_mode=off`；`full_eval.vae_model=taesd`；data_root=`data/train_d5_taesd` | `runs/taesd/brk_a_15ep` | `taesd_brk_a_15ep` | 消融 |
| `config_d5_adain2.json` | D5 retrain，train 阶段就用 adain=2.0 | base=config.json；`endpoint_adain_scale=2.0`；num_epochs=15 | `runs/d5_adain2` | `d5_adain2` | 消融 |
| `config_p2a256_adain2.json` | P2A-256 训练 | base=config.json；data_root=`data/train_p2a`；style_subdirs=cezanne/Hayao/monet/photo/vangogh；max_images_per_style=1000；endpoint_adain_scale=2.0 | `runs/p2a256_adain2` | `p2a256_adain2` | P2A 主表 |
| `inference.json` | Eval override | `endpoint_adain_scale=2.0`；`num_steps=8`；`step_size=1.0`；`style_strength=1.0`；max_src_samples=30；max_ref_compare=16 | — | — | 评估协议 |
| `_code_pack/config.json`, `subpack/config.json`, `subpack/config_r5.json` | 子代码包副本 | 同主 config | — | — | 提交打包用 |

---

## 3. 训练日志清单（`WEAVE/logs/`）

共 45 个 `.log` 文件，按实验系列分组：

### 3.1 Phase 4H 系列（D0 → D2 frontier，~7 epoch 消融）
- `4h1c_train.log`, `4h1d_train.log`, `4h1e_train.log`, `4h1f_train.log`
- `4h1g_train.log`, `4h1g5ep_train.log`（5-epoch 版本）
- `4h2h_train.log`, `4h2i_train.log`
- `4h3f_train.log`
- `4h4e_train.log`, `4h4f_train.log`, `4h4g_train.log`
- `4h5e_train.log`, `4h5f_train.log`
- `4h7d_train.log`

### 3.2 Phase 4I 系列（结构性突破 + Heun + cosine）
- `4i1a_train.log`, `4i1d_train.log`
- `4i2a_train.log`, `4i2b_train.log`
- `4i5a_train.log`, `4i5b_train.log`, `4i5c_train.log`
- `4i6a_train.log`
- `4i7a_train.log`, `4i7b_train.log`（远程最终 SOTA Heun+cosine_a085_5ep）

### 3.3 Brk_aa 系列（ablation aggressive round）
- `brk_aa_aug24_train.log`, `brk_aa_aug35_train.log`

### 3.4 Semantic-R 系列（语义路由宽度消融）
- `sem_r4_musiq.log`, `sem_r8_soft_musiq.log`, `sem_r8_soft_train.log`
- `sem_r16_musiq.log`, `sem_r16_train.log`
- `sem_r32_train.log`

### 3.5 SWD-CM 系列（spectral-wasserstein content modulation）
- `swd_cm_cons5_run.log`（对应 `exp/swd_cm_cons5/`）
- `swd_cm_sweep.log`

### 3.6 T 系列（local_t11 / T2x local SOTA 探索）
- `t11_long30ep_train.log`（local SOTA T11 长训 30 epoch）
- `t21_train.log`, `t22_train.log`, `t26_train.log`

### 3.7 StyleAligned baseline（D5 上的两个风格子集）
- `stylealigned_rococo.log`, `stylealigned_ukiyoe.log`

### 3.8 R5 评估
- `weave_r5_eval.log`（R5 hold-out 评估）

### 3.9 其他评估/编码日志
- `endpoint_only_eval.log`（D5 endpoint-only 评估）
- `phase4e_db2_lvl1.log`（Phase 4E DB2 Lvl1）
- `preencode_pixel256.log`（P2A-256 预编码 latents）

### 3.10 远程日志（非本地 logs/）
- 远程 `data/train_d5/` 下: `graph_b20.log/.err`, `sweep_b16.log`, `sweep_b16b.log`, `sweep_b4.log`
- 远程 `state/dino/` 下: `run.log/.err`, `run2.log/.err`, `run3.log/.err`, `run4.log/.err`, `batch_run.log`
- 本地根: `repro_eval_local.log`, `repro_r20_eval_local.log`, `t11_repro_local.log`

---

## 4. 实验脚本（`WEAVE/tools/`）

`tools/` 目录共 150+ 个脚本，按用途分组（仅列代表性脚本，完整列表见 `tools/experiments/README.md` 与 `tools/README.md`，其中前者描述 SchrodingerBridge 时期的旧 experiments/ 子目录）：

### 4.1 主训练与评估入口
| 脚本 | 作用 |
|---|---|
| `run.py`（项目根） | 训练入口 |
| `run_evaluation.py`（项目根 + utils/） | CLIP-S / LPIPS 评估 |
| `scripts/batch_eval_all.py` | 批量逐 epoch 评估（infra 修复版，~5x DINO 加速） |
| `scripts/_repro_dino_eval.py` | DINO-S / DINO-C 评估（单进程复用 DINOv2） |
| `inference.json` + `run.py --config_override` | 推理 override 协议 |

### 4.2 B2-B5 论文图表生成
| 脚本 | 实验 | 输出 |
|---|---|---|
| `tools/b2_qualitative_grid.py` | B2 非精选定性网格 | `exp/b2_qualitative/grid_8x9.png` (300 DPI) + `metadata.json` |
| `tools/b3_failure_cases.py` | B3 失败案例分析 | `exp/b3_failure_cases/grid_15x4.png` + `failure_cases.csv` + `stats.json` |
| `tools/b4_reference_stability.py` | B4 reference pool 稳定性 | `exp/b4_reference_stability/boxplot_ref_diversity.png` + `stats.json` + `per_pair_cosine.csv` |
| `tools/b5_compute_cost.py` | B5 计算成本分析 | `exp/b5_compute_cost/summary.json`（参数量 + FLOPs + 时延 + VRAM + 14 baseline） |

### 4.3 信号级实验（论文分频设计论证）
| 脚本 | 实验 |
|---|---|
| `tools/signal_level_exp.py` | A（谱能量不对称）+ C（频带-语义 ANOVA） |
| `tools/probe_gradient_hegemony.py` | B（梯度霸权 / LL 容量耗尽探针） |
| `tools/probe_hh_topology_leak.py` | D（HH-head 拓扑泄漏测定） |
| `tools/probe_fiber_path.py`, `tools/probe_whitening.py` | fiber / whitening 探针 |

### 4.4 Baseline 推理与评估
| 脚本 | 用途 |
|---|---|
| `tools/_run_stylealigned_remote.py` + `tools/_sa_launch.bat` | StyleAligned 远程 schtasks 启动 |
| `tools/_run_zstar_remote.py` + `tools/_zstar_launch.bat` | Z-STAR 远程推理 |
| `tools/_run_styleshot_remote.py` + `tools/_styleshot_launch.bat` + `tools/_start_styleshot.bat` | StyleShot 远程推理 |
| `tools/_run_samam_remote.py`, `tools/_run_samst_remote.py`, `tools/_run_cut_remote.py` | SaMam / SaMST / CUT 远程推理 |
| `tools/infer_sd_variants.py` | SD-Turbo / SDEdit / StyleID 推理 |
| `tools/infer_wct.py`, `tools/infer_latent_wct.py`, `tools/infer_adain.py` | classical-inf baseline 推理 |
| `tools/run_r5_baselines.py`, `tools/bench_all_3060.py`, `tools/bench_2m_baselines.py`, `tools/bench_sd15.py` | R5 / 3060 上批量 baseline benchmark |
| `tools/batch_eval_wfi.py`, `tools/eval_dino_manifest.py`, `tools/eval_plugin_metrics.py`, `tools/eval_xf_qwen_vlm.py` | WFI / DINO / 插件 / VLM 评估 |

### 4.5 ONNX / TensorRT 加速
| 脚本 | 用途 |
|---|---|
| `tools/taesd_encode_d5.py`, `tools/taesd_eval.py`, `tools/vae_encode_p2a256.py`, `tools/vae_encode_r5.py`, `tools/fast_vae_encode_d5.py`, `tools/preencode_d5_packed.py` | VAE 编码 / 预编码 |
| `tools/taesd_check.ps1`, `tools/taesd_gpu_check.ps1`, `tools/taesd_launch.ps1`, `tools/taesd_test_*.ps1` | TAESD 实验 PowerShell 包装 |
| 远程 `experiments/inference_speed/`（见 docs/exp/onnx_inference/README.md） | ONNX 导出 + TRT 编译 + 端到端 benchmark |

### 4.6 远程协调 / 监控 / 调试
| 类别 | 代表脚本 |
|---|---|
| 远程启动器 | `tools/_remote_launcher.bat/.ps1`, `tools/remote_launcher.bat/.ps1`, `tools/_sa_launch.bat`, `tools/_zstar_launch.bat`, `tools/_styleshot_launch.bat`, `tools/_watchdog_launch.bat` |
| 远程检查 | `tools/_remote_gpu_check.ps1`, `tools/_remote_path_check.ps1`, `tools/_remote_probe*.ps1`, `tools/_remote_read_logs.ps1`, `tools/remote_check*.py/.sh` |
| 配置生成 | `tools/make_r5_config.py`, `tools/gen_remote_configs.py`, `tools/_prep_cut_r5_train.py`, `tools/_train_cut_r5.py` |
| 训练检查 | `tools/check_*.py/.sh/.ps1`（30+ 个，覆盖 ckpt config / hashes / training / remote / model scale / png hash） |
| 诊断 | `tools/diag_*.py`（cut_naming / src_lookup / wct / wct_vgg19）, `tools/_debug_p2a.py`, `tools/_debug_styleshot.py`, `tools/probe_620_fog_path.py` |
| 评估汇总 | `tools/extract_metrics.py`, `tools/read_summary.py/.ps1`, `tools/read_film_summary.py`, `tools/summarize_dino_avg.py`, `tools/recompute_dino_avg.py`, `tools/_inspect_dino_avg.py`, `tools/_inspect_dino_main.py`, `tools/_inspect_r5.py` |
| 可视化 | `tools/make_dashboard.py`, `tools/plot_pareto_scatter.py`, `tools/aaai2027_v4/make_radar.py`, `tools/inline_math_drawio.py`, `tools/wrap_math_drawio.py`, `tools/scale_drawio.py`, `tools/y_histogram_fix.py`, `tools/fix_math_delims.py` |
| A3 sweep | `tools/debug_remote_a3.py`, `tools/launch_remote_a3.py`, `tools/launch_remote_a3_v2.py`, `tools/remote_check_a3.py` |
| R5 baseline | `tools/run_r5_baselines.py`, `tools/_remote_setup_r5.ps1`, `tools/fix_r5_manifest.py`, `tools/sample_r5_train.py` |
| Smoke 测试 | `tools/smoke_blockmask.py`, `tools/smoke_db2.py`, `tools/r4a_quick_test.py`, `tools/r4a_extended_test.py`, `tools/launch_smoke.sh`, `tools/run_smoke_test.sh`, `tools/test_detached.py` |
| 消融批跑 | `tools/run_ablation_batch.py`, `tools/scan_loss_landscape.py`, `tools/signal_level_exp.py` |

### 4.7 提交打包
| 脚本 | 用途 |
|---|---|
| `pack_code_zip.py`, `pack_subpack.py` | 打包 `_code_pack/` 与 `subpack/` |
| `prepare_submission.py`, `prepare_submission_v6.py` | 提交准备 |

---

## 5. 主要实验清单

### 5.1 主表 / 训练实验（按 config）

| # | 实验名 | 配置 | 数据集 | 目的 | 主表对应 | 状态 | Checkpoint 路径 |
|---|---|---|---|---|---|---|---|
| **T-Main** | `brk_a_ll03_15ep` | `config.json` | D5-512 | 论文主表，15 epoch + LL partial AdaIN α=0.3 | ✅ 主表（CLIP-S=0.7128, DINO-S=0.4916） | ✅ 完成 | 本地占位 `runs/submission/repro_brk_a_15ep/`；canonical eval ckpt = `runs/submission/canonical_oriented_epoch4/epoch_0004.pt`；远程实际目录 `I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep/epoch_0001~0015.pt` |
| **T-R5** | `weave_r5_15ep` | `config_r5.json` | R5-512 | R5 hold-out 风格泛化测试，batch=8 | ✅ R5 主表 | ✅ 完成 | 本地占位 `runs/r5/weave_r5/`；远程 `I:/datasets/wikiarts20_512_test` 评估 |
| **T-P2A** | `p2a256_adain2` | `config_p2a256_adain2.json` | P2A-256 | P2A-256 协议对照 | ✅ P2A 主表 | ✅ 完成 | `runs/p2a256_adain2/` |
| **T-D5-AdaIN2** | `d5_adain2` | `config_d5_adain2.json` | D5-512 | 训练阶段直接用 adain=2.0（消融） | 消融 | ✅ 完成 | `runs/d5_adain2/` |
| **T-TAESD** | `taesd_brk_a_15ep` | `config_taesd.json` | D5-512 | TAESD VAE 替换 SD1.5 VAE 的消融 | 消融 | ✅ 完成 | `runs/taesd/brk_a_15ep/` |

### 5.2 B 系列论文实验（B2-B5，2026-07-21 生成）

所有 B 系列共用 canonical checkpoint `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` 与 `inference.json` override (`endpoint_adain_scale=2.0`)，本地硬件 RTX 4070 Laptop GPU 8GB。

| # | 实验 | 脚本 | 目的 | 输出文件 | 状态 |
|---|---|---|---|---|---|
| **B2** | 非精选定性网格 | `tools/b2_qualitative_grid.py` | 8 对随机抽样（seed=2026）× 9 方法网格，证明非 cherry-pick | `exp/b2_qualitative/grid_8x9.png` (2412×2443, 300 DPI) + `metadata.json` | ✅ |
| **B3** | 失败案例分析 | `tools/b3_failure_cases.py` | 按 LPIPS / CLIP-S / DINO-S 各取前 5 = 15 个失败案例，诚实展示局限 | `exp/b3_failure_cases/grid_15x4.png` + `failure_cases.csv` + `stats.json` | ✅ |
| **B4** | Reference 稳定性 | `tools/b4_reference_stability.py` | reference pool 多样性 + CLIP-S 方差代理（保守上界） | `exp/b4_reference_stability/boxplot_ref_diversity.png` + `stats.json` + `per_pair_cosine.csv` | ✅ |
| **B5** | 计算成本分析 | `tools/b5_compute_cost.py` | 参数量分解 + FLOPs + 推理时延 + VRAM + 14 baseline 对照 | `exp/b5_compute_cost/summary.json` | ✅ |

### 5.3 信号级实验（论文分频设计论证，2026-07-19）

数据: `Dataset/distinct5_512_latents_ema/wikiart_distinct5_samam_512_latents_ema/train`（5 风格 × 1000 latent = 5000 sample）。文档: `docs/exp/signal_level/README.md`。

| # | 实验 | 脚本 | 论点 | 关键发现 | 状态 |
|---|---|---|---|---|---|
| **A** | 谱能量不对称 | `tools/signal_level_exp.py` | 分频**有用**（前提存在） | E_LL=65.9%, E_HH=10.1%, LL/HH=6.5× | ✅ |
| **B** | 梯度探针（容量耗尽） | `tools/probe_gradient_hegemony.py` | 分频重加权**必要** | w10 LL loss 坍塌至 3e-08，w03 ratio=0.367 健康 | ✅ |
| **C** | 频带-语义对齐 | `tools/signal_level_exp.py` | Haar 分解**正确** | F_HH=123.43 最高, F_LL=95.79（LL 也含风格），反向支撑 α=0.3 LL 部分风格化 | ✅ |
| **D** | HH 拓扑泄漏 | `tools/probe_hh_topology_leak.py` | AdaIN 注入**正确** | HH-head v_hh corr=0.21±0.13 不稳定，AdaIN corr=0.024±0.012 稳定（数学可证目标无关性） | ✅ |

> Paper-facing audit（2026-07-19）注: 原始数值有效，但 D 中"8.7× leakage factor"不可严格相除，论文已采用克制解释。

### 5.4 Baseline 实验（12 + 3 个新增，`docs/baseline/README.md`）

**统一协议**: HF transformers CLIP ViT-B/32 + LPIPS Alex + 750 pairs（CUT 为 745）+ distinct5_512。

| # | Baseline | 类别 | CLIP-S | LPIPS | 训练/调用时间 | 状态 | 评估文件 |
|---|---|---|---|---|---|---|---|
| F001 | Identity | baseline | 0.6933 | 0.0000 | 0 | ✅ | `baseline_reeval/identity/summary.json` |
| F002 | AdaIN | classical-inf | 0.6679 | 0.7425 | 0 | ✅ | `baseline_reeval/adain_v32k/summary.json` |
| F019 | WCT (VGG19) | classical-inf | 0.7063 | 0.6348 | 0 | ✅ | `baseline_v2/eval/wct_vgg19_summary.json` |
| F007 | SD-Turbo | diffusion-inf | 0.6933 | 0.0033 | 0 | ✅ | `baseline_reeval/sdturbo/summary.json` |
| F005 | SDEdit s=0.35 | diffusion-sweep | 0.7797 | 0.4508 | 0 | ✅ | `baseline_reeval/sdedit_035/summary.json` |
| F006 | SDEdit s=0.40 | diffusion-sweep | 0.7934 | 0.4826 | 0 | ✅ | `baseline_reeval/sdedit_040/summary.json` |
| F008 | StyleID | diffusion-inf | **0.8223** | 0.5523 | 0 | ✅ | 远程 `exp_baselines/styleid/` |
| F014 | CUT | gan-train | 0.7137 | 0.3743 | 322.6 min | ⚠️ 缺 train.log | 远程 `exp_baselines/cut/`（n=745） |
| F011 | SaMST | mamba-train | 0.6183 | 0.7490 | 39.5 min | ✅ e5/e15 plateau | `baseline_reeval/samst_stepalign40/summary.json` |
| F020 | SaMam | mamba-train | 0.5816 | **0.2434** | ~436 min | ✅ 81 ckpt 完整曲线 | `tools/samam_distinct5_scratch/curve_metrics_hf.csv` step 20000 |
| F021 | Seedream 4.5 | commercial-API | 0.7198 | 0.4767 | API | ✅ | 远程 `exp_baselines/seedream45_api/` |
| — | Latent-WCT | classical-inf | — | — | free | ✅（论文新增） | 18 s/750 张 |
| — | **StyleAligned** | diffusion-inf（SD1.5） | TBD | TBD | free | 🟡 P2A+R5 待跑（schtasks 已重启） | 远程 `exp/baseline_stylealigned/{photo2art256,random5}/images/`，D5 已完成 CLIP-S=0.8739 / LPIPS=0.7825 |
| — | **Z-STAR** | diffusion-inf（CVPR 2024） | TBD | TBD | free | ⬜ 未开始 | 远程 `tools/zstar/` 待建 |
| — | **StyleShot** | diffusion-inf（AAAI 2025） | TBD | TBD | free | ⬜ 未开始 | 远程 `tools/styleshot/` 待建 |

**主表 Pareto 关键点**（WEAVE 本身，来自 `docs/archive/exp/README.md` v4）:

| 配置 | CLIP-S | LPIPS | 角色 | Checkpoint |
|---|---|---|---|---|
| **4F.1** | **0.7319** | 0.3428 | 远程 SOTA（无 DWT route） | `FCSB/phase4/630_phase4f_lvl3/epoch_0003.pt` |
| 4I.7b | 0.7272 | 0.3218 | 远程最终 SOTA Heun+cosine | `FCSB/phase4/630_phase4i7b_cosine_heun_a085_5ep/epoch_0005.pt` |
| 4I.11 | 0.7250 | 0.3129 | 远程 per-subband WCT | （远程） |
| 4J.1 | 0.7226 | 0.3068 | DWT route 起点（本地） | `FCSB/phase4/630_phase4j1_dwt_route/epoch_0005.pt` |
| **T11** | **0.7213** | **0.2868** | **本地 SOTA（FC-SB 主报告）** | `FCSB/local_t/630_local_t11_stochastic_dwt_p08/epoch_0005.pt` |
| T10 | 0.7083 | 0.2480 | lpips BEST 极值对照 | `FCSB/local_t/630_local_t10_stochastic_dwt/epoch_0005.pt` |
| brk_a（主表） | 0.7128 | 0.2593 | 主表 canonical | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |

> 注: `FCSB/phase4/...` 路径来自旧 `exp/README.md` 描述的 2026-07-03 重组结构。当前 `exp/` 目录已再次重组（见 §5.5），这些路径在 `exp/` 下已不可直接定位，但远程 `I:\Github\Latent_Style\SchrodingerBridge\exp\` 上仍可按以下对应找到。

### 5.5 当前 `exp/` 目录结构（占位）

当前 `WEAVE/exp/` 下子目录均为空占位，反映最新一次重组（早于 2026-07-21）：

```
exp/
├── 256/             # P2A-256 历史占位
├── 72_fewshot/      # 72 系列 fewshot 实验占位（含 run_all.py）
├── a3_sweep/        # A3 sweep（含 beta_2p0.log, summary.csv）
├── fewshot6/        # fewshot6 实验占位
├── fewshot8_new3/   # fewshot8_new3 实验占位（含 src/run.py）
├── idt_baseline/    # Identity baseline（含 dino.json, metrics.csv）
├── sem_r8_strict/   # 语义路由 r8 strict（含 src/run.py）
├── sem_r8_v2/       # 语义路由 r8 v2（含 src/{model,run,trainer}.py + config.json）
├── swd_cm_cons5/    # SWD-CM cons5（含 src/run.py + config.json）
├── swd_cm_hh_r8/    # SWD-CM hh r8（含 src/run.py + config.json）
├── swd_cm_sem_r8/   # SWD-CM sem r8（含 src/run.py）
├── swd_cm_w14/      # SWD-CM w14（含 src/{model,run}.py + config.json）
├── swd_cm_w16/      # SWD-CM w16（含 src/{model,run}.py + config.json）
└── README.md        # 已过时（描述旧 FCSB 结构，与当前实际目录不符）
```

> ⚠️ `exp/README.md` 描述的 `FCSB/{early,phase4,local_t}/` + `baseline/{reeval,images,v2}/` + `wiki5/` + `legacy/` 结构已**不复存在**于当前 `exp/` 物理目录。该 README 仅作为历史脉络参考（见 §5.4 Pareto 表中 FCSB 路径的出处）。

### 5.6 历史实验脉络（来自 `EXPERIMENT_LOG.md` + `docs/archive/ablation_log.md`）

#### 5.6.1 Phase 2 A/B Test（早期，`ablation_log.md`）
3 个消融（8 epoch / 实验）:
- Exp 1: `skip_routing_mode = "normalized"` → CLIP-S=0.635, LPIPS=0.303（与 baseline 几乎无差异，**skip routing 无贡献**）
- Exp 2: `w_curvature = 1.0` → 未填表（结果未记录）
- Exp 3: `output_moment_match = true` → 未填表

#### 5.6.2 D0 → D2 Frontier（2026-05-16）
基于 `ablation_destructive_7epoch/combined_summary/combined_750_with_destructive_ablations.csv`:
- `D0_full_correct_7ep`: CLIP-S=0.7014, LPIPS=0.4593, 290.65 s/epoch
- `D1_no_terminal_swd`: CLIP-S=0.6708（**Terminal SWD 必要**）
- `D2_no_kinetic`: CLIP-S=0.7159, LPIPS=0.6375（**风格↑但内容崩塌**）
- `D10_micro_hf_swd`: CLIP-S=0.6989（**micro HF SWD 方向错误**）
- `SaMST strict`: CLIP-S=0.7194, LPIPS=0.4664

**重启点**: `S-add__K-1_C-0_W-20_Col-0/epoch_0007.pt`（D0/K1-style OMF branch），首选调 `w_kinetic`。

#### 5.6.3 Phase 4H / 4I 系列（远程 RTX 3060）
详见 §3.1 / §3.2 日志清单。关键产物:
- 4F.1 (`630_phase4f_lvl3`): 远程 SOTA CLIP-S=0.7319
- 4I.2b (`630_phase4i2b_sota_heun_5ep`): Heun 结构性突破
- 4I.7b (`630_phase4i7b_cosine_heun_a085_5ep`): 远程最终 SOTA
- 4J.1 (`630_phase4j1_dwt_route`): DWT route 起点

#### 5.6.4 Local T 系列（本地 SOTA）
- T11 (`630_local_t11_stochastic_dwt_p08`): **本地 SOTA** CLIP-S=0.7213, LPIPS=0.2868（Stochastic DWT p=0.8 + w_ll=0.0）
- T10 (`630_local_t10_stochastic_dwt`): LPIPS BEST=0.2480
- T21/T22/T26: 见 `logs/t21_train.log`, `t22_train.log`, `t26_train.log`
- T11 长训 30 epoch: `logs/t11_long30ep_train.log`

#### 5.6.5 15 Epoch 模块有效性验证（2026-07-14, `docs/exp/brk_a_15ep_handoff.md`）
- **训练**: brk_a 主表配置，15 epoch，~23 min（含每 epoch eval），远程 `I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep/epoch_0001~0015.pt`
- **CLIP-S 峰值**: ep11 / ep13 = 0.7186（adain=1.0）
- **DINO-S 峰值**: ep7 = 0.4837（adain=1.0），综合最佳 `epoch_0007.pt`
- **Probe 实验排名**（DINO-S）:
  1. `target_hf_spatial` 0.4901（内容崩塌，伪提升）
  2. `target_hf_subband` 0.4886 ✅ 有效 +0.003
  3. `target_hf_subband+texture` 0.4884 ✅
  4. `target_hf_subband_head` 0.4873 ✅
  5. `target_hf_head (strong)` 0.4870 ✅ +0.002
  6. `target_hf_spatial_energy` 0.4861（边际）
  7. `target_hf_texture` 0.4860（边际）
  8. `target_hf_hybrid` 0.4858（≈baseline）
  — brk_a baseline 0.4859（adain=2.0）
  9-12. `target_hf_delta_ft15` 0.4825-0.4850（低于 baseline）
  13-17. **`target_latent_token_fusion` 0.4809-0.4817（负面 -0.005）**

#### 5.6.6 信号级实验（2026-07-19，见 §5.3）

### 5.7 ONNX / TensorRT 推理加速实验（`docs/exp/onnx_inference/README.md`）

- **目标**: VAE Decoder 导出 ONNX → TRT FP16 加速
- **环境**: RTX 3060 12GB, CUDA 12.8, TRT 10.x, ONNX 1.17, FP16
- **导出文件**: `data/onnx_exported/{encoder,decoder}_b{8,16}.onnx`
- **TRT 编译**: b=8 ws=2GB FP16，~15 min；b=16 ws=4GB OOM
- **端到端 benchmark**（750 图，b=8, TRT ws=2GB）:
  - Stage 1+2（VAE Encoder + WEAVE PyTorch bf16）= 14.0 s
  - Stage 3（VAE Decoder TRT FP16）= **58.98 s**
  - 总计 = **73.0 s**（97.3 ms/img）
- **加速比**: TRT 2.19× vs PyTorch fp32 baseline（129.3 s）；torch.compile 1.71×
- **关键结论**: Decoder 是瓶颈（80.8%），RTX 3060 已到极限，推荐 b=8/TRT FP16/ws=2GB

### 5.8 R5-512 推理时序测量（2026-07-21, `docs/exp/r5_512_infer_timing.md`）

| Method | Total (s) | sec/img | 论文 Infer 列 | 修正动作 |
|---|---|---|---|---|
| WEAVE | 126 | 0.168 | 126s | ✅ 保持 |
| Identity / TGT | <1 | 0.001 | 0 | — |
| WCT (VGG19) | 61.4 | 0.082 | — | 不在主表 |
| AdaIN | 30.1 | 0.040 | — | 不在主表 |
| Latent-WCT | 220.1 | 0.293 | 18s | **修正为 4m** |
| SD-Turbo | 524 | 0.66 | 5m | **修正为 9m** |
| StyleID | 2383 | 3.14 | 63m | **修正为 40m** |
| StyleAligned | 4643 | 6.19 | 77m | ✅ 保持（吻合） |
| StyTR-2 | 291 | 0.388 | 38m | **修正为 5m**（论文 38m 可能含训练） |
| AesPA-Net | 252 | 0.336 | 23m | **修正为 4m**（同上） |
| Z-STAR | 1871 | 2.49 | 3h | ✅ 吻合 |
| SaMam | 4152 | 5.54 | 17.6m | ✅ 吻合 |
| SaMST | 265.9 | 0.355 | 10m | ✅ 吻合 |
| CUT | 104.5 | 0.14 | 5m | ✅ 吻合 |
| StyleShot | pending | — | 5.1h | 待测 |

### 5.9 后处理亮度对齐实验（`docs/exp/postprocess_brightness_alignment*.md`）

两个文档:
- `postprocess_brightness_alignment_report.md`
- `postprocess_brightness_alignment_deep_analysis.md`

实验目标: 评估后处理亮度对齐对 WEAVE 输出的影响（具体内容未读取，但目录存在）。

---

## 6. 代表性实验（推荐保留）

按"是否直接进入论文 / 是否唯一证据源 / 是否提供 Pareto 前沿"标准筛选:

### 6.1 论文主表与主图（必须保留）

| 实验 | 理由 | Checkpoint |
|---|---|---|
| **brk_a_ll03_15ep（canonical）** | 论文主表 WEAVE 行 | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` + 远程 `brk_a_ll03_15ep/epoch_0001~0015.pt` |
| **weave_r5_15ep** | R5 主表 | `runs/r5/weave_r5/` |
| **p2a256_adain2** | P2A-256 主表 | `runs/p2a256_adain2/` |
| **d5_adain2** | 训练阶段 adain=2.0 消融 | `runs/d5_adain2/` |
| **taesd_brk_a_15ep** | TAESD VAE 消融 | `runs/taesd/brk_a_15ep/` |
| **B2/B3/B4/B5** | 论文图 4 张 + Table 1 复现证据 | `exp/b{2,3,4,5}_*/` |
| **signal_level A/B/C/D** | 论文分频设计论证 | `docs/exp/signal_level/{results,gradient_hegemony,hh_topology_leak}.json` + `fig_*.png` |
| **ONNX/TRT 加速** | 补充材料推理速度证据 | `data/onnx_exported/` + 远程 `experiments/inference_speed/` |

### 6.2 历史 Pareto 前沿（保留 1 个代表）

| 配置 | 理由 | Checkpoint |
|---|---|---|
| **T11（local SOTA，主报告）** | 论文 FC-SB 主报告 SOTA，CLIP-S=0.7213 / LPIPS=0.2868 | `FCSB/local_t/630_local_t11_stochastic_dwt_p08/epoch_0005.pt`（远程） |
| 4I.7b（远程最终 SOTA） | 远程 Heun+cosine 突破点 | `FCSB/phase4/630_phase4i7b_cosine_heun_a085_5ep/epoch_0005.pt`（远程） |
| 4F.1（远程 SOTA 无 DWT route） | 远程 D0 frontier 最高点 | `FCSB/phase4/630_phase4f_lvl3/epoch_0003.pt`（远程） |
| 4J.1（DWT route 起点） | 本地 DWT 路线起点 | `FCSB/phase4/630_phase4j1_dwt_route/epoch_0005.pt` |
| T10（LPIPS BEST 极值对照） | 极端内容偏置对照点 | `FCSB/local_t/630_local_t10_stochastic_dwt/epoch_0005.pt` |

### 6.3 关键消融（保留作为论文消融表）

| 实验 | 理由 | 来源 |
|---|---|---|
| **D0_full_correct_7ep**（7 epoch baseline） | Phase 4 重启点基准 | `EXPERIMENT_LOG.md` |
| **D1_no_terminal_swd** | 证明 Terminal SWD 必要 | 同上 |
| **D2_no_kinetic** | 证明 kinetic 正则化必要 | 同上 |
| **target_latent_hf_subband_fusion probe** | DINO-S +0.003 有效模块（下一步改进方向） | `docs/exp/brk_a_15ep_handoff.md` |
| **w10_global_mse 梯度探针** | 证明 λ_LL=0.3 必要性 | `docs/exp/signal_level/gradient_hegemony.json` |

### 6.4 Baseline 收敛证据（必须保留）

| Baseline | 理由 | 评估文件 |
|---|---|---|
| SaMam 81 ckpt 完整曲线 | 论文 Table 1 SaMam 行来源 | `tools/samam_distinct5_scratch/curve_metrics_hf.csv`（远程 I 盘） |
| SaMST e5/e15 plateau 测试 | SaMST 收敛证据 | `baseline_reeval/samst_stepalign40/summary.json` |
| StyleID / StyleAligned / Z-STAR / StyleShot | 新增 3 baseline 推理 + 评估 | 远程 `exp_baselines/styleid/`, `exp/baseline_stylealigned/` |

---

## 7. 可删除实验

按"已被取代 / 失败探索 / 死代码验证 / 重复 sweep"分类。建议删除前在文档中保留其结论摘要。

### 7.1 Phase 2 早期消融（已被取代）

| 实验 | 删除理由 |
|---|---|
| `skip_routing_mode = "normalized"` | 与 baseline 几乎无差异（CLIP-S 0.635 vs 0.635），skip routing 已确认无贡献 |
| `w_curvature = 1.0`（未填表） | 结果未记录，无价值 |
| `output_moment_match = true`（未填表） | 结果未记录，无价值 |

### 7.2 Phase 4H/4I 中间过程（已被 T11 / 4I.7b 取代）

| 实验 | 删除理由 |
|---|---|
| 4h1c–4h7d 共 15 个（除 4F.1=4h?） | Phase 4H 系统消融已完成，仅 4F.1 入选 Pareto；其余中间配置已被 T 系列超越 |
| 4i1a, 4i1d, 4i2a, 4i5a-c, 4i6a | Phase 4I 中间配置，最终仅 4I.2b / 4I.7b 入选 |
| brk_aa_aug24 / aug35 | aggressive ablation，结论已并入 ablation_log |

### 7.3 Semantic-R 中间宽度（保留两端 + soft）

| 实验 | 删除理由 |
|---|---|
| sem_r4 / sem_r16 / sem_r32 中间值 | r8 soft 已是最佳，r4 太窄、r32 太宽无收益 |

### 7.4 SWD-CM 部分变体（保留 cons5 + sem_r8）

| 实验 | 删除理由 |
|---|---|
| swd_cm_w14 / swd_cm_w16 | w14/w16 与 w8 差异小（见 `swd_cm_sweep.log`） |
| swd_cm_hh_r8 单独 | 已并入 swd_cm_sem_r8 |

### 7.5 Few-shot 失败实验

| 实验 | 删除理由 |
|---|---|
| `exp/fewshot6/` 全部 3 个（`630_phase4j6_fewshot_popart{,_v2,_v3}`） | 全 FAIL，无主线价值 |
| `exp/fewshot8_new3/` | 占位，无完整结果 |
| `exp/72_fewshot/`（如未用于论文） | 5p1/5p2/5p3 × shot{01,06,10,30,50} 共 15 组，如未进入论文可全部归档 |

### 7.6 探针实验中确认无效的模块代码

依据 `docs/exp/brk_a_15ep_handoff.md` §4.4:

| 模块 | 判定 | 删除理由 | model.py 死代码行数 |
|---|---|---|---|
| `target_latent_token_fusion` | 负面 -0.005 | probe 确认 DINO-S 低于 baseline | ~46 行（init 488-509 + forward 837-860） |
| `intrinsic_style_cnn` | 无效（sensitivity=0.000） | probe 确认全频段零敏感度 | ~32 行（init 463-481 + forward 794-804） |
| `enable_hh_head` | 无效（HH 梯度恒为 0） | `loss_fm_spectral_hh=0.000` | ~30 行 |
| `style_adaln` | 无效（Round10-12 已禁用） | 文档明确 "Do not reintroduce" | ~10 行 |
| `semantic_self_topology_gate` | 无效（phantom metric） | model.py 无 `last_semantic_topology_attn` | — |
| `CFG (cfg_dropout_prob=0.0)` | 完全禁用 | 死代码 | ~20 行 |
| 25 个孤儿配置标志 | model.py 零引用 | 含 `spectral_ode_enabled=true`（唯一启用态孤儿，但完全不消费） | — |

> ⚠️ **不要删除**: `target_latent_hf_subband_fusion`（DINO-S +0.003）与 `target_latent_hf_head_fusion`（+0.002）是 probe 验证的有效模块，是下一步改进方向。

### 7.7 旧 exp/ README 描述的废弃结构

| 路径 | 删除理由 |
|---|---|
| `exp/FCSB/early/`（3 个） | clean_base_v2_local / clean_base_v2_relu2 / 628_ablation，已被 brk_a 取代 |
| `exp/wiki5/`（11 个） | 非主线 smoke test + 早期 task4_iter，不可与主线混用 |
| `exp/legacy/`（shared 7 + logs 12） | 废弃/共享/散落，已归档 |

> 物理上这些目录已不在当前 `exp/` 下（见 §5.5），但若远程 I 盘仍有副本，建议同样清理。

---

## 8. Checkpoint 路径汇总

### 8.1 本地声明的 save_dir（来自 config*.json）

| save_dir | 配置 | ablation.name | 用途 |
|---|---|---|---|
| `runs/submission/repro_brk_a_15ep` | `config.json` | `brk_a_ll03_15ep` | 主表训练 |
| `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` | `config.json` `_main_table_metrics.checkpoint` | — | 主表 eval canonical ckpt |
| `runs/r5/weave_r5` | `config_r5.json` | `weave_r5_15ep` | R5 retrain |
| `runs/taesd/brk_a_15ep` | `config_taesd.json` | `taesd_brk_a_15ep` | TAESD VAE 消融 |
| `runs/d5_adain2` | `config_d5_adain2.json` | `d5_adain2` | D5 adain=2.0 消融 |
| `runs/p2a256_adain2` | `config_p2a256_adain2.json` | `p2a256_adain2` | P2A-256 训练 |

### 8.2 远程实际路径（来自 BASELINE_PROGRESS.md / brk_a_15ep_handoff.md）

| 远程路径 | 内容 |
|---|---|
| `I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep/epoch_0001~0015.pt` | brk_a 15 epoch checkpoints + `full_eval/epoch_XXXX/summary.json` |
| `I:/Github/Latent_Style/SchrodingerBridge/exp/model_probe/` | Probe 实验（target_hf_*） |
| `I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned/{photo2art256,random5}/images/` | StyleAligned 输出 |
| `I:/Github/Latent_Style/exp_baselines/{styleid,cut,seedream45_api}/` | 远程 baseline 输出 |
| `I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned/` | StyleAligned 远程目录 |
| `I:/datasets/wikiart_distinct5_samam_512_classview/test` | D5-512 测试集 |
| `I:/datasets/legacy256_overfit50/test` | P2A-256 测试集 |
| `I:/datasets/wikiarts20_512_test` | R5-512 测试集 |

### 8.3 历史 Pareto checkpoints（来自 `docs/archive/exp/README.md` v4，物理位置可能已变）

| Checkpoint | 配置 | CLIP-S | LPIPS |
|---|---|---|---|
| `FCSB/phase4/630_phase4f_lvl3/epoch_0003.pt` | 4F.1 | 0.7319 | 0.3428 |
| `FCSB/phase4/630_phase4i2b_sota_heun_5ep/epoch_0005.pt` | 4I.2b | 0.7266 | 0.3229 |
| `FCSB/phase4/630_phase4i7b_cosine_heun_a085_5ep/epoch_0005.pt` | 4I.7b | 0.7272 | 0.3218 |
| `FCSB/phase4/630_phase4j1_dwt_route/epoch_0005.pt` | 4J.1 | 0.7226 | 0.3068 |
| `FCSB/local_t/630_local_t11_stochastic_dwt_p08/epoch_0005.pt` | T11（本地 SOTA） | 0.7213 | 0.2868 |
| `FCSB/local_t/630_local_t10_stochastic_dwt/epoch_0005.pt` | T10 | 0.7083 | 0.2480 |

### 8.4 信号级实验引用的 checkpoints

来自 `docs/exp/signal_level/README.md` §6 复现命令:
- `G:/GitHub/Latent_Style/WEAVE/exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt`（w03_weave）
- `G:/GitHub/Latent_Style/WEAVE/exp/dino_s_break/brk_d_wll10_10ep/epoch_0010.pt`（w10_global_mse）
- `G:/GitHub/Latent_Style/WEAVE/exp/dino_s_break/brk_u_hh_head/epoch_0010.pt`（hh_head）

> 这些 10-epoch 版本是 brk_a_ll03_15ep 的早期版本，与主表 15-epoch 共享同一 config（仅 num_epochs 不同）。

### 8.5 15 epoch 评估推荐 checkpoint（来自 `brk_a_15ep_handoff.md`）

- **CLIP-S 峰值（adain=1.0）**: `epoch_0011.pt` 或 `epoch_0013.pt`（CLIP-S=0.7186）
- **DINO-S 峰值（adain=1.0）**: `epoch_0007.pt`（DINO-S=0.4837，综合最佳）
- **主表报告（adain=2.0）**: `epoch_0004.pt`（DINO-S=0.4859，对应 `canonical_oriented_epoch4`）

---

## 9. 关键文档索引

### 9.1 入口文档

| 文档 | 作用 |
|---|---|
| `README.md`（项目根） | WEAVE 项目入口 + 15-epoch repro 命令 |
| `EXPERIMENT_LOG.md` | 实验叙事日志（2026-05-16 D0→D2 frontier） |
| `BASELINE_PROGRESS.md` | 3 个新 baseline（StyleAligned/Z-STAR/StyleShot）补充进度 |
| `docs/README.md` | 文档导航 + Active Contract + Evidence Rule |
| `docs/exp/experiment_inventory.md` | **本文档** |

### 9.2 实验文档（`docs/exp/`）

| 文档 | 内容 |
|---|---|
| `brk_a_15ep_handoff.md` | 15 epoch 训练 + 模块有效性分析（2026-07-14） |
| `b2_qualitative.md` | B2 非精选定性网格方法学与结果 |
| `b3_failure_cases.md` | B3 失败案例分析（Minimalism 双向不对称） |
| `b4_reference_stability.md` | B4 reference pool 稳定性 + α=0.1 鲁棒性论证 |
| `b5_compute_cost.md` | B5 计算成本 + 14 baseline Table 1 转录 |
| `r5_512_infer_timing.md` | R5-512 推理时序测量 + 论文 Infer 列修正建议 |
| `postprocess_brightness_alignment_report.md` | 后处理亮度对齐报告 |
| `postprocess_brightness_alignment_deep_analysis.md` | 后处理亮度对齐深度分析 |
| `signal_level/README.md` | 信号级 A/B/C/D 实验 + 话术修正 |
| `onnx_inference/README.md` | ONNX/TRT 加速实验 |

### 9.3 Baseline 与历史文档

| 文档 | 内容 |
|---|---|
| `docs/baseline/README.md` | 12 baseline 数据完整性核查（2026-07-03） |
| `docs/archive/exp/README.md` | 旧实验整理总入口（2026-07-03 M1-M9，已部分过时） |
| `docs/archive/ablation_log.md` | Phase 2 A/B Test log |
| `docs/archive/630/` | 630 系列历史文档（HANDOVER/mask/skill 等） |
| `docs/archive/{616,618,619,620,622}/` | 各阶段历史设计/调试文档 |
| `docs/72/` | docs/72 系列（代码/理论/实验/结论） |
| `docs/maths/{ANALYSIS,MODEL,REFLECTIONS}.md` | 数学分析 |

### 9.4 工具与配置文档

| 文档 | 内容 |
|---|---|
| `tools/experiments/README.md` | SchrodingerBridge 时期 experiments/ 工具索引（部分路径过时） |
| `config_schema.py` | 配置 schema 定义 |
| `data/README.md` | 数据目录说明 |
| `configs/README.md` | configs 目录说明 |
| `archives/README.md` | 归档策略 |

### 9.5 论文与提交

| 路径 | 内容 |
|---|---|
| `aaai2027_v4/paper.tex` | 主论文（Table 1 在 L543-558） |
| `aaai2027_v4/supplement.tex` | 补充材料 |
| `aaai2027_v4/refs.bib` | 参考文献 |
| `aaai2027_v4/build.ps1`, `gen_figures.ps1` | 编译脚本 |
| `aaai2027_v4/make_radar.py` | 雷达图生成 |

---

## 10. 后续维护建议

### 10.1 新增实验时
1. 训练 checkpoint 放入 config 中声明的 `save_dir`（如 `runs/<exp_name>/`）。
2. 在 `EXPERIMENT_LOG.md` 顶部新增带日期的章节，记录目的/配置/结果/checkpoint 路径。
3. 若进入论文主表或消融表，同步更新本文档 §5.1 / §5.6 / §6 / §8。
4. 评估协议沿用 `inference.json`（`endpoint_adain_scale=2.0`, 8-step, 750 pairs, HF CLIP ViT-B/32 + LPIPS Alex + DINOv2-small）。

### 10.2 数据一致性检查（每月或重大实验后）
1. 比对 `exp/baseline/v2/eval/unified_results.json` 与 `docs/baseline/README.md` §0 速览表。
2. 比对 `aaai2027_v4/paper.tex` Table 1 与 `docs/exp/b5_compute_cost.md` §3.5 baselines 对照表。
3. 比对 `EXPERIMENT_LOG.md` 中 trusted anchors 与实际 checkpoint 文件存在性。
4. 任何数值变更需在所有文档同步更新（参考 `docs/archive/exp/README.md` §4.2 已对齐文档列表）。

### 10.3 旧实验清理原则（沿用 `docs/archive/exp/README.md` §5.3）
- 无效代码和机制确认无效后直接删除（不 ablate）。
- 任何后续优化（text、cross-attn、DINO）必须先通过 WFI 检查。
- 训练必须从零开始独立目录（禁止 `--skip-train` resume）。
- 评估阶段显存严格不超过 7G（远程 RTX 3060 12GB 余量）。

### 10.4 待完成事项（来自 BASELINE_PROGRESS.md §6）
1. ⬜ StyleAligned P2A+R5 推理完成并评估（schtasks 已重启）。
2. ⬜ Z-STAR 克隆 + wrapper + 3 数据集推理 + 评估。
3. ⬜ StyleShot 克隆 + 权重下载 + wrapper + 3 数据集推理 + 评估。
4. ⬜ 统一评估脚本 `tools/_eval_baselines_v3.py`（待建）跑 CLIP-S / LPIPS / MUSIQ。
5. ⬜ 结果填入 `aaai2027_v4/paper.tex` 主表（编号 5 StyleAligned / 6 Z-STAR / 8 StyleShot + caption + baseline 列表 + Related Work + 首页散点图 `plot_page1_summary.py`）。
6. ⬜ 论文 Table 1 Infer 列修正（见 §5.8：SD-Turbo 5m→9m, StyleID 63m→40m, StyTR-2 38m→5m, AesPA 23m→4m, Latent-WCT 18s→4m）。

---

## 11. 整理元信息

- **本次整理方法**: 静态文档扫描（无代码修改）。
- **覆盖文档**: `EXPERIMENT_LOG.md`, `BASELINE_PROGRESS.md`, 5 个 `config*.json`, `inference.json`, `docs/README.md`, `docs/baseline/README.md`, `docs/exp/{b2,b3,b4,b5,brk_a_15ep_handoff,r5_512_infer_timing,signal_level/README,onnx_inference/README}.md`, `docs/archive/{exp/README,ablation_log}.md`, `tools/experiments/README.md`, `state/remote_inventory.md`, `README.md`。
- **未读取但存在的文档**: `docs/exp/postprocess_brightness_alignment_{report,deep_analysis}.md`（仅列名）, `docs/archive/{616,618,619,620,622,630,72,710,79,713,SWD}/` 下子文档, `docs/{architecture,attn,bridge,known_issues,quickstart,review,tokenizer,writing}.md`, `maths/{ANALYSIS,MODEL,REFLECTIONS}.md`, `aaai2027_v4/paper.tex` 与 `supplement.tex` 全文。
- **本地 exp/ 与 runs/ 状态**: 多为空占位（实际产物在远程 I 盘）；`docs/exp/` 与 `tools/` 文件齐全。
- **整理局限性**: 无法访问远程 I 盘验证 checkpoint 实际存在性；FCSB Pareto checkpoints 路径来自旧 README，物理位置可能已变；`exp/README.md` 描述的旧结构与当前物理目录不符（见 §5.5 警示）。

---

**最后更新**: 2026-07-23
**整理执行**: 静态文档扫描（无代码修改）
**数据状态**: ✅ 论文主表与 B2-B5 实验文档齐全；⚠️ 历史 FCSB 路径与远程实际目录需远程核查；⚠️ 3 个新 baseline (StyleAligned/Z-STAR/StyleShot) 待跑完。
