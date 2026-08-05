# CHECKPOINT

> WEAVE 检查点清单与加载指南（双盲匿名版本）。
> 所有路径均为项目相对路径；实验位置与复现命令详见 `EXPERIMENTS.md`。
> 术语统一：本文档使用 **Channel-wise AdaIN** 指代 endpoint 子带的对角协方差匹配（mean+std），对应代码字段 `endpoint_adain_scale`。

---

## 1. 检查点清单

### 1.1 总览

| 项目 | 数量 | 说明 |
|------|------|------|
| `exp/` 下含 `.pt/.pth/.ckpt` 的子目录数 | 12 | 训练实验目录（含模型权重） |
| `exp/` + `runs/` 下 `.pt/.pth/.ckpt` 文件总数 | 153 | 包含模型权重 + latent cache + 等价性文件 |
| 模型权重检查点 | 147 | `exp/dino_s_break/`(41) + `dwt_route_distinct5/`(10) + `musiq_*/`(90) + `sem_*/`(5) + `runs/`(2 非训练) |
| 非模型权重 `.pt` | 6 | `exp/train_taesd/.latent_cache/packed/`(5 latent cache) + `runs/equiv_*.pt`(2 等价性文件，不参与推理) |

### 1.2 检查点分类索引

| 类别 | 路径模式 | 文件数 | 单文件大小 | 用途 |
|------|----------|--------|-----------|------|
| **主检查点（推理用）** | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` | — | — | 主表指标来源（`config.json` 引用） |
| **主检查点训练源** | `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt` | 1 | 10.15 MB | 主检查点的 10-epoch 训练产物 |
| DINO-S break 消融 | `exp/dino_s_break/brk_*/epoch_0010.pt` 或 `epoch_0015.pt` | 41 | ~10.15 MB | §5 Ablation（T12/T14 等消融轴） |
| DWT 路由消融 | `exp/dwt_route_distinct5/epoch_0001.pt` ~ `epoch_0010.pt` | 10 | 10.26 MB | §5 Ablation（T1） |
| MUSIQ 消融 | `exp/musiq_d1_ll_anchor/`, `exp/musiq_s1_sem_region/` ~ `musiq_s8_combined/` | 90 | 8.09 MB | MUSIQ 系列（每子目录 10 epoch） |
| SEM 机制消融 | `exp/sem_r4_k_sweep/`, `sem_r8_*_mechanism/`, `sem_r16_k_sweep/` | 5 | 10.26 MB | 语义模块机制消融（仅 `epoch_0005.pt`） |
| TAESD latent cache | `exp/train_taesd/.latent_cache/packed/00_*.pt` ~ `04_*.pt` | 5 | 62.57 MB | **非模型权重**，TAESD 训练的 latent 缓存 |
| 等价性检查文件 | `runs/equiv_old.pt`, `runs/equiv_root.pt` | 2 | 50.1 KB | **非训练检查点**，等价性测试产物 |

### 1.3 检查点格式说明

- **文件后缀**: 所有训练检查点统一使用 `.pt`（PyTorch ZIP 格式）。
- **命名规则**: `epoch_{NNNN}.pt`（4 位填充 epoch 编号；例如 `epoch_0004.pt`、`epoch_0010.pt`、`epoch_0015.pt`）。
- **保存间隔**: 由 `config.json` 中 `training.save_interval=1` 控制（每 epoch 保存一次）。
- **存储字段**: 见 §6.1（payload 结构）。

### 1.4 主检查点状态

> ⚠️ **重要约束**: WEAVE 推理使用 **4-epoch 检查点**（`epoch_0004.pt`），不是训练终点的 `epoch_0010.pt`。详见 §2 与 §4。

| 字段 | 值 |
|------|----|
| 配置中引用路径 | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| 引用位置 | `config.json` → `_main_table_metrics.checkpoint` |
| 训练源目录 | `exp/dino_s_break/brk_a_ll03_10ep/`（10-epoch 训练，命名 `_10ep`） |
| 训练源最末检查点 | `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt`（10.15 MB，已确认存在） |
| 远程 `runs/` 目录状态 | 仅含 5 个顶层文件（3 CSV + 2 `.pt`），**不含** `runs/submission/` 子目录 |
| 复现路径 | 通过 `scripts/run_submission_repro.ps1` 重新训练 + 评估生成 |

---

## 2. 主检查点

### 2.1 主检查点信息

| 字段 | 值 |
|------|-----|
| 检查点路径（推理引用） | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| 训练源检查点 | `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt` |
| 训练 epoch 数 | 10 |
| **推理用 epoch** | **4**（best on R5-512 tuning split） |
| 训练 `endpoint_adain_scale` | 1.0 |
| 评估 `endpoint_adain_scale` | **2.0**（由 `inference.json` 覆盖；最优值） |
| 主表指标 C/D/L | 0.8103 / 0.4916 / 0.2593 |
| CLIP-S（参考） | 0.7128 |
| 评估数据集 | D5-512（5 风格 × 30 内容 = 150 测试图像；750 pairs 含对角 identity） |
| DINO 口径 | max（all-pairs，对每个生成图取与所有参考图 CLS 余弦最大值） |

### 2.2 主检查点引用位置

| 引用位置 | 字段 | 值 |
|----------|------|-----|
| `config.json` | `_main_table_metrics.checkpoint` | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` |
| `config.json` | `_main_table_metrics.eval_protocol` | `train endpoint_adain_scale=1.0; eval uses inference.json override endpoint_adain_scale=2.0; oriented-HF: experiments/architecture/hf_oriented_nohh.json` |
| `config.json` | `_main_table_metrics.training_endpoint_adain_scale` | 1.0 |
| `config.json` | `_main_table_metrics.eval_endpoint_adain_scale` | 2.0 |
| `inference.json` | `model.endpoint_adain_scale` | 2.0 |
| `verify_main_table.py` | `PAPER_DINO_C / PAPER_CLIP_S / PAPER_LPIPS` | 0.8103 / 0.7128 / 0.2593 |

### 2.3 主检查点 vs 训练源

| 维度 | 主检查点（推理用） | 训练源检查点（训练产物） |
|------|---------------------|--------------------------|
| 路径 | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` | `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt` |
| Epoch | 4 | 10（训练终点） |
| 是否用于主表推理 | ✅ 是 | ❌ 否（仅作为训练产物保留） |
| 是否需要从训练源重新导出 | 是（见 §6.3 复现命令） | — |
| 远程存在性 | ❌ 不在远程 `runs/` 下 | ✅ 远程 `exp/dino_s_break/brk_a_ll03_10ep/` 下存在 |

> **约束**: 主表指标必须从唯一 checkpoint（4-epoch 版本）重新生成。`epoch_0010.pt` 仅为训练源最末检查点，不用于主表推理。

---

## 3. 训练配置

### 3.1 训练超参数（`config.json` → `training`）

| 字段 | 值 | 说明 |
|------|-----|------|
| `seed` | 42 | 随机种子 |
| `batch_size` | 96 | 训练批大小 |
| `accumulation_steps` | 1 | 梯度累积步数 |
| `num_workers` | 0 | DataLoader workers |
| `learning_rate` | 2e-4 | AdamW 学习率 |
| `min_learning_rate` | 1e-5 | cosine 调度下限 |
| `weight_decay` | 1e-4 | 权重衰减 |
| `scheduler` | `cosine` | 学习率调度器 |
| `grad_clip_norm` | 1.0 | 梯度裁剪 |
| `num_epochs` | 15 | 训练总 epoch（主检查点训练源 `brk_a_ll03_10ep` 使用 10 epoch） |
| `save_interval` | 1 | 每 epoch 保存一次 |
| `use_amp` | true | 混合精度 |
| `amp_dtype` | `bf16` | AMP 数据类型 |
| `allow_tf32` | true | TF32 加速 |
| `cudnn_benchmark` | true | cuDNN benchmark |
| `fused_adamw` | true | 融合 AdamW |
| `resume_prefer_local_checkpoint` | true | 优先使用本地最新检查点 resume |
| `resume_checkpoint` | `""` | 显式 resume 路径（空表示自动发现） |
| `resume_optimizer` | true | resume 时加载优化器状态 |
| `resume_model_strict` | true | resume 时严格加载模型 state_dict |

### 3.2 模型配置（`config.json` → `model`，关键字段）

| 字段 | 值 | 说明 |
|------|-----|------|
| `latent_channels` | 4 | VAE latent 通道数 |
| `num_styles` | 5 | 风格数（D5） |
| `style_dim` | 160 | 风格向量维度 |
| `style_tokenizer` | `factorized` | 风格 tokenizer 类型 |
| `contract_family` | `weave` | 模型契约族（WEAVE） |
| `solver_family` | `solver_i2sb` | 求解器族（I2SB） |
| `tokenizer_family` | `legacy_factorized` | tokenizer 实现 |
| `lowpass_mode` | `dwt_haar` | 低通模式（ Haar DWT） |
| `spectral_ode_enabled` | true | 启用 spectral ODE |
| `spectral_ode_levels` | 1 | spectral ODE 层数 |
| `style_extrap_alpha` | 0.1 | 风格外推强度 |
| `endpoint_adain_scale`（训练） | 1.0 | 训练时 Channel-wise AdaIN 强度 |
| `transport_prediction_mode` | `endpoint` | 传输预测模式 |
| `transport_endpoint_scale` | 4.0 | endpoint 尺度 |
| `bridge_path_mode` | `tri_band` | 桥路径模式（三带） |
| `semantic_supervision_family` | `legacy_terminal_swd` | 语义监督族 |

### 3.3 桥/损失配置（`config.json` → `bridge`，关键字段）

| 字段 | 值 | 说明 |
|------|-----|------|
| `objective_mode` | `flow_matching` | 目标模式（flow matching） |
| `loss_type` | `mse` | 损失类型 |
| `coupling_solver` | `independent` | 耦合求解器 |
| `sinkhorn_epsilon` | 0.05 | Sinkhorn 正则 |
| `sinkhorn_iters` | 60 | Sinkhorn 迭代数 |
| `bridge_sigma` | 0.02 | 桥噪声 sigma |
| `bridge_noise_schedule` | `delayed` | 噪声调度 |
| `ll_partial_style_enabled` | true | LL 子带部分风格化启用 |
| `ll_partial_alpha` | 0.3 | LL 部分风格化 alpha（对应 `brk_a_ll03`） |
| `ll_partial_mode` | `adain` | LL 部分风格化模式 |
| `training_objective_mode` | `velocity` | 训练目标（velocity） |
| `spectral_w_ll` / `w_lh` / `w_hl` | 0.3 / 1.0 / 1.0 | 三带损失权重 |

### 3.4 数据配置（`config.json` → `data`）

| 字段 | 值 | 说明 |
|------|-----|------|
| `data_root` | `data/train` | 训练数据根目录 |
| `style_subdirs` | `["Early_Renaissance","Impressionism","Minimalism","Rococo","Ukiyo_e"]` | 5 风格子目录 |
| `allow_hflip` | true | 水平翻转增强 |
| `balance_target_styles_per_batch` | true | 批内平衡目标风格 |
| `pairing_cache_path` | `data/train/.latent_cache/prototype_pairing_top8.pt` | 配对缓存路径 |
| `pairing_cache_topk` | 8 | 配对 top-k |
| `pairing_cache_cross_only` | true | 仅跨风格配对 |
| `latent_cache_mode` | `packed` | latent 缓存模式 |
| `latent_cache_dir` | `.latent_cache/packed` | latent 缓存子目录 |
| `dino_bank_limit_per_style` | 8 | 每风格 DINO bank 上限 |

### 3.5 检查点配置（`config.json` → `checkpoint`）

| 字段 | 值 | 说明 |
|------|-----|------|
| `save_dir` | `runs/submission/repro_brk_a_15ep` | 检查点保存目录（配置默认） |
| `resume_checkpoint` | `""` | 显式 resume 路径 |

### 3.6 消融元信息（`config.json` → `ablation`）

| 字段 | 值 | 说明 |
|------|-----|------|
| `name` | `brk_a_ll03_15ep` | 消融名（主检查点训练源对应 `brk_a_ll03_10ep`） |
| `axis` | `dino_s_break` | 消融轴 |
| `stage` | `brk_round1` | 消融阶段 |
| `notes` | `LL AdaIN alpha=0.3 + 15 epochs sweep to find best checkpoint. Each-epoch eval enabled.` | 消融说明 |

### 3.7 推理配置（`inference.json`）

| 字段 | 值 | 说明 |
|------|-----|------|
| `model.endpoint_adain_scale` | **2.0** | 推理时 Channel-wise AdaIN 强度（覆盖训练的 1.0） |
| `inference.num_steps` | 8 | 推理步数 |
| `inference.step_size` | 1.0 | 推理步长 |
| `inference.style_strength` | 1.0 | 风格强度 |
| `full_eval.batch_size` | 16 | 评估批大小 |
| `full_eval.max_src_samples` | 30 | 每风格最大源样本数（5 × 30 = 150） |
| `full_eval.max_ref_compare` | 16 | 参考图比较上限 |
| `full_eval.vae_model` | `ema` | VAE 模型（EMA） |
| `full_eval.save_generated_images` | true | 保存生成图像 |
| `full_eval.only_lpips_clip_style` | true | 仅计算 LPIPS/CLIP-S/风格指标 |
| `full_eval.clip_style_idt_baseline` | 0.639920825263 | CLIP-S IDT 基线（参考） |

---

## 4. 停止规则

### 4.1 训练 epoch 与推理 epoch 的分离

WEAVE 采用 **训练-推理 epoch 分离**策略：

| 阶段 | epoch | 用途 |
|------|-------|------|
| 训练 | 0 → 10 | 训练 `brk_a_ll03_10ep` 配置，每 epoch 保存检查点 |
| 推理选型 | **4**（best on R5-512 tuning split） | 在 R5-512 调谐集上评估每个 epoch，选出最佳 epoch |
| 训练终点 | 10 | `epoch_0010.pt`（仅作为训练产物保留，不用于主表） |

### 4.2 选型规则

| 项 | 值 |
|----|----|
| 调谐集 | R5-512（tuning split，非测试集） |
| 选型指标 | DINO-S（max 口径）+ LPIPS 综合考量 |
| 最佳 epoch | **4** |
| 是否在测试集上调参 | ❌ 否（R5-512 是调谐集，D5-512/P2A-256 是测试集） |

### 4.3 一致性约束

> **约束**: Checkpoint epoch 必须与论文停止规则、`config.json`、`README`/`CHECKPOINT.md`、训练日志一致。

| 一致性项 | 值 | 验证位置 |
|----------|----|----------|
| 论文停止规则 | epoch 4（best on R5-512） | 论文 §5 与 §?（实验细节） |
| `config.json` | `_main_table_metrics.checkpoint = runs/submission/canonical_oriented_epoch4/epoch_0004.pt` | `config.json` |
| 训练日志 | `exp/dino_s_break/brk_a_ll03_10ep/logs/training_*.csv` | 远程 |
| 每epoch评估 | `full_eval_each_epoch=true`（每 epoch 评估） | `config.json` → `training.full_eval_each_epoch` |

### 4.4 `endpoint_adain_scale` 选取约束

> **约束**: `endpoint_adain_scale=2.0` 是最优值；>2.0 会模型崩溃。

| scale | 结果文件 | 状态 |
|-------|----------|------|
| 1.5 | `runs/endpoint_adain_1p5.csv` | 次优 |
| **2.0** | `runs/endpoint_adain_2p0.csv` | **最优（主表使用）** |
| 2.5 | `runs/endpoint_adain_2p5.csv` | 模型崩溃（T12:3.0→0.32, T14:2.5→0.36） |

**无 cherry-picking 验证**:
- `endpoint_adain_scale=2.0` 在 R5-512（tuning split）上选取
- 应用到 D5-512 / P2A-256（test split）无提升，表明未在测试集上调参

### 4.5 训练 convergence 声明

> **约束**: 复现需跑到收敛。

| 项 | 值 |
|----|----|
| 训练 epoch | 10（`brk_a_ll03_10ep`）/ 15（`brk_h_ll03_15ep`，对比组） |
| 收敛标志 | `brk_a_ll03_10ep` 在 R5-512 上 epoch 4 取得最佳 → 训练已收敛（早停） |
| 多 seed 验证 | `brk_k_ll03_seed7`（seed=7 复现实验，`epoch_0010.pt`） |

---

## 5. 检查点与论文表对应关系

### 5.1 主表（Table 1, D5-512）

| 论文行 | 方法 | 检查点 / 数据来源 | C (DINO-C) | D (DINO-S) | L (LPIPS) | S (CLIP-S) |
|--------|------|-------------------|------------|------------|-----------|-----------|
| WEAVE (ours) | WEAVE | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` | 0.8103 | 0.4916 | 0.2593 | 0.7128 |
| IDT (baseline) | IDT | `exp/idt_baseline/`（750 pairs 实时计算） | 1.0 | (clip 0.6399) | ~0 | 0.6399 |
| Latent-WCT | baseline | `exp/latent_wct_baseline/`（**目录为空，结果在 `results/ours/`） | — | — | — | — |
| 其他 baseline | baseline | `state/dino/D5-512__{cut,samam,samst,sdturbo,styleid,wct,zstar}.json` | — | — | — | — |

> **关键约束**:
> - IDT baseline **必须从固定 750 pairs 实时计算**，不能手动填旧值（由 `verify_main_table.py` 自动执行）。
> - 主表 WEAVE 行指标 **必须从唯一 4-epoch 检查点重新生成**，不能从其他 checkpoint 拼接。
> - DINO-S 用 **max（all-pairs）口径**，不是 avg。

### 5.2 消融表（Table 2, Ablation）

| 消融 ID | 论文位置 | 检查点 / 数据来源 | 说明 |
|---------|----------|-------------------|------|
| T1 | §5 Ablation | `exp/dwt_route_distinct5/epoch_0010.pt` + `full_eval/epoch_0010/` | DWT 路由消融 |
| T2 | §5 Ablation | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` + `inference_endpoint_only.json` | Endpoint-only（仅端点 Channel-wise AdaIN） |
| T7 | §5 Ablation | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` + `inference_block.json` | Stepwise（逐步 Channel-wise AdaIN） |
| T8 | §5 Ablation | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` + `runs/endpoint_adain_*.csv` | `endpoint_adain_scale` 敏感性 |
| T12 | §5 Ablation | `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt` | LL 子带部分风格化（alpha=0.3） |
| T14 | §5 Ablation | `exp/dino_s_break/brk_w_hf_wct/epoch_0010.pt` + `brk_z_ll_wct/epoch_0010.pt` | HF/LL WCT 替换 |

### 5.3 其他论文位置

| 论文位置 | 检查点 / 数据来源 | 说明 |
|----------|-------------------|------|
| §5 Qualitative | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` + `exp/b2_qualitative/` | B2 定性对比（8 pairs × 9 methods） |
| §5 Analysis | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` + `exp/b3_failure_cases/` | B3 失败案例（15 cases） |
| §5 Efficiency | `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` + `exp/b5_compute_cost/` | B5 计算开销 |
| §5 Discussion | `exp/dino_avg_vs_max/` | DINO 口径对比（avg vs max） |
| §S9 (Supplement) | `exp/latent_wct_baseline/`（**目录为空**） | Latent-WCT 基线 |
| Reviewer response | `exp/rebuttal/`（task1/task2/task4 子任务） | 审稿回应实验 |

---

## 6. 加载方式

### 6.1 检查点 payload 结构

每个 `.pt` 检查点是 PyTorch `torch.save` 序列化的字典，包含以下字段（见 `trainer.py:1870` `save_checkpoint`）:

```python
payload = {
    "epoch": int,                       # epoch 编号
    "global_step": int,                 # 全局步数
    "model_state_dict": dict,           # 模型 state_dict
    "optimizer_state_dict": dict,       # 优化器 state_dict
    "scheduler_state_dict": dict|None,  # 调度器 state_dict
    "loss_state_dict": dict|None,       # 损失模块 state_dict
    "config": dict,                     # 训练配置快照
    "metrics": dict,                    # 保存时的指标
}
```

### 6.2 训练 resume 加载

`trainer.py:1061` `_maybe_resume` 实现训练 resume:

```python
# 自动发现最新检查点
def _find_latest_checkpoint(self) -> Optional[Path]:
    ckpts = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None

# resume 加载逻辑（简化）
state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
model_state = strip_compile_prefix(state["model_state_dict"])
model_state, _ = prune_state_dict_for_tokenizer_family(model_state, ...)
# 加载 model / optimizer / scheduler / loss state_dict
# 详见 trainer.py:1081-1130
```

resume 优先级（`config.json` → `training`）:
1. 若 `resume_prefer_local_checkpoint=true`（默认）且本地 `checkpoint_dir` 下有 `epoch_*.pt` → 用本地最新
2. 否则若 `resume_checkpoint` 非空 → 用显式路径
3. 否则从零开始训练

### 6.3 推理加载（`utils/inference.py:357`）

推理引擎从检查点加载模型 state_dict（不含优化器/调度器）:

```python
# 简化的推理加载流程（utils/inference.py:357-412）
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
raw_config = checkpoint.get("config", {}) or {}
if config_override_path:
    raw_config = merge_config_dicts(raw_config, load_config(config_override_path))
config = ExperimentConfig.from_mapping(raw_config)

# 推理时 config override（如 endpoint_adain_scale=2.0）
# 通过 --inference inference.json 传入

state_dict = strip_compile_prefix(checkpoint["model_state_dict"])
state_dict, _ = prune_state_dict_for_tokenizer_family(state_dict, ...)
state_dict, _ = prune_retired_checkpoint_keys(state_dict)

model = build_model_from_config(config.model, bridge_cfg=config.bridge, use_checkpointing=False).to(device)
model.load_state_dict(state_dict, strict=False)
model.eval()
```

### 6.4 推理命令（主表复现）

```powershell
# 单检查点评估（主表指标复现）
python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt --config config.json --inference inference.json
```

参数说明:
- `--checkpoint`: 检查点路径（必填，单检查点模式）
- `--config`: 训练配置 JSON（用于构建模型结构）
- `--inference`: 推理 override JSON（覆盖 `endpoint_adain_scale` 等推理字段）

### 6.5 主表验证脚本

```powershell
# 一键输出主表 4 个指标（自动从 config.json 读取 WEAVE 值，实时计算 IDT baseline）
python verify_main_table.py
```

`verify_main_table.py` 行为:
- 读取 `config.json` → `_main_table_metrics`（WEAVE 行指标）
- 读取 `exp/idt_baseline/metrics.csv` + `exp/idt_baseline/dino.json` 实时计算 IDT baseline
- 与论文参考值（`PAPER_DINO_C=0.8103` 等）比对，输出 `Match: YES/NO`
- 30 秒超时保护（`_TIMEOUT_SEC=30`）

### 6.6 主检查点重新生成（从训练源）

> **约束**: 远程 `runs/submission/` 子目录不存在，需通过训练源重新生成 4-epoch 检查点。

```powershell
# 方法 1: 重新训练 + 每epoch评估（推荐）
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1

# 方法 2: 仅评估已有检查点（复用 epoch 检查点，评估缺失 epoch）
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1 -EvalOnly
```

训练源信息:
- 训练源目录: `exp/dino_s_break/brk_a_ll03_10ep/`
- 训练源最末检查点: `exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt`（10.15 MB，远程存在）
- 训练配置: `exp/dino_s_break/brk_a_ll03_10ep/config.json`（若存在）或根目录 `config.json`
- 训练日志: `exp/dino_s_break/brk_a_ll03_10ep/logs/training_*.csv`

### 6.7 检查点加载注意事项

| 注意项 | 说明 |
|--------|------|
| `weights_only=False` | 加载时必须设置（因 payload 含 config 字典等非张量对象） |
| `strip_compile_prefix` | 处理 `torch.compile` 前缀（如 `_orig_mod.`） |
| `prune_state_dict_for_tokenizer_family` | 根据 `tokenizer_family` 剪枝不兼容的 key |
| `prune_retired_checkpoint_keys` | 剪枝已废弃的 key（见 `utils/inference.py:41`） |
| `strict=False`（推理） | 推理加载允许部分 key 缺失（如可选 head） |
| `strict=True`（训练 resume） | 训练 resume 默认严格加载（`resume_model_strict=true`） |

---

## 7. 附录

### 7.1 完整 `exp/dino_s_break/` 检查点清单（41 个）

| 子目录 | 文件 | 大小(MB) | 用途 |
|--------|------|----------|------|
| `brk_a_ll03_10ep` | `epoch_0010.pt` | 10.15 | **主检查点训练源** |
| `brk_aa_aug24` | `epoch_0010.pt` | 10.15 | 数据增强 aug24 |
| `brk_aa_aug35` | `epoch_0010.pt` | 10.15 | 数据增强 aug35 |
| `brk_ab_hf_ovs13` | `epoch_0010.pt` | 10.15 | HF 过采样 ovs13 |
| `brk_ab_hf_ovs15` | `epoch_0010.pt` | 10.15 | HF 过采样 ovs15 |
| `brk_ac_fft_loss` | `epoch_0010.pt` | 10.15 | FFT 损失 |
| `brk_ad_adain_base` | `epoch_0010.pt` | 10.15 | Channel-wise AdaIN 基线 |
| `brk_ad_adain_hfonly` | `epoch_0010.pt` | 10.15 | Channel-wise AdaIN 仅 HF |
| `brk_ad_adain_hi` | `epoch_0010.pt` | 10.15 | Channel-wise AdaIN hi |
| `brk_ad_adain_lo` | `epoch_0010.pt` | 10.15 | Channel-wise AdaIN lo |
| `brk_ae_adain_05` | `epoch_0010.pt` | 10.55 | AdaIN scale=0.5 |
| `brk_ae_adaln_02` | `epoch_0010.pt` | 10.55 | AdaLN scale=0.2 |
| `brk_ae_adaln_05` | `epoch_0010.pt` | 10.55 | AdaLN scale=0.5 |
| `brk_ae_adaln_10` | `epoch_0010.pt` | 10.55 | AdaLN scale=1.0 |
| `brk_af_adaln_g10` | `epoch_0010.pt` | 10.35 | AdaLN g10 |
| `brk_af_adaln_q10` | `epoch_0010.pt` | 10.55 | AdaLN q10 |
| `brk_af_adaln_qg05` | `epoch_0010.pt` | 10.75 | AdaLN qg05 |
| `brk_b_baseline_10ep` | `epoch_0010.pt` | 10.15 | 基线（无 LL 部分风格化） |
| `brk_d_wll10_10ep` | `epoch_0010.pt` | 10.15 | w_LL=0.1 |
| `brk_e_wll30_10ep` | `epoch_0010.pt` | 10.15 | w_LL=0.3 |
| `brk_f_ll04_10ep` | `epoch_0010.pt` | 10.15 | LL alpha=0.4 |
| `brk_g_ll05_10ep` | `epoch_0010.pt` | 10.15 | LL alpha=0.5 |
| `brk_h_ll03_15ep` | `epoch_0015.pt` | 10.15 | LL alpha=0.3, 15 epoch |
| `brk_i_ll02_10ep` | `epoch_0010.pt` | 10.15 | LL alpha=0.2 |
| `brk_j_ll03_zwct_10ep` | `epoch_0010.pt` | 10.15 | LL alpha=0.3 + zWCT |
| `brk_k_ll03_seed7` | `epoch_0010.pt` | 10.15 | LL alpha=0.3, seed=7（多 seed 验证） |
| `brk_l_w_lh_hl_15` | `epoch_0010.pt` | 10.15 | w_LH/HL=1.5 |
| `brk_m_adain15` | `epoch_0010.pt` | 10.15 | AdaIN scale=1.5 |
| `brk_n_two_stage` | `epoch_0010.pt` | 10.15 | 两阶段训练 |
| `brk_o_endpoint_high_03` | `epoch_0010.pt` | 10.15 | endpoint high=0.3 |
| `brk_p_lr_0003` | `epoch_0010.pt` | 10.15 | lr=3e-4 |
| `brk_q_adain20` | `epoch_0010.pt` | 10.15 | AdaIN scale=2.0 |
| `brk_r_adain18` | `epoch_0010.pt` | 10.15 | AdaIN scale=1.8 |
| `brk_s_adain16` | `epoch_0010.pt` | 10.15 | AdaIN scale=1.6 |
| `brk_t_adain17` | `epoch_0010.pt` | 10.15 | AdaIN scale=1.7 |
| `brk_u_hh_head` | `epoch_0010.pt` | 10.18 | HH head |
| `brk_v_train_adain` | `epoch_0010.pt` | 10.15 | 训练时启用 AdaIN |
| `brk_w_hf_wct` | `epoch_0010.pt` | 10.15 | HF 子带 WCT 替换（T14） |
| `brk_x_dim96` | `epoch_0010.pt` | 15.07 | style_dim=96 |
| `brk_y_ml_dwt_a05` | `epoch_0010.pt` | 10.15 | 多级 DWT alpha=0.5 |
| `brk_z_ll_wct` | `epoch_0010.pt` | 10.15 | LL 子带 WCT 替换（T14） |

### 7.2 完整 `exp/musiq_*/` 检查点清单（9 子目录 × 10 epoch = 90 个）

| 子目录 | 文件数 | 单文件大小 | 说明 |
|--------|--------|-----------|------|
| `musiq_d1_ll_anchor` | 10 (`epoch_0001` ~ `epoch_0010`) | 8.09 MB | MUSIQ D1 LL anchor |
| `musiq_s1_sem_region` | 10 | 8.09 MB | MUSIQ S1 语义 region |
| `musiq_s2_sem_patch` | 10 | 8.09 MB | MUSIQ S2 语义 patch |
| `musiq_s3_sem_band` | 10 | 8.09 MB | MUSIQ S3 语义 band |
| `musiq_s4_sem_xattn` | 10 | 8.09 MB | MUSIQ S4 语义 cross-attn |
| `musiq_s5_band_xattn` | 10 | 8.09 MB | MUSIQ S5 band cross-attn |
| `musiq_s6_patch_xattn` | 10 | 8.09 MB | MUSIQ S6 patch cross-attn |
| `musiq_s7_dwt_energy` | 10 | 8.09 MB | MUSIQ S7 DWT 能量 |
| `musiq_s8_combined` | 10 | 8.09 MB | MUSIQ S8 组合 |

### 7.3 完整 `exp/sem_*/` 检查点清单（5 个，仅 `epoch_0005.pt`）

| 子目录 | 文件 | 大小(MB) | 说明 |
|--------|------|----------|------|
| `sem_r4_k_sweep` | `epoch_0005.pt` | 10.26 | r4 k 扫描 |
| `sem_r8_hier_mechanism` | `epoch_0005.pt` | 10.26 | r8 层次机制 |
| `sem_r8_sinkhorn_mechanism` | `epoch_0005.pt` | 10.26 | r8 Sinkhorn 机制 |
| `sem_r8_soft_mechanism` | `epoch_0005.pt` | 10.26 | r8 soft 机制 |
| `sem_r16_k_sweep` | `epoch_0005.pt` | 10.26 | r16 k 扫描 |

### 7.4 完整 `exp/dwt_route_distinct5/` 检查点清单（10 个）

| 文件 | 大小(MB) |
|------|----------|
| `epoch_0001.pt` | 10.26 |
| `epoch_0002.pt` | 10.26 |
| `epoch_0003.pt` | 10.26 |
| `epoch_0004.pt` | 10.26 |
| `epoch_0005.pt` | 10.26 |
| `epoch_0006.pt` | 10.26 |
| `epoch_0007.pt` | 10.26 |
| `epoch_0008.pt` | 10.26 |
| `epoch_0009.pt` | 10.26 |
| `epoch_0010.pt` | 10.26 |

### 7.5 非模型权重 `.pt` 文件

| 路径 | 大小(MB) | 类型 | 说明 |
|------|----------|------|------|
| `exp/train_taesd/.latent_cache/packed/00_Early_Renaissance.pt` | 62.57 | latent cache | TAESD 训练用 latent 缓存 |
| `exp/train_taesd/.latent_cache/packed/01_Impressionism.pt` | 62.57 | latent cache | 同上 |
| `exp/train_taesd/.latent_cache/packed/02_Minimalism.pt` | 62.57 | latent cache | 同上 |
| `exp/train_taesd/.latent_cache/packed/03_Rococo.pt` | 62.57 | latent cache | 同上 |
| `exp/train_taesd/.latent_cache/packed/04_Ukiyo_e.pt` | 62.57 | latent cache | 同上 |
| `runs/equiv_old.pt` | 0.05 | 等价性文件 | 等价性检查产物（非训练检查点） |
| `runs/equiv_root.pt` | 0.05 | 等价性文件 | 同上 |

### 7.6 数据来源说明

- 本清单基于 `exp_inventory.md`（远程探查脚本生成，只读）。
- 审计方式: 只读，未修改任何远程文件。
- 训练配置来源: `config.json`（项目根目录，584 行）。
- 推理配置来源: `inference.json`（项目根目录，81 行）。
- 验证脚本: `verify_main_table.py`（项目根目录）。

### 7.7 术语与约束备忘

| 术语 / 约束 | 说明 |
|-------------|------|
| **Channel-wise AdaIN** | endpoint 子带的对角协方差匹配（mean+std）；对应代码字段 `endpoint_adain_scale`。**不使用** "Diagonal WCT" 旧称。 |
| **DINO-S max 口径** | 对每个生成图取与所有参考图 CLS 余弦最大值（all-pairs），**不是 avg**。max 比 avg 高 0.11~0.29。 |
| **IDT baseline 实时计算** | 从固定 750 pairs 实时计算，**不能手动填旧值**。由 `verify_main_table.py` 自动执行。 |
| **主表指标唯一 checkpoint** | 主表 WEAVE 行必须从 `runs/submission/canonical_oriented_epoch4/epoch_0004.pt` 重新生成。 |
| **推理用 4-epoch** | WEAVE 推理使用 4-epoch 检查点（best on R5-512 tuning split），**不是**训练终点的 10-epoch。 |
| **`endpoint_adain_scale=2.0` 最优** | 训练用 1.0，推理用 2.0（由 `inference.json` 覆盖）。>2.0（如 2.5）会导致模型崩溃。 |
| **双盲匿名** | 本文档不包含作者、机构、仓库链接等个人信息。 |
| **相对路径** | 所有路径均为项目相对路径（相对于 WEAVE 项目根目录）。 |
