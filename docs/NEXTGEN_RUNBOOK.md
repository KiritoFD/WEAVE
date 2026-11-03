# Next-Gen WEAVE 服务器 Runbook

配套审查方案「Latent_Style 深度审查：风格瓶颈与下一代模型路线」。本文档覆盖路线 A/B/C/D 与 P0 诊断在 GPU 服务器上的启动方法。所有新功能默认关闭；本仓库当前论文模型的训练/推理行为不受影响。

## 0. 环境依赖

在现有 `requirements.txt` 之上：

```bash
pip install torchvision              # VGG 特征提取器（Route A 风格损失）
pip install diffusers                # AutoencoderKL / AutoencoderTiny（VAE 解码反传、伪目标编码）
# DINOv2（Route A selfsim / Route C 参考图特征，推荐）：
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)"  # 预热缓存
```

硬件：单卡 RTX 4090 足够（基线 4 epoch ≈ 1.4 min @ RTX 3060；Route A 多步可微解码约增加 5–10× 单步开销，microbatch + grad checkpoint 后 24GB 显存充裕）。

### 内部监督主干预的 batch 档位（RTX 3060 12GB 实测，2026-11-04）

| batch | 步/epoch（5000 样本） | 峰值显存 | 结果 |
|---|---|---|---|
| 96 | 52 | 6.6GB | 基线档（计划内 600 步） |
| **160** | **31** | **10.38GB guard / 10.02GB reserved / 9.28GB allocated** | **默认标准档**：496 步 248s，0.45–0.50 s/step，无 OOM，`vram_guard` 未误杀 |
| 176 | 29 | 未实测 | 上限（按 160 峰值线性外推约 11.2GB，需先跑 100 步确认） |
| 180–192 | — | 11.6GB+ | **已 OOM，不再尝试** |

- lr 随 batch 重定：`lr = 5e-5 * sqrt(batch/14)`（b160 → 1.69e-4）。
- 步数预算：`每 epoch 步数 = ceil(样本数 / batch)`，`总更新数 = 步/epoch × epoch`。b96×5000 = 52 步/epoch；b160×5000 = 31 步/epoch。
- `vram_guard.sh` 判活已改为看 `ps` 进程状态（僵尸进程不再被当成低显存误杀）。

数据放置（沿用现有约定）：

```
WEAVE/data/
  train_d5_taesd/.latent_cache/packed/   # 18k packed latent（taesd_encode_d5.py 生成）
  taesd/                                 # TAESD 权重（local_files_only）
  pseudo_d5/                             # Route B 伪目标缓存（本 runbook 第 3 节生成）
```

## 1. P0 诊断（先做，决定后续投入）

### 1.1 数据规模对照（5k vs 18k，双协议）

```bash
python tools/diag_data_scale.py --base_config config_d5_adain2.json \
    --packed_dir data/train_d5_taesd/.latent_cache/packed \
    --out runs/diag_data_scale/manifest.json
# 生成 4 个 cell 配置：{1000, 3600}/style × {fixed_updates, fixed_epochs}
# 逐个启动：
python run.py --config runs/diag_data_scale_configs/small_fixed_updates.json
python run.py --config runs/diag_data_scale_configs/small_fixed_epochs.json   # = 历史基线
python run.py --config runs/diag_data_scale_configs/full_fixed_updates.json
python run.py --config runs/diag_data_scale_configs/full_fixed_epochs.json
```

产物：4 个 checkpoint 目录。每个 cell 约数分钟（fixed_updates 统一 208 次更新）。判定：跨风格 DINO-S 是否随数据稳定上升（用第 4 节的 split 评测口径）。

### 1.2 VAE 重建上限探针

```bash
python tools/vae_recon_probe.py --style_root I:/datasets/d5/test --vae taesd \
    --extractor vgg16 --layers relu2_2,relu3_3 --out runs/vae_recon_probe.json
```

产物 JSON 含 `hf_energy_ratio_median` 与 `verdict`（`latent_ok` / `vae_limited`）。若 `vae_limited`：latent HH 加强收益有限，转图像空间 refiner 路线（Route D 暂缓）。

### 1.3 三方对比（VAE 重建 / 风格化端点 / 最终输出）

`tools/diag_three_way.py` 的指标核心已实现（`three_way_metrics`）；服务器端按其 docstring 组装数据管线后运行，输出 `verdict`：encoding_limited / target_construction_limited / transport_limited。

### 1.4 参考图交换探针

`tools/diag_reference_swap.py`：固定内容 + Style-ID，换同类参考图 K=4，报告 `output_diversity` 与 `alignment_margin`（>0 表示输出跟随各自参考图）。服务器端组装方式见 docstring。

## 2. 路线 A：输出级可微风格监督（后训练）

在基线 config 上追加（bridge 节）：

```json
{
  "bridge": {
    "output_supervision_enabled": true,
    "output_supervision_extractor": "vgg16",
    "output_supervision_decoder": "taesd",
    "output_supervision_style_layers": "relu2_2,relu3_3,relu4_3",
    "output_supervision_content_layers": "relu1_2,relu2_2",
    "output_supervision_selfsim_layer": "relu3_3",
    "output_supervision_ode_steps": 2,
    "output_supervision_lambda_style": 1.0,
    "output_supervision_lambda_content": 0.5,
    "output_supervision_microbatch": 4,
    "output_supervision_grad_checkpoint": true,
    "output_supervision_apply_every": 1
  }
}
```

```bash
python run.py --config config_routeA_posttrain.json   # resume 自基线 checkpoint
```

要点：与历史 Cycle-NCE/单步 SWD 实验的区别写在 `output_supervision.py` / `swd.py` 模块 docstring——这里是最终解码图的多层特征 + 沿真实 ODE 路径反传（2–4 步 Euler + VAE decode）。显存估计：microbatch=4、512px、2 步展开 + TAESD decode ≈ 基线 3–4×；4090 充裕。预期时间：每 step 约基线 5×。

## 3. 路线 B：内容对齐伪目标端点

```bash
# 3.1 生成伪目标缓存（无外部教师时用降级路径）
python tools/gen_pseudo_targets.py --teacher fallback_stats \
    --content_root I:/datasets/d5/train --style_root I:/datasets/d5/train \
    --extractor dinov2 --selfsim_layer block6 --vae taesd \
    --score_threshold 0.35 --out data/pseudo_d5
# 挂真实教师：用 pseudo_targets.CallableTeacher 包装 StyTR2/AesPA-Net/AdaAttN
# 的推理 callable，在同名脚本基础上改两行即可（见文件 docstring）。
```

训练（推荐 mixed——伪目标不作唯一监督，真实目标图的分布级风格匹配经 Route A 保留）：

```json
{
  "bridge": {
    "endpoint_mode": "mixed",
    "pseudo_target_cache_dir": "data/pseudo_d5",
    "pseudo_target_min_score": 0.35,
    "output_supervision_enabled": true
  }
}
```

训练日志关注 `pseudo_target_hit_rate`（缓存命中率）与 `loss_output_sup`。

## 4. 评测口径（所有实验统一）

```bash
# 先按既有 utils/run_evaluation.py 跑 750 请求生成 per-request results.json
# （参考图选择固定为 target test 目录第一张图）
python tools/run_eval_split.py --input results.json \
    --reference_policy first_image_of_target_style_test_dir --out results_split.json
```

主表只报 `main_cross_style`（600 请求）；`identity_same_style`（150 请求）单独作为 Identity Preservation。`reference_selection.policy` 已写入结果文件，跨实验必须一致。

## 5. 路线 C：参考图 Style Encoder（P2，A/B 见效后再做）

```json
{
  "model": {
    "conditioner_variant": "reference_hybrid",
    "reference_encoder_in_dim": 384,
    "reference_encoder_tokens": 32,
    "reference_inject_gate_init": 0.0
  },
  "bridge": { "reference_info_nce_weight": 0.1, "reference_info_nce_temperature": 0.1 }
}
```

训练数据侧需向 batch 提供 `reference_feats`（参考图 DINOv2 patch tokens，(B,N,384)）与 `reference_feats_alt`（同风格不同内容的第二张参考图）——建议离线预提取缓存。注入为零初始化门控：开启时初始行为与基线一致。

## 6. 路线 D：纹理模块

仅占位接口（`model.texture_refiner_enabled = true` 挂载零初始化残差，no-op）。等 1.2 的 VAE 探针给出 `latent_ok` 后再实现重型逻辑。

## 7. t 采样说明

`bridge.t_sampling_legacy_compat` 默认 `true`：无论 `t_sampling_mode` 写什么，t 均匀采样（历史行为、论文口径），并在启动时 warning 提示。新实验若要真实 logit-normal/beta 采样，显式设 `"t_sampling_legacy_compat": false`。

## 8. 预期产物清单

| 实验 | 产物 | 判定指标 |
|---|---|---|
| 1.1 数据规模 | 4 checkpoint + split 评测 JSON | 跨风格 DINO-S 是否随 18k 上升 |
| 1.2 VAE 探针 | runs/vae_recon_probe.json | verdict |
| 1.3 三方对比 | runs/diag_three_way.json | 瓶颈定位 |
| 1.4 参考图交换 | runs/diag_reference_swap.json | alignment_margin > 0 |
| 2 路线 A | post-train checkpoint | DINO-S ≥ 0.53 且 DINO-C/LPIPS 不崩 |
| 3 路线 B | mixed checkpoint | 同等算力风格/内容前沿 |
| 5 路线 C | hybrid checkpoint | 跨内容、跨参考均有提升 |

## 9. 本机（无 GPU/数据）验证

```bash
pytest tests/test_nextgen_skeleton.py          # 18 项 CPU 单测
python tools/gen_pseudo_targets.py --smoke --out <tmp>/pseudo
python tools/vae_recon_probe.py   --smoke --out <tmp>/vae.json
python tools/diag_data_scale.py   --smoke --out <tmp>/scale.json
python tools/diag_reference_swap.py --smoke --out <tmp>/swap.json
python tools/diag_three_way.py    --smoke --out <tmp>/tw.json
python tools/run_eval_split.py    --smoke --out <tmp>/split.json
```
