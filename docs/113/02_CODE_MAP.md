# 02 代码地图

> 只列主线；`archives/legacy-*`、`EXP/`、`experiments/` 是历史产物，不参与运行。行号以本地 HEAD 为准。

## 1 两条入口

| 入口 | 作用 | 关键位置 |
|---|---|---|
| `run.py` | 训练入口：读配置 → 建 dataset → `SBTrainer` → epoch 循环 → 存 ckpt | [run.py#L433-L506](file:///g:/GitHub/Latent_Style/WEAVE/run.py#L433-L506) 配置与数据；[L555](file:///g:/GitHub/Latent_Style/WEAVE/run.py#L555) 建 trainer；[L580-L602](file:///g:/GitHub/Latent_Style/WEAVE/run.py#L580-L602) epoch 循环 |
| `run_evaluation.py` / `utils/run_evaluation.py` | 推理 + 指标（DINO/CLIP/LPIPS/ArtFID 等），也支持批量 ckpt | [run_evaluation.py#L349](file:///g:/GitHub/Latent_Style/WEAVE/run_evaluation.py#L349)；[utils/run_evaluation.py#L2326](file:///g:/GitHub/Latent_Style/WEAVE/utils/run_evaluation.py#L2326) `main`，[L4602](file:///g:/GitHub/Latent_Style/WEAVE/utils/run_evaluation.py#L4602) `generate_summary_json` |

训练主链路：

```
config.json --_base--> load_config() --deep_merge--> ExperimentConfig.from_mapping()
  -> AdaCUTLatentDataset (packed latent + pairing cache)
  -> WEAVE (DWT + style conditioner + velocity heads + solver)
  -> SBTrainer.train_epoch (FlowMatchingObjective.compute + 可选 output_supervision)
  -> save_checkpoint -> run_evaluation (离线)
```

## 2 模块职责

| 文件 | 职责 | 关键符号 |
|---|---|---|
| `config_schema.py` | JSON → dataclass；`_base` 递归继承 + 未知键进 `extra` 并 rehydrate | `load_config` [L1274-L1302](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L1274-L1302)；`_split_known_fields` [L105-L129](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L105-L129)；`ModelConfig` [L309](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L309)、`BridgeConfig` [L917](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L917) |
| `model.py` | 主模型：小波分解、style 条件、三频速度头、Euler/RK4/Heun 求解、stepwise AdaIN | `WEAVE` [L432](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L432)；`forward` [L824](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L824)；`integrate_transport` [L1442](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1442)；`_solver_step` [L1195](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1195) |
| `flow.py` | 训练目标：采样 t、构造 LL/HF 目标、FM loss、AdaIN、伪端点 | `FlowMatchingObjective` [L31](file:///g:/GitHub/Latent_Style/WEAVE/flow.py#L31)；`compute` [L322](file:///g:/GitHub/Latent_Style/WEAVE/flow.py#L322)；`_build_training_target` [L203](file:///g:/GitHub/Latent_Style/WEAVE/flow.py#L203) |
| `trainer.py` | 优化器/调度器/resume/日志/ckpt/内部诊断；`train_epoch` 是实际单步循环 | `SBTrainer` [L197](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L197)；`train_epoch` [L1434](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L1434)；`save_checkpoint` [L1981](file:///g:/GitHub/Latent_Style/WEAVE/trainer.py#L1981) |
| `style.py` | Style-ID embedding → tokens（Style Memory） | `StyleConditioner` [L7](file:///g:/GitHub/Latent_Style/WEAVE/style.py#L7) |
| `blocks.py` | ResBlock + 时间正弦编码 | `ResidualBlock` [L52](file:///g:/GitHub/Latent_Style/WEAVE/blocks.py#L52) |
| `wavelet.py` | Haar DWT/iDWT、多级分解、subband gamma | `dwt2_haar` [L11](file:///g:/GitHub/Latent_Style/WEAVE/wavelet.py#L11)；`idwt2_haar` [L47](file:///g:/GitHub/Latent_Style/WEAVE/wavelet.py#L47)；`subband_gamma` [L168](file:///g:/GitHub/Latent_Style/WEAVE/wavelet.py#L168) |
| `utils/dataset.py` | 逐样本 latent 读取、packed 缓存、pairing cache、style stem 注册 | `AdaCUTLatentDataset` [L159](file:///g:/GitHub/Latent_Style/WEAVE/utils/dataset.py#L159)；packed 读写在 [L371-L412](file:///g:/GitHub/Latent_Style/WEAVE/utils/dataset.py#L371-L412) |
| `internal_dynamics.py` | 内部梯度/门控诊断，训练中写 tensorboard/日志 | `probe_internal_dynamics` [L97](file:///g:/GitHub/Latent_Style/WEAVE/internal_dynamics.py#L97) |

## 3 下一代新增模块（默认全关）

| 文件 | 作用 | 开关 | 接入点 |
|---|---|---|---|
| `features.py` | VGG/DINOv2/tiny_conv 冻结特征 + patch 描述子 + 自相似 | 被 A/C 调用 | `build_feature_extractor` [L199](file:///g:/GitHub/Latent_Style/WEAVE/features.py#L199) |
| `swd.py` | sliced Wasserstein 风格损失（patch 化） | `output_supervision.enabled` | `PatchSWDStyleLoss` [L72](file:///g:/GitHub/Latent_Style/WEAVE/swd.py#L72) |
| `output_supervision.py` | 路线 A：latent → 解码 → 多层特征 → SWD/Gram/内容/自相似；沿 2-4 步 Euler 反传 | `bridge.output_supervision_*`（[config_schema.py#L896-L913](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L896-L913)） | `unroll_euler_decode` [L143](file:///g:/GitHub/Latent_Style/WEAVE/output_supervision.py#L143)；`OutputSupervision` [L303](file:///g:/GitHub/Latent_Style/WEAVE/output_supervision.py#L303) |
| `pseudo_targets.py` | 路线 B：教师适配 + 伪目标缓存读写 | `bridge.pseudo_target_cache_dir` | `PseudoTargetCacheWriter` [L155](file:///g:/GitHub/Latent_Style/WEAVE/pseudo_targets.py#L155) |
| `reference_encoder.py` | 路线 C：DINO patch token 参考编码 + InfoNCE + hybrid conditioner | `model.reference_*`（[L717-L721](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L717-L721)） | `ReferenceStyleEncoder` [L39](file:///g:/GitHub/Latent_Style/WEAVE/reference_encoder.py#L39)；`ReferenceConditioner` [L126](file:///g:/GitHub/Latent_Style/WEAVE/reference_encoder.py#L126) |
| `texture_module.py` | 路线 D：多尺度纹理 refiner（占位） | `model.texture_refiner_enabled`（默认 False） | `MultiscaleTextureRefiner` [L21](file:///g:/GitHub/Latent_Style/WEAVE/texture_module.py#L21) |

## 4 配置如何生效

1. `config.json` 可含 `_base`（字符串或数组），[load_config](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L1274-L1302) 递归合并（后者覆盖前者）。
2. `ExperimentConfig.from_mapping` 把 `model/bridge/data/training/inference/full_eval` 分到各自 dataclass；**未知键进 `extra` 并挂成属性**（[L105-L129](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L105-L129)）——这就是 `enable_hh_head` 没有显式 schema 字段却仍能生效的原因。
3. `run.py` 用 `config.data` 建 dataset，`config.model/bridge` 建模型（`WEAVE` 读 `model_cfg` 与 `bridge_cfg`），`config.training` 控制 loop 与 ckpt。
4. 推理默认值走 `config_schema.resolve_inference_section` / `resolve_full_eval_section`（[L1318-L1350](file:///g:/GitHub/Latent_Style/WEAVE/config_schema.py#L1318-L1350)），`inference*.json` 只做覆盖。

## 5 tools/ 与关键脚本

- 诊断：`diag_data_scale.py`（5k/18k 四格）、`diag_reference_swap.py`（参考图交换）、`diag_three_way.py`（VAE 重建/端点/最终输出）、`vae_recon_probe.py`（重建上限）。
- 数据：`preencode_d5_packed.py`、`fast_vae_encode_d5.py`、`taesd_encode_d5.py`、`encode_wikiart27_sd15.py`、`prepare_d5_512_clean.py`、`extract_dino_cache.py`。
- 评测：`run_eval_split.py`（600/150 拆分）、`merge_eval_metrics.py`、`batch_eval_wfi.py`。
- 路线 B：`gen_pseudo_targets.py`。
- 注意：`tests/_smoke_t26.py`（不在 `tools/`）以 `_` 开头，是临时探针，不属于工具集。

## 6 模型结构速览（`WEAVE.__init__` L435）

```
input latent (4ch)
  └─ Haar DWT → stack(ll,lh,hl,hh) 16ch → conv_in
  └─ time embed + ResBlocks (base_dim=64, 4 blocks)
  └─ style tokens: StyleConditioner(style_id→160d) [+ optional reference tokens]
  └─ cross-attention → trunk h
  └─ heads: v_ll, v_lh, v_hl (+ optional v_hh)
       └─ optional target_latent_hf_* residual deltas (pooled subband / spatial / texture)
  └─ Euler/RK4/Heun over t → iDWT → stepwise AdaIN → output latent
```

推理走 `integrate_transport`（[L1442](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1442)），训练走 `forward`（[L824](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L824)）；二者共用 `_solver_step`（[L1195](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1195)）。

## 7 代码与文档已知不一致

1. `WEAVE.forward` docstring 写"3 velocities, HH removed"，但代码在 `enable_hh_head=True` 时返回 `out["hh"]`（[model.py#L1189-L1191](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L1189-L1191)）——docstring 过期。
2. `config.json` 的 `style_condition_source="target_dino_patches"` 实际走 `style_memory`（模型注释在 [L446 附近](file:///g:/GitHub/Latent_Style/WEAVE/model.py#L446)），字段名误导。
3. `reference_encoder_in_dim=384` 是 DINOv2-small，但服务器缓存用 `dinov2-base`（768d），需按实际改配置。
