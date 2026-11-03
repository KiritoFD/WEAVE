# Spec: WEAVE 训练 infra 补丁包（P1–P9）

## 1. 目标

在不改动训练线正在运行的文件、不改变任何默认数值行为的前提下，交付一个**可应用**的 infra 补丁包：

- P1 冻结特征离线缓存（dinov2 content/style 特征）
- P2 aux 子采样 `K = 8–16`
- P3 分段 backward（FM 先反传，aux 后反传）
- P4 解码器 / 教师 grad checkpoint
- P6 probe 标量节流
- P7 启动校验（空闲显存 + `PYTORCH_CUDA_ALLOC_CONF`）
- P8 评测链快速模式（`runs/eval_one_fast.sh`，不改 `runs/eval_one.sh`）
- P9 `torch.compile` **保持关闭**，只留开关与判断依据

硬约束：

1. 全部新增开关默认值 = 现状（`False` / `0` / `""` / `1`）。
2. 默认路径（`config.json` 不变）下，计算图、随机数、loss 数值逐位不变。
3. `config.json`、`runs/eval_one.sh`、`runs/*.sh` 不修改；训练线文件只在**补丁文件**里描述修改。
4. P1 缓存只在 `output_supervision_feature_res > 0` 且与缓存记录的 `feature_res` 一致时生效，否则自动降级到在线路径并只告警一次。

## 2. 交付物

| 文件 | 类型 | 内容 |
|------|------|------|
| `docs/113/patches/infra_20261104.patch` | 新增 | 统一 diff（`git apply -p1` / `patch -p1` 可应用） |
| `tools/apply_infra_patch.py` | 新增 | 精确 old→new 替换执行器：`--check` / `--apply` / `--emit-diff` |
| `frozen_feature_cache.py` | 新增 | 运行时 memmap 查找（`FrozenFeatureCache`） |
| `tools/build_dino_feature_cache.py` | 新增 | 离线构建 P1 缓存 |
| `runs/eval_one_fast.sh` | 新增 | P8 快速评测链（不改 `eval_one.sh`） |
| `docs/113/INFRA_AUDIT_PATCH.md` | 更新 | 应用顺序、收益/风险、GPU 验证清单、b96 判断 |

## 3. 代码改动点（精确位置）

### 3.1 `config_schema.py`

- `BridgeConfig`（`output_supervision_chunk_rows` 之后）新增 6 个字段：
  `output_supervision_max_samples=0`、`output_supervision_split_backward=False`、
  `output_supervision_decoder_checkpoint=False`、`output_supervision_extractor_checkpoint=False`、
  `output_supervision_frozen_feats_dir=""`、`output_supervision_frozen_feats_dtype="float16"`。
- `TrainingConfig`（`log_interval` 之后）新增 `probe_log_interval: int = 1`。

### 3.2 `utils/dataset.py`

`__getitem__` 的 `item` dict 之后追加 6 个键（不改动任何已有键）：

- `target_cache_index`（实际 `t_idx`，含 pairing 选择后的结果）
- `content_cache_index`（`c_idx`）
- `target_cache_flip` / `content_cache_flip`（float32 0/1）
- `target_cache_valid`（`pairing_cache_dual_target_mix <= 0` 时 1，否则 0）

理由：P1 只能按“单图”缓存；dual-target lerp 行的 `target_style` 是混合图，必须让缓存失效。

### 3.3 `output_supervision.py`

- `unroll_euler_decode(..., decoder_checkpoint: bool = False)`：递归透传；最后一步
  `decoder.decode_image(h)` 在 `decoder_checkpoint and torch.is_grad_enabled()` 时用
  `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 包裹。
- `OutputSupervision.__init__` 新增 `decoder_checkpoint=False, extractor_checkpoint=False`。
- `forward`：`style_image` / `content_image` 允许为 `None`（配合 P1 跳过 style 解码）；
  新增 `style_feats` / `content_feats` 入参，命中缓存时跳过两次 `no_grad` 教师前向；
  `img_out` 的 `forward_layers` 在 `extractor_checkpoint` 时用 checkpoint 包裹（dict 输出，
  必须 `use_reentrant=False`）。
  `content_image is None` 且 `w_edge > 0` 时抛 `ValueError`（edge 损失确实需要像素）。

### 3.4 `trainer.py`

- `__init__`：在 Route A 构造块之后新增
  `_output_sup_max_samples` / `_output_sup_split_backward` / `_output_sup_cache`；
  仅当 `output_supervision_enabled` 且 `frozen_feats_dir` 非空时构建 `FrozenFeatureCache`。
- `_sup_cache_rows(batch, side, sel)`：新增辅助方法，返回全局行号列表；任一条不可缓存时返回
  `None` 并在整次 run 内只告警一次。
- `train_epoch` 每个 step 开头新增局部 `_sup_backward_done = False`。
- Route A 块替换为 P2 采样 + P1 缓存查找 + P3 分段反传的组合实现（见 3.5）。
- 主 backward 处增加 `if not _sup_backward_done:` 守卫。
- probe 段用 `probe_log_interval` 节流。

### 3.5 Route A 块的新行为（默认关闭时等价）

```
默认（max_samples=0, cache=None, split=False）:
  局部变量 = 整批; style/content 均在线解码 + 在线教师前向;
  一次 loss.backward()（走主 backward 原路径）
  与现状唯一的差别: loss_dict["loss_output_sup"] 仍是未 detach 的张量（保持原样）
```

### 3.6 `run.py`

`device` / `Seed` 日志后插入只读校验：`torch.cuda.mem_get_info()` 打印
free/total/used；`free < 9.5 GiB` 时 `logger.warning`；`PYTORCH_CUDA_ALLOC_CONF`
未设置时 `logger.warning`。不改变训练行为。

## 4. 验证方式（本机可执行，不碰训练线）

1. `python tools/apply_infra_patch.py --root . --check`：所有锚点必须**唯一命中**。
2. 在临时副本上 `--apply` 后 `python -m py_compile` 全部被改文件。
3. `--emit-diff` 产出统一 diff；反向 `git apply --check --reverse` 验证 diff 可回滚。
4. 合成小样本（`tiny_conv` extractor + `tiny` decoder，随机 latent）对比：
   - 默认路径：修改前后 `loss_dict` 全字段逐位一致；
   - P3 开启：`max|Δgrad| = 0`（两项相加的导数 = 导数相加）；
   - P1 开启（缓存由同一前向现场生成）：`max|Δfeat| = 0`（float32 缓存）；
   - P2 开启：梯度均值的期望不变，仅方差变大（不做逐位断言）。
5. 不启动任何 GPU 训练；GPU 侧验证项写入 `INFRA_AUDIT_PATCH.md` §验证清单。

## 5. 验收标准

- `--check` 在当前工作树 100% 命中（trainer.py / config_schema.py / utils/dataset.py /
  output_supervision.py / run.py 存在本地未提交修改，因此补丁基于**工作树**而非 HEAD）。
- 默认配置下不新增任何 `.item()` / `.cpu()` / 同步点；除 P7 的两条 warning 外无新增日志。
- `runs/eval_one.sh`、`config.json`、`runs/*.sh` 字节级未变。
