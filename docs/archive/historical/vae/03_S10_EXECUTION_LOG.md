# S10 Decoder 12h 计划执行记录

**起始**：2026-11-09 22:37
**方案**：跳过教师蒸馏；联合目标 `L1 ≤ 0.075`、`LPIPS ≤ 0.10`、`CSD_keep ≥ 0.90`

---

## P0（完成）：13k 池上的感知精修

初始化 = `s10_p1_l1`（L1 恢复 checkpoint），Loss = `L1 + 0.5·LPIPS@512`，lr 2e-5，600 步。

| 指标 | A0 | S10 起点 | P0 终点 | 目标 |
|---|---:|---:|---:|---:|
| L1 | 0.0646 | 0.0659 | **0.0720** | ≤0.075 ✅ |
| LPIPS@512 | 0.0772 | 0.3902 | **0.1099** | ≤0.10 |
| CSD Keep | 0.9162 | 0.6080 | **0.8484** | ≥0.90 |

轨迹：LPIPS 0.1320→0.1197→0.1147→0.1117→0.1107→0.1099；CSD 0.8270→0.8396→0.8438→0.8476→0.8477→0.8484。
**最后三点 |ΔLPIPS| < 0.002、ΔCSD < 0.005 → 触发方案的平台早停规则，进入 P1。**

### 两个关键确认

1. **CSD 从未参与训练，却从 0.608 自动回升到 0.848。** 验证「LPIPS 恢复后 CSD 会跟随恢复，不需要专门风格 Loss」。
2. **256px LPIPS 监督无效。** 先跑过 500 步 256px 版：512px holdout LPIPS 仅 0.3026→0.2888（−4.6%），而训练侧 256px LPIPS 已到 0.05。改用 512px 后同样 500 步做到 0.390→0.110。**必须用 512px。**

### 一个已修的 bug

选择 checkpoint 的守卫原为 `L1 ≤ L_prune × 1.05`，把 L1 略升但 LPIPS 减半的最优模型全部拒掉（`best_sel=inf`，不存盘）。已改为 `L1 ≤ max(L_prune×1.15, 0.075)`，并保证无 checkpoint 通过时仍保存最终权重。

---

## P1（完成）：全量 latent 缓存 + 数据隔离

### 发现严重数据泄漏

`RawMemmapWikiArtDataset(is_training=True)` 的过滤条件是 `raw_index not in val_indices`，**完全忽略 `in_train` 标志**：

```
total records:            102,255
in_train=True:             78,661   ← 真正训练池
in_train=False:            23,594   ← 测试集（含 2,048 冻结验证）
dataset(is_training=True): 100,207  ← 78,661 + 21,546 张泄漏测试图
```

**已有的 12,991 缓存中 2,640 张（20.3%）是泄漏测试图**，干净行 10,351。

**对 P0 结论的影响**：无。评测用的 2,048 holdout 全部来自 `in_train=False` 且被 `val_indices` 排除，训练从未见过，所以 P0 的 L1/LPIPS/CSD 是在真正未见数据上测的。但训练池本身不干净，P2 已修正。

### 修正后的缓存（`runs/vae_decoder_prune/full/`）

显式过滤 `in_train=True and raw_index not in val_indices`，并做完整校验：

```
[Split] PROPER train pool = 78,661（断言：无重复、无 val 重叠）
[Train] latent_train_fp16.bin 2.40 GB in 34.1 min
[Shards] problem shards: 0          ← 无全零行、无 NaN/Inf
[Holdout] 2,048 latents, matches val_indices
[Verify] re-encode cos(cached, fresh): min=0.99998 mean=0.99999
```

| 文件 | 内容 |
|---|---|
| `full/latent_train_fp16.bin` | 78,661 × 16384 fp16（2.40 GB） |
| `full/raw_index_train.npy` | latent 行 → memmap 行 |
| `full/latent_holdout_fp16.bin` | 2,048（冻结 val） |
| `full/split.json` | decoder_train 兼容格式 |

原 12,991 缓存与 `split.json` **未被覆盖**。

---

## P2（进行中）：全量真实图像精修

配置：初始化 P0 权重；lr 3e-5 cosine→3e-6（warmup 20 步）；有效 batch 64；`L1 + 0.5·LPIPS@512`；预算 8000 步（≈5.8h）；平台早停规则生效。

| 更新 | L1 | LPIPS | CSD |
|---:|---:|---:|---:|
| 100 | 0.07179 | **0.1049** | 0.8476 |

速度 2.23–2.61 s/upd（峰值 28.7 img/s）。

---

## 待执行

- **P2 收敛/平台** → 保存 `best_lpips` 与 `best_pareto` 两个 checkpoint
- **P3 验收**（脚本已就绪 `tools/vae_suite/exp_p3_final_eval.py`）：重建指标、实验 B/C 风格响应复测、**梯度代理 cos(g_T,g_S)**、效率（forward / forward+input-grad / 峰值显存）
- **P4-L0**（脚本已就绪 `tools/vae_suite/exp_l0_latent_direction.py`）：按内容匹配的跨风格真实图像对构造 `Δz = E(x_t) − E(x_s)`，回答
  - Q1 可复现性：同一风格对不同内容对的 `Δz` 是否一致（vs 随机方向基线）
  - Q2 可迁移性：在内容对 #1 上估的 `Δz` 能否把**另一张**风格 s 的图推向风格 t（A0 解码，排除 decoder 混淆）
