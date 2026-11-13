# V9-LD 4070 Laptop — 进展交接（2026-11-14 01:35，已更新）

> **更新**：解码协议已改为 **S10 decoder**（原版 decoder 文件已删，协议见根目录 `AGENTS.md`）。dev+test 全量 S10 评测已完成，结果见 `V9_LD_RESULTS.md`。下文中"VAE fallback 解码"相关描述为历史记录，以 S10 结果为准。

方案：`WEAVE_V9_LD_4070LAPTOP_6GB_ETA12H.md`。本文件记录已完成的工程与中间结果，供接手继续。

## 已完成

### 资产与缓存（全部就绪）

- **N2C latent 缓存**：从 3060 `runs/v8/N2C.e/gen/` 剥离 rgb_u8 后 94 个 batch（~119MB tgz），已拉回并合并为 `runs/v9_local_4070/cache/n2c_z.pt`（147MB，750 任务 × {z_src, z0, z_n2}，task_id 顺序与 `eval_protocol/tasks_750.json` 一致，已断言 750 unique）。
  - 远端源：`/mnt/i/Latent_Style/WEAVE/runs/v8/N2C.e/gen_latents`（WSL 路径，Windows scp 不可见；已 tar 到 `C:\Users\Administrator\gen_latents.tgz` 再拉）。
- **N2C ckpt**：`runs/v8/train/N2C/step_04000.pt`（117KB，SPDHead rank=0 对角预条件）。
- **Patch Bank**：`runs/v9_local_4070/bank/style_bank.pt`（200KB），5 画派 × K=128 原型，特征维 77（Haar-LL + HF 差分统计 + 通道 mean/std），由 `weights_local/sd_pool_orig`（3060 真品 latent，490MB 字节级校验）CPU/GPU 混合构建，5 秒完成。**注意：patch_feats 的 patch 网格索引曾修正过（permute 0,2,4,1,3,5），bank 是修复后重建的。**
- **manifest**：`runs/v9_local_4070/manifest.json`（ckpt/tasks/bank sha256、超参、解码路径说明）。

### 代码（`exp/v9_local/`）

| 文件 | 作用 |
|---|---|
| `patch_bank.py` | 8×8 patch 特征 + kmeans 原型（已验证 patch 索引正确性） |
| `local_style.py` | 可微局部风格能量 L_loc（softmin-NN 能量 + wrong-style hinge），`patch_scores` 供区域选择 |
| `direction.py` | 候选方向 {r1, local, mix}（R1 方向 = N2C SPDHead 预条件 × margin 梯度）、content_risk、region_mask（gain/risk top-25%） |
| `solver.py` | 预算化 GD：cost_tot≤ε=0.10 硬约束 + proxy（margin−0.3·L_loc）回溯接受；sid==tid 短路；越预算回滚 z_N2 |
| `run.py` | LD0–LD4 latent 推理（不碰 VAE/指标），dev/test 划分（md5(source_stem)%5==0 → dev=170，test=580），断点续跑 |
| `cache_n2c.py` | gen_latents batch → n2c_z.pt 合并 |
| `decode_and_eval.py` | z_final → VAE 解码 rgb_u8（diffusers fallback，无 TRT）→ `src.eval.measure` 标准评测 |
| `check_direction.py` | 方向有限差分验证（已通过） |
| `manifest.py` | provenance |

### 方向验证（P1 gate，已通过）

- `L_loc(z−η∇L_loc) < L_loc(z)`：η=1e-3/1e-2/5e-2 全部下降 ✅
- N2C 输出上 E_t=34.00 < E_neg=39.69（margin +5.70）→ 局部能量确实指向目标画派 ✅

## Dev 结果（170 任务，cross n=136）——`runs/v9_local_4070/dev/*/eval/eval.json`

| arm | dino_s | dino_c | lpips | top1 | G | affinity |
|---|---:|---:|---:|---:|---:|---:|
| LD0 | .4127 | .7735 | .5098 | .3235 | −.0167 | .5478 |
| LD1 | .4133 | .7740 | .5094 | .3309 | −.0166 | .5482 |
| LD2 | .4127 | .7735 | .5098 | .3235 | −.0167 | .5478 |
| LD3 | .4105 | .7783 | .4954 | .2941 | −.0206 | .5477 |
| LD4 | .4149 | .7731 | .5050 | .3309 | −.0159 | .5495 |

判读：LD1/LD2 采纳率极低（R1 margin 已饱和，ε=0.10 预算内几乎无可改进候选）；LD3 用内容换风格；**LD4 是唯一多指标同向改善的臂**（ds/top1/G/affinity 升、lpips 降），但幅度 ~.002，需 test750 判决。

采纳统计（dev）：LD1 acc 8/170、LD2 acc 1/170、LD3 acc 101/170、LD4 acc 92/170；"over" 字段实为"无改进候选回滚"，非预算违规。

## 进行中 / 待办

1. **Test750 评测**：latent 推理已全部完成（test split：LD0–LD4 各 750/580 任务 batch 文件在 `runs/v9_local_4070/test/<arm>/`）。`decode_and_eval --split test` 正在后台跑（shell `d33c92`），完成时 LD0/LD1 已出 eval.json，LD2–LD4 排队中。**接手先 `get_output d33c92` 或直接重跑缺的臂**（decode ~0.15s/img、measure ~0.06s/img，单臂 ~5min）。
2. test 结果汇总 → 对比 dev → 写正式结果文档（本目录 `V9_LD_RESULTS.md`）。
3. LD5（学习式门控）按方案为条件执行：若 LD4 test 增益不显著或时间不够可跳过并在报告注明。
4. 端到端计时：当前数据是 latent-cache 消融，**不能当 WEAVE 完整推理延迟**；如需端到端需另跑 B4+N2C+LD 全链路并单独计时。
5. 解码为本地 VAE fallback（无 TensorRT），与 3060 官方路径有 ~8.6/255 MAE 差异——所有臂同路径、相对比较有效，绝对值与历史 N2C.e（.4358/.7830）不可直接比。

## 已知坑

- `run.py --resume` 的 batch 文件名按 todo 位置编号，smoke 小样本会与正式批次**同名覆盖**——dev 的 LD0/LD4 曾因此缺 8/2 个任务，已删目录重跑修复。test split 是一次跑完的，无此问题，但若中断续跑需注意核对 done.json 与文件数。
- `split_indices` 用 md5(source_stem)%5==0，dev=170/test=580，与方案"dev 选参、test750 正式"的口径一致（test 非全 750，报告时注明 n=580 cross 口径）。
- solver 的 `over` 标志语义偏松（=首轮无接受），报告时按"回滚率"解读。
- 3060 链路 ~80–125KB/s，只用于拉缓存；本实验不再需要 3060。
