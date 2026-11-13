# V9-LD Results — S10 decoder protocol（2026-11-14）

实验：`WEAVE_V9_LD_4070LAPTOP_6GB_ETA12H.md`；交接与工程细节见 `V9_LD_PROGRESS.md`。
**解码协议：全部结果使用 S10 decoder（`s10_p5_lowlr/decoder.pt`），不用原版 SD-VAE decoder。** 原版 decoder 文件已删除，协议已写入根目录 `AGENTS.md`。

## 协议要点

- 起点：N2C `step_04000.pt` 的 latent 缓存（3060 远端 gen 剥离 rgb_u8，94 batches → `runs/v9_local_4070/cache/n2c_z.pt`，750 任务）。
- LD 在 latent 上做预算化梯度下降（cost_tot≤ε=0.10，proxy=margin−0.3·L_loc，K=2 步，η∈{0.5,1}，mask top-25%），不重新跑 B4/N2C。
- Patch Bank：5 画派 × K=128 Haar-patch 原型（`runs/v9_local_4070/bank/style_bank.pt`，sha256 见 `manifest.json`）。
- 方向验证已通过：L_loc 有限差分下降 ✅；E_t=34.00 < E_neg=39.69 ✅。
- dev/test 按 md5(source_stem)%5 划分（dev=170 / test=580，互补并集=750）；下表 test 为正式结果。

## Test 结果（cross 子集，n≈464；same150 见 eval.json）

| arm | dino_s | dino_c | lpips | top1 | margin | G | affinity |
|---|---:|---:|---:|---:|---:|---:|---:|
| LD0 (N2C) | .4462 | .7739 | .4864 | .3341 | −.0560 | −.0170 | .5633 |
| LD1 (+R1) | .4464 | .7750 | .4857 | .3341 | −.0567 | −.0168 | .5636 |
| LD2 (R1+mask) | .4460 | .7741 | .4863 | .3362 | −.0559 | −.0171 | .5632 |
| LD3 (local 全域) | .4401 | **.7822** | **.4766** | .3190 | −.0660 | −.0173 | .5615 |
| **LD4 (mask+全方向)** | .4461 | .7748 | .4836 | **.3556** | −.0547 | **−.0161** | **.5644** |

Dev（n=136 cross）方向一致：LD4 是唯一多指标同向改善臂。

## 判读

1. **LD4 是本轮唯一净胜方**：vs LD0，top1 **+.0215**、lpips −.0028、G +.0009、affinity +.0011，dino_s/dino_c 持平。局部选择 + 正确方向 + 受约束回溯确实有效，主要增益体现在真实画派分类（Top-1）而非 margin。
2. **LD1/LD2 ≈ LD0**：R1 margin 在 N2C 输出上已饱和（dev 采纳率 8/170、1/170），额外 R1 步在 ε=0.10 预算内几乎无可改进候选——再次印证"margin↔真实分类错位"。
3. **LD3 重演已知 trade-off**：dino_c +.008、lpips −.010 但 dino_s −.006、top1 −.015——无约束局部风格梯度买内容卖风格。
4. **V9 目标核对**（vs V8 N2C.e 历史 sdvae 口径 .4358/.7830/.328）：S10 口径下 LD0 已达 dino_s .4462/top1 .3341，但**跨 decoder 不可直接比**；LD4 top1 .3556 为本仓库 latent-cache 消融的最高值。

## 偏差与限制

- latent-cache 消融，**不代表端到端 WEAVE 推理延迟**；S10 解码 ~0.1s/img、measure ~0.06s/img 仅供参考。
- S10 vs 原版 decoder 的绝对指标差异是协议级选择（S10 全面更优，见 AGENTS.md 证据），所有臂同口径、相对比较有效。
- test split 为 580/750 任务（dev 170 互补）；若需严格 750 单点结果，把 dev+test 两个 eval.json 按 task_id 合并即可（无重叠）。
- solver `over` 字段=首轮无接受回滚，非预算违规；回滚样本保持 z_N2，未删除。

## 产物索引

- latent 输出：`runs/v9_local_4070/{dev,test}/<arm>/batch_*.pt`（含 z_final+infos）
- 解码+指标：`runs/v9_local_4070/{dev,test}/<arm>/{gen,eval}/`
- provenance：`runs/v9_local_4070/manifest.json`（ckpt/tasks/bank sha256、超参）
