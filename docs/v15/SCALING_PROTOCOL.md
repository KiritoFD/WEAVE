# V15 Scaling 实验预注册（2026-11-15，用户issued）

**目的**：把纯 Conv U-Net 从 3.16M 探到 ~35M，回答两个独立问题——
(a) 容量是否继续改善 Cross600 DINO-S/DINO-C（拟合绘画分布）；
(b) 容量能否改善目标画派迁移（Target Top-1/Margin）。
Group A 证据：dim96 尚未明确容量饱和（dino_s 0.46→0.56 未平台），但 (b) 全线随机——本轮分辨"拟合容量"与"编辑能力"。

## 配置 v2（2026-11-15 用户修订：全部 8k updates、显存统一顶到 ~10GB、上探两档）

全部：纯 Conv、无 attention、无 where、全局 AdaLN、eff. batch 128、seed 0。
micro 受"eff. batch 128 ⇒ micro 必须整除 128"约束，10GB 目标按实测线性显存模型（peak ≈ 1517MB + 0.409MB×dim×micro）逐档逼近；@128 会越限的档位取安全上限并标注。

| 实验 | base_dim | 预计参数* | micro/accum | 预测峰值 | updates | out |
|---|---|---|---|---|---|---|
| 基线（4k 对照）| 96 | 3.16M（实测）| 128/1 | 6.5GB | 4000 | Group A 已有 |
| **B8k-96** | 96 | 3.16M | 128/1 | 6.5GB | 8000 | scaling/B8k_dim96 |
| S1 | 128 | 4.13M（实测）| 128/1 | 8.2GB | 8000 | scaling/S1_dim128 |
| S2 | 160 | 6.30M（实测）| 128/1 | **9.9GB** | 8000 | scaling/S2_dim160 |
| S3 | 192 | 8.92M（实测）| 64/2 | 6.5GB（@128=11.6GB 越限）| 8000 | scaling/S3_dim192 |
| S4 | 256 | 15.52M（实测）| 64/2 | 8.2GB | 8000 | scaling/S4_dim256 |
| S5 | 320 | 23.95M（实测）| 64/2 | **10.0GB** | 8000 | scaling/S5_dim320 |
| S6 | 384 | ~33M | 32/4 | 6.5GB（@64=11.6GB 越限）| 8000 | scaling/S6_dim384 |
| S7 | 512 | ~53M | 32/4 | 8.2GB（@64=14.9GB 越限）| 8000 | scaling/S7_dim512 |

*参数量以实测 `n_params()` 为准；实测呈次二次（幂律指数 ~1.68），dim384/512 待冒烟实测。
**8k 对照只保留 dim96**（与 Group A 4k 构成训练长度轴）；原 B8k_dim192/320 在 v2 下与 S3/S5 配置完全相同，冗余，删除。
**五组+两档全跑，不做中途判停。** 8k 对照为全新训练（非 resume：`train.py` 未实现优化器/调度器/RNG 状态恢复，旧权重续训不构成严格续训）。
**ETA 注意**：v2 下大档位是计算受限（非 v1 的 Python 开销受限），dim320≈数小时、dim384/512 可能 5–20h/臂；以实跑曲线为准（train_curves.jsonl 每 200 步回传）。

## 评测协议

- Test750 全量，τ0 = 0 / 0.4 / 0.7 三档（ε 按 (task, level) 与 Group A 跨臂配对）。
- **重点关注 τ0=0.7**（与 WEAVE 风格—内容工作区间最接近）。
- 读数与 Group A §2.4 对齐：cross600 的 dino_s / dino_c / style_top1，另加 CSD top1/margin（本地裁判补齐）。
- **延迟口径**：分别记录 U-Net 采样耗时、VAE（S10）解码耗时、I/O 落盘耗时（eval_arms 已埋点，写入 `runs/v15/eval/timing.json`）。

## 两条必须记录的代码事实（本轮不改，保持与 Group A 严格可比）

1. **V15 不是显式内容条件化的 WEAVE**：接口为 `v(z_tau, s, tau)`，训练目标是画派条件生成 Flow，无独立输入 Canonical Content `c_src`。跨画派编辑时输入源画派 noisy latent、按目标画派条件生成——这一组合在原始训练分布中未直接出现。因此 Group A 与本轮的 Top-1 读数同时受此目标形态限制。
2. **训练集未实现五画派均衡采样**：`train.py` 用普通 `DataLoader(shuffle=True)`，实际类别分布 Impressionism 8089/13992 = **57.8%**（ER 1319 / Min 997 / Roc 2264 / Uki 1323）。均衡采样与显式内容条件化留作后续受控实验，避免与容量效应混淆归因。

## 分析计划（评测完成后）

| 横轴（log） | 纵轴 | 问题 |
|---|---|---|
| 参数量 | Cross600 DINO-S、DINO-C | 容量收益是否饱和 |
| 参数量 | Target Top-1、Margin | 画派特异性是否随容量出现 |
| 端到端延迟（U-Net / VAE / 总） | DINO-S / DINO-C / Top-1 | 计算投入是否值得 |

**判读**：若 DINO-S/C 持续升而 Top-1 不动 → 量化"更大 U-Net 更好拟合绘画分布，但现有训练目标不能把容量转化为指定画派编辑能力"。

## 运行

- 冒烟：`python -m exp.v15.smoke_scaling --pool <pool>`（实测 params/peak/s-per-update）
- 串行：tmux session `scaling`，`_tmp/run_v15_scaling.bat` → `python -m exp.v15.run_scaling`
- 预计串行总时长 ~24–30h（dim320 micro=4 受 GPU 利用率限制，ETA 以冒烟实测为准）
