# V15 Scaling 实验预注册（2026-11-15，用户issued）

**目的**：把纯 Conv U-Net 从 3.16M 探到 ~35M，回答两个独立问题——
(a) 容量是否继续改善 Cross600 DINO-S/DINO-C（拟合绘画分布）；
(b) 容量能否改善目标画派迁移（Target Top-1/Margin）。
Group A 证据：dim96 尚未明确容量饱和（dino_s 0.46→0.56 未平台），但 (b) 全线随机——本轮分辨"拟合容量"与"编辑能力"。

## 配置（全部：纯 Conv、无 attention、无 where、全局 AdaLN、eff. batch 128、seed 0）

| 实验 | base_dim | 预计参数 | micro/accum | updates | out |
|---|---|---|---|---|---|
| 基线 | 96 | 3.16M（已测） | 128/1 | 4000 | Group A 已有 |
| S1 | 128 | ~5.6M | 64/2 | 4000 | runs/v15/scaling/S1_dim128 |
| S2 | 160 | ~8.8M | 32/4 | 4000 | runs/v15/scaling/S2_dim160 |
| S3 | 192 | ~12.6M | 16/8 | 4000 | runs/v15/scaling/S3_dim192 |
| S4 | 256 | ~22.5M | 8/16 | 4000 | runs/v15/scaling/S4_dim256 |
| S5 | 320 | ~35.1M | 4/32 | 4000 | runs/v15/scaling/S5_dim320 |
| B8k-96 | 96 | 3.16M | 128/1 | 8000 | runs/v15/scaling/B8k_dim96 |
| B8k-192 | 192 | ~12.6M | 16/8 | 8000 | runs/v15/scaling/B8k_dim192 |
| B8k-320 | 320 | ~35.1M | 4/32 | 8000 | runs/v15/scaling/B8k_dim320 |

- micro 为保守初值；OOM 时减半 micro / 加倍 accum（eff. batch 128 不变）。
- 参数量以实测 `n_params()` 为准（冒烟脚本 `exp/v15/smoke_scaling.py`）。
- **五组全跑，不做中途判停。** 8k 对照为全新训练（非 resume：`train.py` 未实现优化器/调度器/RNG 状态恢复，旧权重续训不构成严格续训）。

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
