# WEAVE 两大进展（2026-11）

> **2026-11-14 历史快照**：本文记录 S10＋Q2A/Oracle 这两项当时的关键突破，不再代表 V19–V26 的“当前最强”。最新看 [根 README](../README.md)、[VAE 专题](vae/README.md) 和 [文档索引](README.md)。旧测量、阈值与复现记录仍保留。

---

## 1. S10 剪枝 Decoder —— 已取代原版 SD-VAE decoder，成为全仓唯一解码口径

**一句话**：把 SD-VAE decoder 从 49.49M 剪到 **10.66M（−78.5%）**，2.86× 加速，
且在固定 latent 的 Cross600 中**提高 Top-1 和 Margin、改善 LPIPS，但 DINO-C/S 各下降约 0.01**——其价值是统一研究协议下的**实际加速和质量权衡**，不是无损全指标支配。

| 证据（Q2A+Oracle1 × 750，cross600，配对 bootstrap） | S10 (old_s10) | 原版 sdvae | 结论 |
|---|---:|---:|---|
| CSD Target Top-1 ↑ | **.3767** | .3450 | **+3.2pp**，bootstrap CI 排除 0 |
| CSD Target Margin ↑ | **−.0411** | −.0551 | S10 胜 |
| LPIPS ↓ | **.5034** | .5149 | S10 胜 |
| same150 Top-1 ↑ | **.96** | .94 | S10 胜 |
| DINO-S / DINO-C | 各 −.010 | — | 唯一代价：风格表征略降 |

其它关键事实：

- 剪枝方法：按 4 条通道流做结构化剪枝（一阶 Taylor |ΣW·dW| 流内重要性），宽度
  [64,128,192,256]，5 阶段恢复训练（P1→P3，L1+0.5·LPIPS），重建 LPIPS 0.0926。
- 速度：decoder 段 15.57ms（4090，B1），**2.86×**；onnx/TRT 部署已验证。
- S10B（`deliver/checkpoints/s10b_standalone/`）与 `s10_p5_lowlr` **字节级同一权重**，
  只是为 SD0 未缩放 latent 打包的独立加载器。
- 历史文档中"正式质量用标准 SD-VAE 口径；S10 仅部署对照"（docs/v8 系列）**已作废**。

**使用契约（错了会出垃圾图，历史上踩过）**：

```python
from src.utils import load_s10_decoder, decode_s10
s10 = load_s10_decoder("cuda")     # StudentDecoder(pqc, dec).half()
rgb01 = decode_s10(s10, z_scaled)  # z 是 repo 存储的 scaled latent；内部自动 /0.18215
```

1. 输入是 repo 存储的 **scaled** latent，喂 `decode_s10` 前不要再除 0.18215（已封装）；
2. `post_quant_conv` 用 checkpoint 自带的，**恰好应用一次**（StudentDecoder 内部已含）;
3. 禁止调用 `AutoencoderKL.decode()` / `vae.decode()`——原版 decoder 文件已删除。

**权威文档**：[docs/vae/01_S10_TRAINING_FULL.md](vae/01_S10_TRAINING_FULL.md)（训练全记录）·
[docs/vae/00_OVERVIEW.md](vae/00_OVERVIEW.md)（决策链）·
[docs/vae/S10_READY_TO_USE.md](vae/S10_READY_TO_USE.md)（快速上手）·
[docs/v6/Q2A_ORACLE_DECODER_RANKING.md](v6/Q2A_ORACLE_DECODER_RANKING.md)（4-decoder 配对证据）·
[docs/vae/09_S10B_DECISION.md](vae/09_S10B_DECISION.md)（S10B 决策）

**Checkpoint**：`deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt`
（42682134 B，md5 `d4f639a1a82bb5ba0115fffaaac85a0b`）

---

## 2. Q2A + Oracle —— 早期强风格控制基线与 750 任务协议起点

**一句话**：218K 参数的内容条件化 latent 算子（Q2a embed-style）+ 推理期沿 R1 风格
critic 的 target-margin 方向做**一步有约束梯度步**（Oracle 路线），在 32 方法排行榜上
以 **Affinity 0.5587 / CSD Top-1 36.3% 排名第 4**（前 3 名都是全图 StyleShot/StyleID/
AesPA 类基线），并以此协议冻结了全仓评测口径。

- **任务集**：`eval_protocol/tasks_750.json` = Cross600（150 源 × 4 跨风格目标）+ Same150，
  5 画派（Early_Renaissance / Impressionism / Minimalism / Rococo / Ukiyo_e）。
- **指标定义**：DINO-S（DINOv2-small CLS 对 30 张冻结目标风格参考的最大余弦）、
  CLIP-S（CLIP ViT-B/32 原型余弦）、Style Affinity = (DINO-S + CLIP-S)/2、
  CSD Target Top-1 / Margin（对 5 风格原型）、DINO-C、LPIPS；统一 `paper_rgb_v1` 协议，
  cluster bootstrap 95% CI（150 源聚类）。
- **冻结证据文件**：`deliver/q2a_oracle_scores_baseline_protocol.json`（2026-11-12），
  逐任务分 `deliver/q2a_oracle_per_task_baseline_protocol.json`。
- **为什么重要**：它给了 S10 decoder 一把"同协议尺子"（见上表），也是
  [docs/LEADERBOARD.md](LEADERBOARD.md) 上 "Ours" 系列的默认口径；R1 margin 与真实
  分类错位的现象（margin↔Top-1 脱钩）是后续 V9-LD 消融的直接动机。

**权威文档**：[docs/v6/Q2A_ORACLE_BASELINE_RANKING.md](v6/Q2A_ORACLE_BASELINE_RANKING.md)（协议
定义 + 32 方法校验）· [docs/v6/Q2A_ORACLE_DECODER_RANKING.md](v6/Q2A_ORACLE_DECODER_RANKING.md)·
[docs/1111/02_EVALUATION_PROTOCOLS.md](1111/02_EVALUATION_PROTOCOLS.md)（协议族谱系与跨期差异）·
[docs/LEADERBOARD.md](LEADERBOARD.md)（权威排行榜）

---

## 当前权威交付与演进状态（2026-11-18）

- **全量数据与最新最优模型**：见 [`performance/MASTER_LEDGER.md`](performance/MASTER_LEDGER.md)（215+ 实测臂总台账）与 [`icme2027/README.md`](icme2027/README.md)（`PRISM` 独立论文手稿）。
- **分代演进**：[`v18/V18_RESULTS.md`](v18/V18_RESULTS.md)（单步 `38.00%` 超越 `Seedream 4.5`）$\to$ [`next1118/README.md`](next1118/README.md)（`exp/v19/` 正交商—纤维对偶免源标签单步与各向异性 `S10-VGG` 谱传输）。
