# WEAVE 两大进展（2026-11）

> 本文是"当前最值得记住的两件事"的一页纸索引。所有数字都有出处（见文内链接），
> 协议级结论以根目录 `AGENTS.md` 的决策协议（2026-11-14）为准。

---

## 1. S10 剪枝 Decoder —— 已取代原版 SD-VAE decoder，成为全仓唯一解码口径

**一句话**：把 SD-VAE decoder 从 49.49M 剪到 **10.66M（−78.5%）**，2.86× 加速，
且在同协议评测里**全面优于**原版 decoder——因此不是"部署替换"，而是协议级选择。

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
- 速度：decoder 段 15.57ms（4090，B8），**2.86×**；onnx/TRT 部署已验证。
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

## 2. Q2A + Oracle —— 当前最强风格控制路线 + 冻结的 750 任务评测协议

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

## 当前工作（2026-11-14）

- **V9-LD 已收尾**：latent-cache 消融（LD0–LD4，RTX 4070 本地，S10 口径）结论——
  **LD4（mask+全方向）是唯一净胜臂**（test580 Top-1 .3556，vs LD0 +.0215）；
  见 [docs/v9/V9_LD_RESULTS.md](v9/V9_LD_RESULTS.md)。
- 3060 被他人占用做 StyleGallery 复现，**不要碰**（复现记录在
  [docs/baseline/StyleGallery/](baseline/StyleGallery/)）。
