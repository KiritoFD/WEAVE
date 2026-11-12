> 后续修正：本报告的 Top-1 是 **CSD Top-1**。DINO-S / CLIP-S 的补评并不总同向，参见 [多指标复核与归因](STYLE_ATTRIBUTION.md)。不能把这里的 CSD 下降概括为所有意义上的画风退化。

# Ranked WEAVE: paired VAE replacement evaluation

结论（冻结现有 WEAVE 权重，成对替换 Encoder/Decoder）：

- 没有一种替换 VAE 在这五个上榜配置上获得稳定的 CSD Top-1 提升。
- A0：内容保留略升、LPIPS 略降，Top-1 全部下降；Oracle/B3/B4 的下降通过源图聚类配对检验。
- S10：Oracle/B3/B4 提升 0.83–1.50 个百分点，95% CI 均包含 0；ScoreDiff N24/N12 则显著下降 3.50/4.67 个百分点。此处仅是 CSD 结论；DINO-S / CLIP-S 的不同结论见后续复核。
- C_edit：五项 Top-1 均显著下降。这是直接替换的兼容性结果，不代表在新 latent 空间重新训练 WEAVE 的上限。
- 本轮新增图片 0 张。每配置仅使用冻结的 750 个任务，600 cross 与 150 same 分开汇总。


750 frozen tasks per configuration: 600 cross / 150 same. No new image files.
Raw RGB source reference; CSD ViT-L, DINOv2-small, AlexNet LPIPS.
All VAE variants use paired encoder/decoder, with frozen WEAVE weights.
The sdvae baseline is measured in this run (PyTorch BF16); historical TRT numbers are separate.

Historical eval_tiers.py feeds [0,1] images directly to VAE.encode and uses round-trip references. This run uses the VAE's [-1,1] input and raw RGB metric references; compare VAE variants against this run's sdvae, not against the archived ranking.

| Model | VAE | Top-1 | change pp [95% CI] | DINO-C | LPIPS | Same dS |
|---|---|---:|---|---:|---:|---:|
| Oracle1 | sdvae | 34.83% | baseline | 0.7307 | 0.5222 | -0.1488 |
| Oracle1 | a0 | 32.67% | -2.17 [-4.00, -0.17] | 0.7365 | 0.5132 | -0.1465 |
| Oracle1 | s10 | 36.33% | +1.50 [-1.17, +4.33] | 0.7345 | 0.5014 | -0.1657 |
| Oracle1 | cedit | 19.00% | -15.83 [-19.50, -12.00] | 0.7138 | 0.6758 | -0.2549 |
| B4 | sdvae | 21.33% | baseline | 0.8199 | 0.4523 | -0.0209 |
| B4 | a0 | 17.17% | -4.17 [-6.17, -2.17] | 0.8259 | 0.4428 | -0.0233 |
| B4 | s10 | 22.17% | +0.83 [-2.00, +3.50] | 0.8152 | 0.4396 | -0.0624 |
| B4 | cedit | 13.00% | -8.33 [-11.50, -5.50] | 0.7843 | 0.6535 | -0.1914 |
| B3 | sdvae | 17.50% | baseline | 0.8325 | 0.4397 | -0.0074 |
| B3 | a0 | 14.83% | -2.67 [-4.33, -1.17] | 0.8386 | 0.4302 | -0.0103 |
| B3 | s10 | 18.83% | +1.33 [-1.33, +4.00] | 0.8270 | 0.4282 | -0.0509 |
| B3 | cedit | 12.33% | -5.17 [-8.33, -2.33] | 0.7947 | 0.6504 | -0.1822 |
| ScoreDiff_N24 | sdvae | 32.00% | baseline | 0.6308 | 0.5751 | +0.0266 |
| ScoreDiff_N24 | a0 | 31.50% | -0.50 [-2.33, +1.17] | 0.6386 | 0.5650 | +0.0207 |
| ScoreDiff_N24 | s10 | 28.50% | -3.50 [-5.50, -1.33] | 0.6369 | 0.5217 | -0.0054 |
| ScoreDiff_N24 | cedit | 24.50% | -7.50 [-10.00, -4.67] | 0.6384 | 0.5714 | -0.0060 |
| ScoreDiff_N12 | sdvae | 27.50% | baseline | 0.6776 | 0.5219 | +0.0266 |
| ScoreDiff_N12 | a0 | 26.33% | -1.17 [-2.67, +0.17] | 0.6843 | 0.5108 | +0.0207 |
| ScoreDiff_N12 | s10 | 22.83% | -4.67 [-6.50, -2.67] | 0.6797 | 0.4738 | -0.0054 |
| ScoreDiff_N12 | cedit | 18.67% | -8.83 [-11.17, -6.50] | 0.6845 | 0.5185 | -0.0060 |
