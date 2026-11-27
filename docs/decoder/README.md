# Decoder｜S10B：压缩与编辑信号响应

> 10,657,943 参数（终稿 ledger），原版49.49M，参数减少78.5%。S10B 是旧 S10 同权重独立打包，不是另一次训练、不是 C1 响应蒸馏模型。

## 1. 为什么解码器是独立贡献

生成器输出的是编辑 latent，不是自然重建 latent。同一编辑 latent 在不同 decoder 上可能产生不同的源纹理残留、目标判别与感知表现。剪枝不仅降低计算，还改变了输出响应。必须固定 Encoder、最终 latent 与裁判，只替换 decoder，才能归因。

S10 采用 Taylor 结构剪枝与真实图 L1/LPIPS 感知恢复，最终 widths_forward=[64,128,192,256]（standalone 配置）。不依赖额外生成模型。历史 C0/C1 重建更接近官方，但下游目标判别反而不及旧 S10。

## 2. 最新固定 latent 对照：统一 P8 latent（P9b）

| 工作点 | D0 原版 Top-1 | D1 WikiArt FT Top-1 | D2 S10B Top-1 | D2−D0 |
|---|---:|---:|---:|---:|
| E+A3 l090 | 48.17% | 50.67% | 57.33% | +9.17pp [6.50,11.83] |
| P s085 | 50.50% | 53.67% | 57.83% | +7.33pp [4.67,10.17] |
| P s100 | 58.00% | 63.50% | 70.17% | +12.17pp [9.33,15.17] |
| P s150 | 82.83% | 80.33% | 91.17% | +8.33pp [6.00,10.83] |

P9b 与 P8 主表 s100 69.67% 存在0.50pp差异，保持来源独立，不覆盖主表。P2 另一组固定 latent 对照 s150 +10.00pp、E l100 +9.83pp；两者不能叠加。S10 在前三个 P9b 工作点 C 低于原版（例如 s100 .5503 vs .5615），不是全部指标严格支配。

s100 Haar HH：D0=.0125、D1=.0037、D2=.0070。S10B 处于原版和过度平滑 WikiFT 之间；错目标 latent 的跟随率 s100 56.50→71.17%，而原真类误报1.33→1.50%。支持解码器能更好传递实际目标方向，不证明每个被削弱频带都是源画法。当前生成器也对 S10 存在适配优势。Texture Retraction 是由受控输出响应支持的机制解释，不是已定位通道的因果定理。

## 3. 历史训练决策链（不再重复）

| 方向 | 裁决 |
|---|---|
| 早期 Local Reader | hard AUROC .4645/.4770，未通过；latent 有信息不等于稳定可读画法 |
| 原 S10 Taylor 剪枝+感知恢复 | 胜出，保留 |
| SD0-C0/C1 响应蒸馏 | holdout LPIPS .08588/.08504 优于 old S10 .09340；下游 Top-1 36.83/36.00% 低于 old S10 37.67% |
| 官方 SD0 Encoder 兼容 | 不需校准、重训；S10B 与旧 S10 state_dict 及 PQC 一致 |

历史固定 latent Q2a 对照 +3.17pp，DINO-C/S 略降约.01；P9b 更新证据不删除旧口径。来源：[VAE 总览](../archive/historical/vae/00_OVERVIEW.md) · [训练全过程](../archive/historical/vae/01_S10_TRAINING_FULL.md) · [S10B 身份](../archive/historical/vae/10_S10B_FINAL.md)。

## 4. 使用契约与部署

repo scaled latent = posterior mean ×.18215；通过 src.utils.decode_s10 自动 / .18215；PQC 由 checkpoint wrapper 恰好一次应用。standalone decoder.py 直接接受**未缩放** latent，不要再除一次。S10B 无需 A0 文件作为 Decoder 运行依赖，但完整 RGB→RGB 仍需 Encoder。原版 decoder 仅在显式固定 latent 消融使用。

- 主权重：deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt。
- [独立加载器](../../deliver/checkpoints/s10b_standalone/decoder.py)；[src/utils](../../src/utils/__init__.py)。
- P2 4090 纯解码 D0 74.84→S10 21.82ms/img（3.43×）；P8 全链路 B16 S10段12.49ms/img，计时口径不同。
- 3060 旧 C0 同拓扑 TRT B8 27.08ms/img，PyTorch C0 68.80ms/img（约2.54×）；不能直接当最终 S10B 实测。重新导出最终权重，固定真实 latent parity，再接入完整 E/P 计时。

原始证据：[P2](../archive/historical/v35/data/p2_decoder_report.json) · [P9b](../archive/historical/v35/data/p9_paper_audit_report.json) · [3060历史 Profile/TRT](../archive/historical/vae/08_STAGED_PROFILE_AND_TRT.md) · [完整优化计划](../system/3060_PROFILE_OPTIMIZATION.md)。
