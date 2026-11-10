# VAE / Decoder Checkpoints（从 4090 拉回）

拉回时间：2026-11-10
全部通过 SHA256 与远端 `/root/Workspace/WEAVE_VAE/` 逐字节校验。

## 关键：Decoder 必须与**它训练时用的 Encoder** 配对

| 文件 | 大小 | Decoder | **配套 Encoder** | 说明 |
|---|---:|---|---|---|
| `runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt` | 40.7 MB | 10.658M | **A0 原版 Encoder** | 在 A0 冻结 latent 上训练的独立 decoder；`meta` 内含 `post_quant_conv` |
| `runs/vae_style_vae/C_edit_best.pt` | 171.1 MB | 10.658M | **自身联合训练的 Encoder（在此文件内）** | 完整 StyleVAE：`encoder` + `decoder` + `style_head` |
| `runs/vae_style_vae/C_edit_decoder_only.pt` | 40.7 MB | 10.658M | 只能配 `C_edit_best.pt` 里的 Encoder | 从 C_edit 抽出的独立 decoder，**不可与 A0 Encoder 混用** |
| `runs/vae_style_vae/B_style_decoder_only.pt` | 40.7 MB | 10.658M | 同上（B_style 的 Encoder） | B 阶段中间产物，仅供对照 |

**`C_edit_decoder_only.pt` 与 `s10_p5_lowlr/decoder.pt` 参数量相同但权重不同**：前者与一个被改过的 Encoder 共同适配，后者只适配 A0。用 A0 Encoder 配 C_edit decoder 会得到错误的 latent 分布。

## 实测指标

| 模型 | 评测集 | L1 | LPIPS@512 | CSD Keep |
|---|---|---:|---:|---:|
| A0 原版（49.49M decoder） | 独立测试 256 张 | 0.06235 | 0.06789 | 0.94972 |
| `s10_p5_lowlr`（A0 Encoder） | 独立测试 256 张 | 0.06727 | 0.09264 | 0.85346 |
| `C_edit_best`（自带 Encoder） | holdout 128 张 | 0.06480 | 0.07423 | 0.88159 |
| `B_style_best` | holdout 128 张 | 0.06435 | 0.07447 | 0.87196 |

注：两组数字的评测集不同（测试集 vs holdout），不可直接相减比较。

## 加载方式

```python
# 独立 decoder（s10_* 与 *_decoder_only.pt 都是这个格式）
ck = torch.load(path, map_location="cpu", weights_only=False)
from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
dec = build_decoder(ck["meta"]["widths_forward"])   # [64,128,192,256]
dec.load_state_dict(ck["state_dict"])
model = StudentDecoder(pqc, dec)      # pqc 来自配套 Encoder 所在的 AutoencoderKL
# 注意：StudentDecoder 内部已应用 post_quant_conv，不要再手动套一次

# 完整 StyleVAE
ck = torch.load("C_edit_best.pt", map_location="cpu", weights_only=False)
# ck["encoder"] / ck["decoder"] / ck["style_head"] / ck["meta"]
```

## 结构

`widths_forward = [64, 128, 192, 256]`（forward 顺序，decoder 内部反向使用）
→ `mid_block/up_blocks[0]=256, ub1=192, ub2=128, ub3=64`
512px 分辨率下的通道和 384 → 192（A0 的一半）。
