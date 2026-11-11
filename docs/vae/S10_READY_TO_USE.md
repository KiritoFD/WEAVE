# S10 能直接拿来用（原版 SD0 Encoder）

**结论**：旧 S10（`s10_p5_lowlr/decoder.pt`，A0 配套训练）**不需要任何重训、微调、校准或适配器**，直接喂未缩放的原版 SD0 latent 即可使用，效果不降反升。

## 怎么直接用

```python
ck = torch.load("s10_p5_lowlr/decoder.pt", map_location="cpu", weights_only=False)
dec = build_decoder([64, 128, 192, 256]); dec.load_state_dict(ck["state_dict"])
pqc = nn.Conv2d(4, 4, 1); pqc.load_state_dict(ck["post_quant_conv"])
student = StudentDecoder(pqc, dec).cuda().eval()   # 内部恰好一次 pqc

z = sd0_vae.encode(x).latent_dist.mean             # 未缩放！不要乘 scaling_factor
y = student(z)
```

三条硬约束：
1. **未缩放** posterior-mean latent（`scaling_factor=0.18215` 只用于外部生成流水线）
2. `post_quant_conv` 用 **checkpoint 自带的那个**（A0 训练的），**恰好应用一次**
3. 不调用 `vae.decode()`（会二次应用 PQC）

## 证据

| 档 | Encoder | PQC | P0 重建 DINO-C | P2 Oracle1 Affinity |
|---|---|---|---:|---:|
| A（历史榜单第 4 名） | A0 | A0 原版 | 0.9695 | 0.5587 |
| **B（直接用）** | **SD0** | **A0 原版（原封不动）** | **0.9678** | **0.5616** |
| C（闭式校准 PQC） | SD0 | 最小二乘 | — | 0.5608（更差） |
| D（官方 49.49M） | SD0 | SD0 原版 | — | 0.5548 |

- 接口审计：未缩放 latent、恰好一次 PQC；`cos(A0,SD0)=0.992142`、RMS 比 1.001719
- **历史「SD0→旧S10 DINO-C≈0.22」是接口/缩放错误，正确接口下是 0.968**
- C 的闭式校准在 5 个主要指标上显著劣于 B（配对 Bootstrap CI 全排除 0）→ 不做
- B 显著优于官方 D0（Affinity +0.0068、LPIPS −0.0114）
- B4 分支同结论；A 的 0.5587 与 `docs/baseline` 榜单逐位吻合，证明同协议可比

## 字节级证明：S10B 就是旧 S10

`deliver/checkpoints/s10b_standalone/decoder.pt` 与 `s10_p5_lowlr/decoder.pt`：
```
state_dict sha256 : f8edda08…d5f  ==  f8edda08…d5f
140 张量 torch.equal : True ｜ pqc weight/bias : True / True
```
B 没改模型任何一个参数，只换了喂它的 Encoder。独立包只是把 PQC、契约、哈希、零依赖加载器打成自包含交付物，**彻底去掉 A0 运行时依赖**。

## 一句话

**不要再为兼容 SD0 重训 10.66M Decoder。** 旧 S10 拿原版 SD0 latent 直接跑，零成本、零额外推理开销，且是当前最好的 Ours 配置（Affinity 0.5616，高于榜单第 4 名的 0.5587 与本轮新 C0/C1 的 0.5575/0.5576）。

详见：[S10_SD0_ADAPT_REPORT.md](S10_SD0_ADAPT_REPORT.md) · [S10B_FINAL.md](S10B_FINAL.md) · [NOV11_12_WORK_SUMMARY.md](NOV11_12_WORK_SUMMARY.md)
