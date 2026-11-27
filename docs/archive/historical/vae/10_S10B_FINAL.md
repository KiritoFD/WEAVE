# S10B 定稿：零重训、零 A0 运行时依赖

**日期**：2026-11-12 ｜ **核心答案**：**S10B 就是我们一直用的那个旧 S10，字节级完全相同**。B 实验没有改模型任何一个参数，只把喂它的 Encoder 从 A0 换成了官方 SD0。

---

## 1. 用户要求的 4 步，全部通过

### ① B 实验实际 PQC vs checkpoint 内 PQC 的参数哈希

| 张量 | sha256（原始 fp16 字节） |
|---|---|
| `post_quant_conv.weight` | `684d5fc23767b72cc54c3d00e5aec65f5b666f06d3b69161c9e7625cc1cc221e` |
| `post_quant_conv.bias` | `e82ce4c5192022fa80c40265d8635aa501747e2fe106c6a655598d701a7a5987` |
| S10 主体 `state_dict`（140 张量，按键排序） | `f8edda08f3b51a9f19ab18ac96b30a00359d5f50077b162294feacce1d03ad5f` |
| 原始 checkpoint 文件 | `ac0bf25007d28b2234aa6afcfeab3dd35242f5aef69f9ed49194a9a2ba83a58d` |

P0 接口审计里记录的 B 实际 PQC 与上述 checkpoint 值 **max\|diff\| = 4.99e-6 / 4.04e-6**（仅为 audit 存了 5 位小数展示值的舍入误差）→ **B 用的就是 checkpoint 原装 PQC，没有重校准**。

### ② 把 B 实际使用的 PQC 和 S10 主体一起导出

`deliver/checkpoints/s10b_standalone/`：
- `decoder.pt`（40.7 MB，`state_dict` + `post_quant_conv` + `meta`）
- `config.json`（完整契约）
- `decoder.py`（**零仓库依赖**的独立加载器，只用 `diffusers.models.autoencoders.vae.Decoder`）

### ③ 独立加载后与 B 实验输出逐元素一致

| 验证 | max\|diff\| |
|---|---:|
| 随机未缩放 latent（2 张，cuda） | **0.0** |
| **真实缓存 SD0 latent**（4 张 holdout） | **0.0** |

独立 `decoder.py` 加载后的 `standalone(z)` 与 B 实验的 `StudentDecoder(A0-PQC, 旧S10)(z)` **完全逐元素相同**。

### ④ 固定输入缩放、模型版本、checkpoint 哈希

`config.json` 固化：
- `latent_convention`：**未缩放** `z = E_SD0(x).latent_dist.mean`，**不乘 scaling_factor**
- `scaling_factor_reference = 0.18215`（仅外部生成流水线用）
- `post_quant_conv`：**包含在包内，由 wrapper 恰好应用一次**
- `encoder` = 官方 `sd-vae-ft-ema` + 双 sha256
- `origin_checkpoint_sha256` / `s10_body_state_dict_sha256` / PQC 双 sha256
- `widths_forward=[64,128,192,256]`、`params=10,657,923`
- `torch_version` / diffusers 版本、IO 数值范围

**→ 运行时彻底不再需要 A0 模型文件**。`runtime_a0_dependency: "NONE"`。

---

## 2. 「S10B 和我们现在一直用的 S10 是不是一个东西」——字节级实证

本地对比 `deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt` 与 `deliver/checkpoints/s10b_standalone/decoder.pt`：

```
state_dict 张量数         : 140 vs 140
state_dict sha256        : f8edda08...d5f  ==  f8edda08...d5f
全部张量 torch.equal     : True
pqc weight torch.equal   : True   (sha 684d5fc23767b72c)
pqc bias   torch.equal   : True   (sha e82ce4c5192022fa)
```

**是的，完全同一个东西**。S10B 不是新模型、不是校准版、不是适配版；它只是给旧 S10 换了个名字，用来明确「在 SD0 Encoder 下直接可用」。

回顾整个适配实验：
- B 从 `s10_p5_lowlr/decoder.pt` 读 `state_dict` 和 `post_quant_conv`，原样构造 `StudentDecoder`，喂未缩放 SD0 latent
- C（闭式校准）改了 PQC，但 P2 证明**反而更差**，未采纳
- S10 主体 10.658M 参数**自始至终一个没动**

所以 `s10b_standalone/` 与 `s10_p5_lowlr/decoder.pt` 的权重完全相同；差别只在**包装**：前者把 PQC 与主体、契约、哈希、独立加载器打包成一个自包含交付物。

---

## 3. 为什么还需要这个包

旧 `s10_p5_lowlr/decoder.pt` 本身已含 PQC（`needs_post_quant_conv: True`），理论上也能直接用。独立包的价值是：
1. **显式消除 A0 运行时依赖**：加载器不需要 `AutoencoderKL.from_pretrained(A0_DIR)`，也不会误用 A0 的 `vae.decode()`（会二次应用 PQC）
2. **契约写死**：未缩放 latent、单次 PQC、官方 SD0 Encoder 哈希，避免重演历史「DINO-C≈0.22」的接口错误
3. **可复现校验**：保留全部 sha256，任何部署都能核对

---

## 4. 交付物

| 文件 | 说明 |
|---|---|
| `deliver/checkpoints/s10b_standalone/decoder.pt` | 自包含权重（PQC + S10 主体 + meta） |
| `deliver/checkpoints/s10b_standalone/config.json` | 固定契约（缩放/版本/哈希/IO） |
| `deliver/checkpoints/s10b_standalone/decoder.py` | 零依赖独立加载器 |
| `deliver/s10b_final_manifest.json` | 4 步验收结果汇总 |
| `tools/vae_suite/export_s10b.py` | 导出 + 哈希 + 逐元素验证脚本 |
| 4090: `runs/sd0_s10/s10_adapt/export_s10b.log` | 完整验证日志 |

**使用方式**：
```python
import importlib.util, torch
spec = importlib.util.spec_from_file_location(
    "s10b", "deliver/checkpoints/s10b_standalone/decoder.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dec = m.load("deliver/checkpoints/s10b_standalone").cuda().eval()
z = official_sd0_vae.encode(x).latent_dist.mean   # 未缩放！
y = dec(z)                                        # 内部恰好一次 pqc
```

**结论**：不需要重新训练；旧 S10 在官方 SD0 Encoder 下零成本可用；4 步全部完成，A0 运行时依赖已彻底去除。
