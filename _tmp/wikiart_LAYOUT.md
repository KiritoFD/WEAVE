# wikiart_new_packed — 落地布局（编码进行中，2026-11-05）

> 面向实验会话：只读约定，loader/采样器接入由实验侧完成。
> 源图像：`datasets/wikiart_new_512/<Style>/*.jpg`（10,545 张，tier 1+2，512×512 q95）
> 源清单：`datasets/wikiart_new_512/manifest.jsonl`（每张图一行）

---

## 1. 目录结构与分片

```
data/wikiart_new_packed/
├── packed/
│   ├── 00_Early_Renaissance.pt     N = 474
│   ├── 01_Impressionism.pt         N = 7,989   ← 单文件最大，约 523.6 MB
│   ├── 02_Minimalism.pt            N = 84
│   ├── 03_Rococo.pt                N = 1,519
│   └── 04_Ukiyo_e.pt               N = 479
├── manifest.json       每张一行的 sidecar（10,545 条）
├── meta.json           编码耗时 / 显存 / 每类字节
├── qc.json             6 张 encode→decode 回像素核对
├── qc/*.png            左右拼图（左=源 512，右=decode）
└── LAYOUT.md           本文件
```

- **每类一个 `.pt`，不分 shard。** 分片边界 = 风格边界，文件名沿用 D5 的 `{sid:02d}_{Style}.pt`（sid 顺序与 D5 `STYLES` 完全一致）。
- 单文件最大为 `01_Impressionism.pt`：7,989 × 4 × 64 × 64 × 4 B ≈ 523.6 MB（float32）。其余均 < 100 MB。
- 全量约 691 MB（10,545 × 65,536 B + 序列化开销）。
- `.pt` 内**不**写相对路径重定向，`files[]` 是**源 JPEG 文件名**（相对 `datasets/wikiart_new_512/`），不是 latent 单文件路径。
- 为空类别不会生成文件（当前 5 类均非空）。

---

## 2. Tensor 约定

| 项目 | 值 |
|---|---|
| 每个 `.pt` 的 key | `schema, subdir, count, files, latents`（与 D5 逐键一致） |
| `latents` shape | `(N, 4, 64, 64)` |
| dtype | `torch.float32`（CPU 存储；编码时 VAE 为 bf16，落盘统一转 fp32） |
| 归一化 / scale | **已乘 `vae.config.scaling_factor = 0.18215`**。即 `z_stored = latent_dist.sample() * 0.18215` |
| 与 D5 是否一致 | **完全一致**：同一 `utils.inference.encode_image`，同一 SD1.5 EMA VAE，同一 scaling 约定，无额外 mean/std 归一化 |
| 随机性 | 编码用 `latent_dist.sample()`（后验采样），**不保证逐位可复现**；与 D5 packed 同口径 |
| 解码 | `decode_latent()` 内部先 `z / scaling_factor` 再 `vae.decode`，故**直接喂训练 dataloader 的 latents 即可**，无需二次缩放 |

> 注意：不要对这批 latent 再跑 `style_classifier.build_latent_dataset` 里的 "auto-scale by 1/0.18215" 分支——那一步是为**未缩放**的 per-image latent 准备的；D5 packed 与本次 packed 都已是缩放后的值。

---

## 3. 元数据如何携带

`.pt` 内**只**带 `files[]`（顺序 = tensor 行顺序），其余字段全部在 **sidecar JSON**（不是 jsonl）：

`manifest.json`：

```json
{
 "schema": 1,
 "image_root": "<datasets/wikiart_new_512 绝对路径>",
 "packed_root": "<data/wikiart_new_packed/packed 绝对路径>",
 "styles": ["Early_Renaissance", "...", "Ukiyo_e"],
 "total": 10545,
 "source_manifest": "<datasets/wikiart_new_512/manifest.jsonl>",
 "samples": [
   {"index": 0, "packed": "packed/00_Early_Renaissance.pt",
    "path": "Early_Renaissance/Early_Renaissance__<artist-slug>_<title-slug>.jpg",
    "style": "Early_Renaissance", "artist": "...", "split": "train|val",
    "provenance": "official_train|official_test", "tier": 1,
    "sha256": "<512 JPEG 的 sha256>",
    "latent_shape": [4, 64, 64]}
 ]
}
```

**对齐关系**：`samples[i]` 与 `latents[i]` 一一对应，`i` 是该 `.pt` 内的行号（`index` 字段就是它，按 `style` 分组后独立从 0 起算）。
反向查找：`samples` 按 `packed` 分组后，组内顺序即 tensor 行序。

`split` 分布：train 8,752 / val 1,793（画家级分组，画家不跨侧，见 `datasets/wikiart_new_512/split_artists.json`）。

---

## 4. 与 D5 `data/train/.latent_cache/packed` 的字段差异

| 项目 | D5 | 本目录 |
|---|---|---|
| 目录层级 | `packed/manifest.json` + `packed/packed/*.pt`（**双层 packed**） | `packed/*.pt`（单层）+ 根级 `manifest.json` |
| `.pt` keys | `schema/subdir/count/files/latents` | **完全相同**（5 键） ✅ |
| `files[]` 含义 | 相对 `data/train/<Style>/` 的**单图 `.pt` 路径**（`Style/xxx.pt`） | 相对 `datasets/wikiart_new_512/` 的**JPEG 路径**（`Style/xxx.jpg`） |
| `manifest.json` 顶层 | `schema, data_root, style_subdirs, styles` | `schema, image_root, packed_root, styles, total, source_manifest, samples` |
| manifest 每类结构 | `{count, files}`（**无 split / provenance / sha256 / artist / tier**） | 逐张：`index, packed, path, style, artist, split, provenance, tier, sha256, latent_shape` |
| 附加文件 | `meta.json` 在 `train_d5/.latent_cache/`（vae/source/wall_sec/ms_per_img/peak_vram_mib） | `meta.json`（更全：per_style 字节/耗时、packed_bytes_total、peak alloc/reserved）+ `qc.json` + `qc/` |
| 文件大小 | `data/train/.latent_cache/packed/packed/*.pt` = 65.6 MB/1000 张（1×）；`data/train_d5/.latent_cache/packed/*.pt` = 327.7 MB/1000 张（**5× storage 放大**） | 1×，65,536 B/张（与 `data/train` 那份一致） |

**结论**：`.pt` 载荷层**零差异**（可直接用同一个 `torch.load` 读取函数 + 同一个 `payload["latents"]` 入口）；
差异只在 sidecar 层——本目录是 "D5 的 `{count, files}`" 的超集，多 7 个字段（`split/provenance/artist/tier/sha256/index/packed`），少的只有 D5 的 `data_root`/`style_subdirs`（等价信息以 `image_root`/`styles` 表达）。

---

## 5. 合并 latent 根的路径约定（给 loader）

推荐 **多根读取**，不要物理合并：

```python
LATENT_ROOTS = [
  "data/train/.latent_cache/packed/packed",   # D5, 5 × 1000
  "data/wikiart_new_packed/packed",           # new, 5 × [474,7989,84,1519,479]
]
# 每个 style 按 {sid:02d}_{Style}.pt 分别 load，再按需 torch.cat([a["latents"], b["latents"]], 0)
```

理由：两处 `.pt` 文件名同构、payload 5 键同构，但 `files[]` 语义不同（`.pt` vs `.jpg`）——合并时**必须同时保留两份 `files` 列表**，或改用各自 sidecar 的 `path` 字段，否则无法回溯。

若实验侧确实要单根（例如 `config.data.data_root` 只接受一个目录），两种做法：
1. **符号链接**（零拷贝，推荐）：`data/merged_packed/packed/` 下按 style 建 2 个链接会撞名，改用 `packed/00_Early_Renaissance__d5.pt` / `...__new.pt` 组合命名；
2. **物理合并** `torch.cat` 后另存 → 约 691 MB 额外占用，且需重写 manifest。

**不要**把新数据写进 `data/train/.latent_cache/`——现有 D5 `manifest.json` 的 `styles[*].count==1000` 与 `files` 长度是训练侧强校验点，写进去会破坏一致性。

---

## 6. 编码实测（2026-11-05 17:11:16 → 17:34:51）

| style | count | 字节 | s/张 | 该类耗时 |
|---|---|---|---|---|
| Early_Renaissance | 474 | 31,110,980 | 0.137 | 65.1 s |
| Impressionism | 7,989 | 524,224,040 | 0.133 | 1,064.0 s |
| Minimalism | 84 | 5,512,275 | 0.138 | 11.6 s |
| Rococo | 1,519 | 99,668,407 | 0.133 | 202.3 s |
| Ukiyo_e | 479 | 31,430,462 | 0.134 | 64.0 s |
| **合计** | **10,545** | **691,946,164** | **0.1334** | **1,407.1 s（23.45 min）** |

- 吞吐 **7.49 img/s**（batch=8，SD1.5 EMA VAE bf16，RTX 3060 12 GB）。
- 显存：`nvidia-smi` 实测 **3,811 MiB / 12,288 MiB**（编码 Impressionism 期间快照）；batch=8 未触发 OOM。
- 平均 **65,617 B/张**（1×，无 storage 放大）。
- 编码启动前显存闸门：free ≥ 5,120 MiB 且无 `trainer.py` 进程，17:11:16 满足后立即启动；期间未抢占 A1。

### QC：encode → decode 回像素（`qc.json` + `qc/*_srcdst.png`）

| style | 样本 | PSNR (dB) | MAD (0–255) |
|---|---|---|---|
| Early_Renaissance | andrea-del-castagno_crucifixion-and-saints | 26.38 | 8.89 |
| Impressionism | a-back-yard-reminiscence-of-brooklyn_wash-day | 24.33 | 10.44 |
| Minimalism | 14_r-71 | 30.30 | 4.07 |
| Rococo | allan-ramsay_george-iii | 25.12 | 8.02 |
| Ukiyo_e | a-court-lady-of-the-kyowa-era_looking-refined | 29.39 | 4.96 |
| Impressionism | horia-bernea_springtime | 23.21 | 12.23 |

PSNR 23.2–30.3 dB 属 SD1.5 VAE 正常有损往返区间（512→64×64 latent→512），非链路故障。拼图在 `qc/`（左源、右解码）。

### 风格可辨识性读数（val 1,793，画家级防泄漏）

`style_readout.json`；探针 = subband 统计（64 维）+ 标准化 + 逻辑回归，训练集 8,752 张。

| 探针 | acc | macro_recall | ER | Imp | Min | Roc | Uki |
|---|---|---|---|---|---|---|---|
| subband / 普通 CE（自然先验，Imp 占 80%） | 0.739 | 0.373 | 0.054 | 0.957 | 0.300 | 0.320 | 0.234 |
| **subband / 类别加权 CE** | **0.591** | **0.590** | 0.500 | 0.576 | 0.600 | 0.657 | 0.617 |
| raw_global / 加权 CE | 0.490 | 0.577 | 0.446 | 0.428 | 0.700 | 0.651 | 0.660 |
| D5 参考（同 subband 探针，800/200 均衡） | 0.639 | 0.639 | 0.485 | 0.515 | 0.850 | 0.590 | 0.755 |

- 未加权读数被 Impressionism 主导（8,752 训练里 ~80%），ER 几乎全被吞（5.4%）——这是**类别先验**问题，不是 latent 损坏。
- 加权后 macro **0.590**（5 类随机 = 0.200），与 D5 同探针的 0.639 同量级 → 新数据 latent 的风格可分性与 D5 相当，**可比 D5 略低但不退化**。
- 差异主要在 Rococo/Ukiyo_e（新数据 +0.06/+0.07 vs D5 在 Min/ER/Uki 上更高）；Min 仅 84 张（val 10 张），其 recall 波动大，不宜单独解读。

