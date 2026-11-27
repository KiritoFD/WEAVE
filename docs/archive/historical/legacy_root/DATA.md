> **已归档（2026-11-14 仓库重组）**：论文快照期/阶段性文档，现状见 [docs/README.md](../README.md)。

# WEAVE 数据集说明（DATA.md）

> 本文档描述 WEAVE 项目使用的全部数据集、下载方式、预处理流程、目录结构、评估数据组成、去重结论，以及添加自定义数据集的步骤。
> 所有训练 / 评估图像均来自公开的 **WikiArt** 与 **COCO val2017**；VAE 为公开的 **SD1.5 EMA** 权重。仓库代码包不重新分发任何图像、packed latent 或 ONNX 图——需按本文档在本地准备。

---

## 1. 概述

WEAVE 在频域（Haar DWT）上求解 ODE，训练在 SD1.5 VAE 的 latent 空间进行。下表汇总项目涉及的全部数据集。

| 数据集名称 | 分辨率 | 用途 | 规模 | 主要位置（相对仓库根） |
|---|---|---|---|---|
| WikiArt distinct5 (W5 / D5-512 源) | 512 | D5 训练 + 测试源图 | train 18000 + test 150（5 风格 × 30） | `datasets/wikiart512_5style/` |
| WikiArt random20 (R20) | 512 | 随机 20 风格扩展训练 / 泛化评估 | 11050 唯一图 | `datasets/wikiart_random20_512/` |
| D5-512（distinct5_512 主集） | 512 | 主训练 / 评估数据集 | train 5000 + test 150 | `datasets/distinct5_512/{train,test}/` |
| D5-512 clean | 512 | D5 训练图统一重编码（便于快速 VAE encode） | train 5000（与 d5_train 文件名一一对应） | `datasets/distinct5_512/train_512clean/` |
| D5-512 latents (EMA) | latent | D5 packed latent 缓存 | train 5000 + test 150 | `datasets/distinct5_512_latents_ema/`、`data/train/.latent_cache/packed/` |
| D5-512 SDXL / VAVAE 变体 | latent | 跨 VAE 编码对照实验 | 5000 | `datasets/distinct5_512_sdxl/`、`datasets/distinct5_512_vavae/` |
| R5-512（Random 5 styles） | 512 | 与 D5 同 5 风格、不同源采样，仅 test | test 150 | `datasets/r5/`（远程）/ `data/images/r5/` |
| P2A-256（Photo-to-Art） | 256 | 内容（照片）→ 风格（艺术品）评估 | 配对集合 | `data/test_p2a/`、`data/train_p2a/`（可选） |
| fewshot6_512 / fewshot8_512 | 512 | few-shot 风格注册评估 | 160 / 多类 | `datasets/fewshot6_512/`、`datasets/fewshot8_512/` |
| other5 | 512 | 额外 5 风格评估（hold-out） | 152 | `datasets/eval/other5/` |
| eval（评估全集） | 512 | 主表评估汇总集 | 550 | `datasets/eval/` |
| legacy256_overfit50 | 256 | 旧 256 overfit 调试集 | 50 | `datasets/legacy256_overfit50/` |
| latent-256 | latent | 256 分辨率 latents（训练在用，非 512 重复） | — | `datasets/latent-256/` |
| clip-feats-vitb32 | 特征 | CLIP ViT-B/32 离线特征缓存 | — | `datasets/clip-feats-vitb32/` |
| WikiArt stress splits | 512 | 压力测试划分 | — | `datasets/wikiart_stress_splits_512/` |
| `data/train/` | latent | D5 训练 packed latents + pairing cache | 5000 | `data/train/`、`data/train/.latent_cache/` |
| `data/test/` | 512 | D5 测试图（5 风格 × 30） | 150 | `data/test/<style>/*.jpg` |
| `data/train_d5/` | latent | D5 干净版 packed latent cache | 5000 | `data/train_d5/.latent_cache/` |
| `data/onnx_exported/` | ONNX | ONNX VAE encoder/decoder 导出图 | 4 | `data/onnx_exported/{encoder,decoder}_b{8,16}.onnx` |
| `data/taesd/` | 权重 | TAESD 解码器（快速预览） | 1 | `data/taesd/` |

> D5 5 种风格类别（在代码、配置、manifest、输出目录中完全一致）：**Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e**（见 `data/README.md`、`tools/prepare_d5_512_clean.py:11`、`tools/prepare_samam_eval.py:8`）。

---

## 2. 数据集下载

### 2.1 WikiArt 原始图像

WikiArt 数据本身公开可获取，可通过以下任一渠道下载：

**渠道 A — ArtEmis 项目（推荐，已整理为按风格分类的目录结构）：**

- 项目主页：https://github.com/cs-chan/ArtEmis
- WikiArt 图像索引与情感标注：https://github.com/cs-chan/ArtEmis/releases
- 下载后按风格子目录组织到 `datasets/wikiart512_5style/<style>/*.jpg`。

```bash
# 示例：从 ArtEmis release 下载 wikiart 图像归档（请以仓库实际 release 为准）
mkdir -p datasets/wikiart512_5style
wget -c https://github.com/cs-chan/ArtEmis/releases/download/v1.0.0/wikiart.zip -O /tmp/wikiart.zip
unzip -q /tmp/wikiart.zip -d datasets/wikiart512_5style_raw
# 之后用 tools/prepare_d5_512_clean.py 选 5 风格并 resize 到 512
```

**渠道 B — WikiArt 官方站点抓取（自行合规使用）：**

- 站点：https://www.wikiart.org/
- 仅用于学术研究；请遵守 WikiArt ToS。

### 2.2 D5-512（distinct5_512）子集准备

从 WikiArt 中按 5 个目标风格筛选并 resize 到 512×512：

```bash
# 输入：包含全部风格的 WikiArt 根目录
# 输出：datasets/distinct5_512/train/<style>/*.jpg
python tools/prepare_d5_512_clean.py \
    --src  datasets/wikiart512_5style/train \
    --dst  datasets/distinct5_512/train_512clean \
    --workers 12 --quality 95
```

### 2.3 R5-512（Random 5 styles）

R5 与 D5 共用同一组 5 风格，但采样自 WikiArt 的不同子集（远程已就绪为 `datasets/r5/`）。如需本地重建：

```bash
python tools/make_r5_config.py   # 生成 R5 配置
python tools/vae_encode_r5.py    # 编码 R5 latents
```

### 2.4 P2A-256（Photo-to-Art）内容图来源

P2A 的内容图为 **COCO 2017 验证集（val2017）**，风格图为 D5 训练集中的艺术品。

- COCO val2017 下载：https://cocodataset.org/#download
  - 直接文件：http://images.cocodataset.org/zips/val2017.zip

```bash
mkdir -p data/test_p2a/content
wget -c http://images.cocodataset.org/zips/val2017.zip -O /tmp/val2017.zip
unzip -q /tmp/val2017.zip -d /tmp/coco_val2017
# 选择若干张作为 P2A 内容图（项目默认 30 张）
cp /tmp/coco_val2017/val2017/0000000000{00..29}.jpg data/test_p2a/content/ 2>/dev/null || true
# 风格图来自 D5 训练集
python tools/vae_encode_p2a256.py
```

### 2.5 SD1.5 EMA VAE（必需）

WEAVE 使用公开的 Stable Diffusion 1.5 EMA VAE 权重，latent scale = 0.18215：

- 模型：https://huggingface.co/runwayml/stable-diffusion-v1-5
- 由 `diffusers` 自动下载到 HF 缓存（`runs/cache/hf/`）。

---

## 3. 数据预处理

### 3.1 预处理脚本位置

仓库根下没有 `scripts/prepare_*.py` 或 `tools/prepare_data.py`，所有预处理脚本位于 `tools/`：

| 脚本 | 用途 |
|---|---|
| `tools/prepare_d5_512_clean.py` | D5 训练图统一 resize 到 512×512 并重编码为 JPEG（`--src` / `--dst`） |
| `tools/preencode_d5_packed.py` | D5 图像 → SD1.5 VAE encode → packed latent 缓存（`.latent_cache/packed/`） |
| `tools/fast_vae_encode_d5.py` | D5 VAE 编码快速路径 |
| `tools/vae_encode_r5.py` | R5 图像 VAE 编码 |
| `tools/vae_encode_p2a256.py` | P2A-256 内容图 + 风格图 VAE 编码 |
| `tools/taesd_encode_d5.py` | TAESD 编码 D5（仅用于快速预览） |
| `tools/encode_wikiart27_sd15.py` | WikiArt 27 风格 SD1.5 编码 |
| `tools/prepare_samam_eval.py` | 准备 SaMam 基线评估图（命名规范化） |

### 3.2 预处理流程

```
WikiArt 原图
   │
   ▼ tools/prepare_d5_512_clean.py  (resize 512×512, JPEG q=95)
D5-512 clean (datasets/distinct5_512/train_512clean/<style>/*.jpg)
   │
   ▼ tools/preencode_d5_packed.py   (SD1.5 VAE encode, latent_scale=0.18215)
packed latents (data/train/.latent_cache/packed/00_<style>.pt … 04_<style>.pt)
   │
   ▼ trainer.py 运行时按 packed cache 直接读取
训练 / 评估
```

要点：
- **Resize**：512×512（D5/R5）或 256×256（P2A / latent-256 / legacy256）。
- **VAE encode**：固定使用 SD1.5 EMA VAE，`latent_scale_factor = 0.18215`（见 `config.json` 的 `model.latent_scale_factor`）。
- **Latent cache 模式**：`data.latent_cache_mode = "packed"`（默认）。每个风格子目录的 1000 张 latents 拼接为一个 `XX_<style>.pt`，减少小文件 IO。
- **Pairing cache**：训练时的 (content, style) 配对预计算 top-k，保存为 `data/train/.latent_cache/prototype_pairing_top8.pt`，对应配置 `data.pairing_cache_topk = 8`。

### 3.3 Manifest 格式

每个数据集目录下放置 `manifest.json`，记录每张图的相对路径与标签：

```json
{
  "images": [
    {
      "image_path": "Minimalism/Minimalism__agnes-martin_aspiration-1960.jpg",
      "style_label": "Minimalism",
      "content_label": "agnes-martin_aspiration-1960",
      "split": "train"
    }
  ],
  "styles": ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
}
```

仓库内可见的 manifest 文件：
- `data/train/.latent_cache/packed/manifest.json` — D5 packed latent 索引
- `datasets/wikiart512_5style/manifest.json` — W5 全集索引（18000 train + 150 test）
- 各评估集目录下的 `test_manifest.json` / `train_manifest.json`

### 3.4 预处理命令示例（接收 `--wikiart-root` 或 `--src`）

```bash
# Step 1: 准备 D5-512 clean 训练图
python tools/prepare_d5_512_clean.py \
    --src  datasets/wikiart512_5style/train \
    --dst  datasets/distinct5_512/train_512clean \
    --workers 12 --quality 95

# Step 2: 预编码 packed latents（写入 data/train/.latent_cache/packed/）
python tools/preencode_d5_packed.py \
    --wikiart-root datasets/distinct5_512/train_512clean \
    --output-dir   data/train/.latent_cache/packed \
    --vae ema --latent-scale 0.18215

# Step 3（可选）: P2A-256 编码
python tools/vae_encode_p2a256.py \
    --wikiart-root datasets/distinct5_512/train_512clean \
    --content-dir  data/test_p2a/content \
    --output-dir   data/train_p2a

# Step 4: 路径自检（dry-run，不修改文件）
python tools/prepare_submission_data.py --sanitize-only --dry-run
```

> 注：`tools/prepare_submission_data.py` 由 `data/README.md` 推荐，用于在放置完文件后做一次路径净化与校验。

---

## 4. 数据集 Splits

### 4.1 D5-512

| Split | 规模 | 位置 | 说明 |
|---|---|---|---|
| `train` | 5000 | `datasets/distinct5_512/train/<style>/*.jpg` | 每风格 1000 张 |
| `test` | 150 | `datasets/distinct5_512/test/<style>/*.jpg`、`data/test/<style>/*.jpg` | 5 风格 × 30，hold-out，与 train 无 MD5 重叠 |
| `train_512clean` | 5000 | `datasets/distinct5_512/train_512clean/<style>/*.jpg` | `train` 的统一 resize + JPEG 重编码版，文件名一一对应但 MD5 全部不同 |
| `train_d5` (latent) | 5000 | `data/train_d5/.latent_cache/packed/` | train_512clean 的 packed latent 缓存 |

### 4.2 R5-512

- **仅 test**：150（5 风格 × 30），位于 `datasets/r5/test/`（远程）或 `data/images/r5/test/`（本地镜像）。
- **无独立 train**：R5 与 D5 共享同一份训练集（`datasets/distinct5_512/train`）。R5 仅用于在不同源采样上验证泛化性。
- 配置：`config_r5.json`、由 `tools/make_r5_config.py` 生成。

### 4.3 P2A-256

- **content + style pairs**：内容图来自 COCO val2017，风格图来自 D5 训练集。
- 位置：`data/test_p2a/{content,style}/`、`data/train_p2a/`（可选 latent）。
- 配置：`config_p2a256_adain2.json`。

### 4.4 Manifest 文件

| Manifest | 位置 | 用途 |
|---|---|---|
| `test_manifest.json` | `datasets/<set>/test_manifest.json` | 测试集索引 |
| `train_manifest.json` | `datasets/<set>/train_manifest.json` | 训练集索引 |
| `manifest.json` | `data/train/.latent_cache/packed/manifest.json` | packed latent 索引 |

### 4.5 风格类别一致性

D5 的 5 种风格在以下位置完全一致：
- **Paper / config**：`config.json` 的 `data.style_subdirs`
- **Manifest**：`manifest.json` 的 `styles` 字段
- **输出目录**：`data/test/<style>/`、`data/train/<style>/`
- **预处理脚本**：`tools/prepare_d5_512_clean.py:11`、`tools/prepare_samam_eval.py:8`

风格列表：**Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e**

> ⚠️ 已知偏差：部分外部记忆（project_memory）将 D5 风格误记为 `Cubism/Expressionism/Pop_Art/Romanticism/Symbolism`。代码与磁盘上的真实风格为上面列出的 5 种。本文档以代码与磁盘为准。

---

## 5. 数据集目录结构

### 5.1 `datasets/` 下的子目录（基于 `datasets/README.md`，去重后布局）

> `Dataset/`（仓库外绝对路径，被 `.gitignore` 忽略）通过软链接 / 目录联接以 `datasets/` 暴露给 WEAVE。下表列出全部子目录。

| # | 目录 | 大小 | 内容 / 规模 | 用途 |
|---|---|---|---|---|
| 1 | `wikiart512_5style/` | 5.47 GB | WikiArt 5 风格主源（train 18000 + test 150 + manifest） | D5/W5 主训练源 |
| 2 | `wikiart_random20_512/` | 3.38 GB | WikiArt 随机 20 风格 @ 512（11050 唯一图） | R20 扩展训练 / 泛化 |
| 3 | `wikiart_random20_512_latents_ema/` | 1.30 GB | R20 对应的 EMA latents | R20 训练 latent |
| 4 | `wikiart_stress_splits_512/` | 4.62 GB | 压力测试划分（多 split） | 极端规模 / 鲁棒性测试 |
| 5 | `distinct5_512/` | 3.25 GB | D5 主数据（train 5000 + test 150） | D5-512 主集 |
| 6 | `distinct5_512_latents_ema/` | 0.95 GB | D5 EMA latents（512） | D5 训练 latent 主副本 |
| 7 | `distinct5_512_sdxl/` | 0.29 GB | D5 的 SDXL VAE 编码变体 | 跨 VAE 对照实验 |
| 8 | `distinct5_512_vavae/` | 0.62 GB | D5 的 VA-VAE 编码变体 | 跨 VAE 对照实验 |
| 9 | `distinct5_512_train_zstar_src/` | 0.03 GB | D5 train 的 z* 源（用于 z-star 计算） | 训练目标对齐 |
| 10 | `fewshot6_512/` | 0.04 GB | fewshot6 评估集（160 图） | few-shot 风格注册评估 |
| 11 | `fewshot6_512_latents_ema/` | 0.31 GB | fewshot6 EMA latents | 同上 |
| 12 | `fewshot8_512/` | ~0 GB | fewshot8 评估集 | few-shot 扩展 |
| 13 | `fewshot8_512_latents_ema/` | 0.65 GB | fewshot8 EMA latents | 同上 |
| 14 | `style_data/` | 0.38 GB | 训练 / 评估用风格数据（含 overfit50 等） | overfit 调试、风格库 |
| 15 | `latent-256/` | 0.17 GB | 256 分辨率 latents（按风格子目录，如 `Hayao/`） | 256 训练（非 512 重复） |
| 16 | `latent_cyclegan/` | ~0 | CycleGAN latents | 实验性 |
| 17 | `clip-feats-vitb32/` | 1.27 GB | CLIP ViT-B/32 离线特征缓存 | 评估加速 |
| 18 | `legacy256_overfit50/` | 0.33 GB | 旧 256 overfit 调试集（50 图） | 历史 overfit 复现 |
| 19 | `legacy256_overfit50_vavae/` | 0.65 GB | 上者的 VA-VAE 编码 | 同上 |
| 20 | `eval/` | 0.11 GB | 评估全集缓存（550 图，含 d5_test 150 + legacy_test 150 + 其他） | 主表评估 |

> 兼容性 junction（仓库根目录，非副本）：
> - `Latent_Style/style_data` → `Dataset/style_data`
> - `Latent_Style/latent-256` → `Dataset/latent-256`

### 5.2 `data/` 下的处理数据

```
data/
├── train/                          # D5 训练 latents（按风格子目录）
│   ├── .latent_cache/
│   │   ├── packed/
│   │   │   ├── packed/
│   │   │   │   ├── 00_Early_Renaissance.pt   # 每风格 1000 latents 拼接
│   │   │   │   ├── 01_Impressionism.pt
│   │   │   │   ├── 02_Minimalism.pt
│   │   │   │   ├── 03_Rococo.pt
│   │   │   │   └── 04_Ukiyo_e.pt
│   │   │   └── manifest.json                  # packed latent 索引
│   │   └── prototype_pairing_top8.pt          # (content, style) 配对 top-8 缓存
│   ├── Early_Renaissance/*.pt                 # 单图 latents（备份形式）
│   ├── Impressionism/*.pt
│   ├── Minimalism/*.pt
│   ├── Rococo/*.pt
│   └── Ukiyo_e/*.pt
├── train_d5/                       # D5 干净版 packed latent cache
│   └── .latent_cache/              # 由 configs/exp_rebuttal_*.json 的 data.latent_cache_dir 指向
├── test/                           # D5 测试图（5 风格 × 30，hold-out）
│   ├── Early_Renaissance/*.jpg
│   ├── Impressionism/*.jpg
│   ├── Minimalism/*.jpg
│   ├── Rococo/*.jpg
│   └── Ukiyo_e/*.jpg
├── train_p2a/                      # 可选：P2A latents
├── test_p2a/                       # 可选：P2A 评估图（content + style 子目录）
├── images/                         # 可选：原图镜像
│   ├── d5/train/                   #   D5 原始 JPG（用于重新 VAE encode）
│   └── r5/{train,test}/            #   R5 board
├── onnx_exported/                  # ONNX VAE encoder/decoder
│   ├── encoder_b8.onnx
│   ├── encoder_b16.onnx
│   ├── decoder_b8.onnx
│   └── decoder_b16.onnx
└── taesd/                          # TAESD 解码器权重（快速预览）
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

配置默认值（见 `config.json`、`inference.json`）：
- `data.data_root = "data/train"`
- `training.test_image_dir = "data/test"`
- `data.latent_cache_mode = "packed"`
- `data.latent_cache_dir` 在 rebuttal 实验中指向 `data/train_d5/.latent_cache`

---

## 6. 评估数据

### 6.1 750 pairs 的构成

主表评估在 D5-512 上生成 **750 个 (content, style_ref) 配对**：

```
5 内容图 × 30 内容图/风格 × 5 风格参考 = 750
```

具体计算：
- 5 个目标风格（Early_Renaissance / Impressionism / Minimalism / Rococo / Ukiyo_e）
- 每风格 30 张内容图（来自 `data/test/<style>/`，hold-out）
- 每个内容图配 5 张风格参考图（从该目标风格的训练集中选取）

5 × 30 × 5 = **750 pairs**

对应配置：
- `full_eval.max_src_samples = 30`
- `full_eval.max_ref_compare = 16`
- `full_eval.max_ref_cache = 16`
- `full_eval.target_chunk_size = 1`

### 6.2 IDT baseline 的 750 pairs

`full_eval_clip_style_idt_baseline = 0.639920825263` 来自 **identity baseline（IDT）**：

- **来源**：从训练集随机采样的图像作为"风格参考"，但其风格 ≠ 目标风格。
- **目的**：衡量"用错误风格参考时的 CLIP 风格相似度"作为下界。
- **配对数**：与主评估一致，仍为 750 pairs，仅参考图替换为非目标风格图。
- 由 `tools/create_idt_other5.py`、`tools/build_other5_dataset.py` 协助生成。

### 6.3 风格参考图选择规则

1. **来源池**：目标风格在 `data/train/<style>/` 下的全部 1000 张图。
2. **去重**：与该内容图同源的图（同一艺术家 + 同一标题）被排除。
3. **采样**：默认按 CLIP 视觉相似度 top-k 缓存（`pairing_cache_topk = 8`），评估时从 top-8 中再采 5 张作为参考。
4. **缓存**：`data/train/.latent_cache/prototype_pairing_top8.pt` 保存所有 (content, top-8 style_ref) 索引。

---

## 7. 数据集去重说明

> 来源：`.task_release/state/code_analysis.md` §8（基于 `.task_cleanup/dedup_result.txt` 与 `dedup_md5.txt`，2026-07-25 生成）。

### 7.1 数据集规模（MD5 唯一性）

| 数据集 | 文件数 | 唯一 MD5 |
|---|---|---|
| `d5_test` | 150 | 150 |
| `d5_test_pd`（packed decode） | 150 | 150 |
| `w5`（W5 全集） | 18151 | 18062 |
| `r20`（WikiArt random20 @ 512） | 11101 | 11050 |
| `legacy_test` | 150 | 150 |
| `fewshot6` | 160 | 160 |
| `eval`（评估全集） | 550 | 550 |
| `other5` | 152 | 152 |

### 7.2 关键重叠结论

1. **`d5_test ⊂ eval`**：`d5_test ∩ eval = 150`（100% of d5_test，27.3% of eval）。
   - 即 `eval (550) = d5_test (150) + legacy_test (150) + 其他测试样本 (250)`。
2. **`d5_test ⊂ fewshot6`**：`d5_test ∩ fewshot6 = 150`（100% of d5_test，**93.8% of fewshot6**）。
   - fewshot6 的另外 10 张为额外 few-shot 样本。
3. **`legacy_test ⊂ eval`**：`eval ∩ legacy_test = 150`（100% of legacy_test，27.3% of eval）。
4. **三套测试集相互独立**：`d5_test` vs `w5_test` vs `legacy_test` 两两 MD5 重叠 = 0。可安全用于不同评估协议，**无训练-测试泄露**。
5. **`r20 ∩ w5 = 1258`**：11.4% of r20，7% of w5。R20 与 W5 存在部分源数据共享，若同时用于训练需注意去重。
6. **`d5_train` vs `d5_train_clean`**：5000 vs 5000 文件，**identical=0，different=5000**。clean 版本经过 resize + JPEG 重编码（`tools/prepare_d5_512_clean.py`），文件名一一对应但像素内容全部变化——这是**处理产物而非重复**。
7. **训练集无跨集泄露**：`d5_train ∩ w5_train = 0`，`d5_train ∩ legacy_train = 0`，`w5_train ∩ legacy_train = 0`。

### 7.3 处理产物 vs 源图

下表说明 `datasets/` 与 `data/` 中的 .pt / .onnx 文件**不是**源图的重复，而是处理产物：

| 处理产物目录 | 对应源图 | 关系 |
|---|---|---|
| `datasets/distinct5_512_latents_ema/` | `datasets/distinct5_512/train/` | VAE encode 后的 latent，文件名对应但格式 / 内容完全不同 |
| `data/train/.latent_cache/packed/` | `datasets/distinct5_512/train_512clean/` | 进一步 packed 拼接（每风格 1 个 .pt） |
| `data/train_d5/.latent_cache/` | 同上 | rebuttal 实验使用的副本 |
| `datasets/distinct5_512_sdxl/` / `_vavae/` | `datasets/distinct5_512/train/` | 跨 VAE 编码对照，非源图重复 |
| `datasets/clip-feats-vitb32/` | 各评估集 | CLIP 离线特征，非图像重复 |
| `data/onnx_exported/` | — | ONNX 计算图，与图像无重叠 |

### 7.4 文档建议（已落地到本节）

- 测试集隔离：`d5_test` / `w5_test` / `legacy_test` MD5 完全独立，可放心用于不同评估协议。
- `d5_test` 是 `eval` 的子集：使用 eval 时需意识到其中 150 张已参与 D5 评估。
- `d5_train_clean` 是 `d5_train` 的处理版：文件名一一对应但 MD5 全部不同，应视为同一逻辑数据集的两种物理表示。
- 训练集无跨集泄露：d5_train / w5_train / legacy_train 可同时用于不同实验。
- R20 与 W5 有 ~11% 重叠：若同时用于训练需先去重。

---

## 8. 添加自定义数据集

### 8.1 总体流程

1. **准备源图**：按风格子目录组织为 `<my_set>/<style>/*.jpg`，分辨率统一 resize 到 512×512（或 256）。
2. **生成 manifest**：在数据集根目录写 `manifest.json`，格式见 §3.3。
3. **VAE encode**：调用预处理脚本生成 packed latent 缓存。
4. **配置 `config.json`**：修改 `data.data_root` 与 `data.style_subdirs` 指向新数据集。
5. **路径自检**：运行 `python tools/prepare_submission_data.py --sanitize-only --dry-run`。

### 8.2 示例：添加一个 3 风格的新数据集

```bash
# 假设新数据集名为 myart3，3 个风格：Baroque, Pointillism, Realism
# Step 1: 准备源图
mkdir -p datasets/myart3/{Baroque,Pointillism,Realism}
# 将原图按风格放入对应子目录（每风格建议 ≥ 200 张）

# Step 2: 生成 manifest（Python 伪代码）
python - <<'PY'
import json, pathlib
root = pathlib.Path("datasets/myart3")
styles = ["Baroque", "Pointillism", "Realism"]
images = []
for s in styles:
    for p in sorted((root / s).glob("*.jpg")):
        images.append({
            "image_path": f"{s}/{p.name}",
            "style_label": s,
            "content_label": p.stem,
            "split": "train",
        })
(root / "manifest.json").write_text(json.dumps({"images": images, "styles": styles}, indent=2))
print(f"wrote {len(images)} entries")
PY

# Step 3: VAE encode（复用 preencode_d5_packed.py，按需调整 --wikiart-root）
python tools/preencode_d5_packed.py \
    --wikiart-root datasets/myart3 \
    --output-dir   data/train_myart3/.latent_cache/packed \
    --vae ema --latent-scale 0.18215

# Step 4: 修改 config.json
#   data.data_root = "data/train_myart3"
#   data.style_subdirs = ["Baroque", "Pointillism", "Realism"]
#   model.num_styles = 3

# Step 5: 路径自检
python tools/prepare_submission_data.py --sanitize-only --dry-run
```

### 8.3 Manifest 格式要求

- 顶层为 JSON 对象，必含 `images`（数组）与 `styles`（数组）。
- 每条 `images` 项必含字段：
  - `image_path`：相对数据集根的路径（含风格子目录前缀）
  - `style_label`：风格名（必须在顶层 `styles` 列表中）
  - `content_label`：内容标识（通常为去后缀文件名）
  - `split`：`"train"` / `"test"` / `"val"` 之一
- 风格名使用下划线连接（如 `Early_Renaissance`），避免空格与中文。
- 编码：UTF-8，无 BOM。
- 大数据集可分片写为 `manifest.part1.json`、`manifest.part2.json`，但 `manifest.json` 必须存在并作为入口。

### 8.4 添加新风格到既有数据集

如需在 D5 基础上注册第 6 个风格（fewshot 场景），使用 `register_new_style.py`：

```bash
python register_new_style.py \
    --base_ckpt       code/checkpoints/weave_epoch_0004.pt \
    --style_name      Cubism \
    --train_img_dir   datasets/r5/train_reg64/Cubism \
    --output_ckpt     datasets/r5/registered/Cubism.pt \
    --num_steps       500 \
    --batch_size      4 \
    --lr              1e-3
```

注册完成后，将新风格的 latent 加入 `data/train/<new_style>/` 并在 `data.style_subdirs` 中追加。注意 `model.num_styles` 也需同步更新。

---

## 附录：关键路径速查

| 用途 | 路径 |
|---|---|
| 训练 packed latent | `data/train/.latent_cache/packed/packed/0[0-4]_<style>.pt` |
| 训练配对缓存 | `data/train/.latent_cache/prototype_pairing_top8.pt` |
| 测试图 | `data/test/<style>/*.jpg` |
| D5 主源 | `datasets/distinct5_512/{train,test}/` |
| W5 主源 | `datasets/wikiart512_5style/` |
| R20 主源 | `datasets/wikiart_random20_512/` |
| 评估全集 | `datasets/eval/`（550 图） |
| ONNX VAE | `data/onnx_exported/` |
| TAESD | `data/taesd/` |
| 配置 | `config.json`、`config_r5.json`、`config_p2a256_adain2.json`、`configs/*.json` |
| 推理协议 | `inference.json`、`configs/inference_*.json` |

---

*文档生成时间：2026-07-25 · 基于 `data/README.md`、`datasets/README.md`、`.task_release/state/code_analysis.md`、`.task_release/state/audit_report.md` 与 `tools/prepare_*.py` 实际代码。*
