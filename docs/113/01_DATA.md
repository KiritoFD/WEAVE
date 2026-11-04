# 01 数据全景

> 全部路径以远端 `/mnt/i/Latent_Style/WEAVE`（Windows `I:\Latent_Style\WEAVE`）为准；本文件数字均为 2026-11-03 在 3060 上实测（`find`/`PIL`/`torch.load`）。本地仓库**没有** `data/`、`datasets/`。

## 0 速查

| 集合 | 原图 | latent | 规模 | 用途 | 状态 |
|---|---|---|---|---|---|
| 5k D5 | `datasets/d5/train`（亦 `distinct5_512/train`） | `data/train/*.pt` + `data/train_d5/.latent_cache/packed` | 5000 = 5 风格 × 1000 | 唯一已训练过的训练集 | 完整 |
| 18k | `datasets/wikiart512_5style/train` | **无** | 18000 = 5 × 3600 | 数据规模对照（路线 C/P0） | 仅原图 |
| 评测 | `data/test` = `datasets/d5/test` | 不需要 | 150 = 5 × 30 内容图 | 600 跨 + 150 同风格 | 完整 |
| 18k 评测 | `datasets/wikiart512_5style/test` | 不需要 | 150 = 5 × 30 | 18k 协议评测 | 完整 |
| R5 备选 | `datasets/r5/train` + `test` | 无现成（同名风格不同） | 5000 + 150 | 跨数据集泛化 | 完整 |

关键结论：**5k 与 18k 不是同一风格集合**，只有 `Impressionism` 重合（`docs/HANDOVER.md#L158-L163`）。D5 五风格为 Early_Renaissance / Impressionism / Minimalism / Rococo / Ukiyo_e；18k 为 Expressionism / Impressionism / Post_Impressionism / Realism / Symbolism。因此"5k vs 18k 同风格对照"不成立，只能做同协议不同规模的近似对照。

## 1 5k D5（论文模型唯一训练集）

- 原图：`datasets/d5/train`，5 风格 × 1000，JPEG，长边/高 512（实测样本 423×512，比例不一）；`distinct5_512/train_512clean` 是 512×512 居中裁剪版（实测 512×512）。
- 逐样本 latent：`data/train/<Style>/<Style>__<artist>_<title>.pt`，每个文件是 `Tensor(1,4,64,64)`（SD1.5 VAE，512px → latent 64）。
- 训练用 packed 缓存：`data/train_d5/.latent_cache/packed/`，5 个 `.pt`，每个约 313MB，共约 1.6G；`torch.load` 实测 `count=1000`、`latents=(1000,4,64,64) float32`（打包逻辑见 [utils/dataset.py](file:///g:/GitHub/Latent_Style/WEAVE/utils/dataset.py#L371-L412)）。
- packed 文件字段：`schema=1, subdir, count, files[], latents`；同名 `manifest.json` 记录 `style_subdirs` 与每风格 `count/files`。
- 像素备份：`data/train_d5/images_hwc_uint8.bin` 3,932,160,000 B + `images_index.json`（仅为重编码用，训练不读）。
- 其他 latent 版本：`data/train_taesd`（约 314M）、`data/train_sdxl`；`datasets/*_latents_ema`（distinct5_512 / fewshot6 / random20 / samam）为历史备份。
- `data/train_d5` 里还有 TensorRT 现场（`encoder_b8.onnx`、`encoder_aoti_b8.pt2`、`sweep_b*.log`、`trt_cache/`），与数据无关，属可归档杂项。

## 2 18k（只有原图）

- 路径：`datasets/wikiart512_5style/train`，5 风格 × 3600 = 18000；实测样本 800×588（尺寸不一，训练时需 resize）。
- 无 latent、无 packed 缓存（`docs/HANDOVER.md#L160` 明确"不存在"）。要用必须先用当前 VAE 离线编码。
- 18k 测试集 `datasets/wikiart512_5style/test` = 150（5×30），与 D5 测试集内容不同。

## 3 评测集与协议

- 内容图：`data/test` = `datasets/d5/test`，150 = 5 风格 × 30；原图未预缩放（实测 1033×760），由评测脚本 resize。
- 配对：150 内容图 × 5 个目标风格 = 750 对；去掉对角线得到 **600 跨风格主表**，对角线 150 为 **同风格 Identity 补充**（`Exp.md#L3-L4`、`Plan.md#L3-L4`）。
- 参考图：固定策略 `first_image_of_target_style_test_dir`，即每个目标风格取 test 目录第一张；没有现成配对清单，由 `tools/run_eval_split.py` 生成（`docs/HANDOVER.md#L169-L172`）。
- 参考图池：就是各风格 test 目录 30 张；DINOv2 patch 特征缓存 `dino_cache/refs_test`（150 样本，见 `Plan.md#L22-L23`）。
- 18k 协议：用 `datasets/wikiart512_5style/test` 的 150 张按同样 600/150 拆分。

## 4 备选数据集（未进入当前主线）

| 路径 | 规模 | 风格 | 备注 |
|---|---|---|---|
| `datasets/r5/train` | 5000 | Cubism, Expressionism, Pop_Art, Romanticism, Symbolism | test 150；**与 D5 风格不同** |
| `datasets/wikiart_random20_512_latents_ema` | ~9900 latent | 20 风格（每风格 500，Abstract_Expressionism 890） | 有 latent |
| `datasets/wikiart_stress_splits_512/{wikiart_stress1,2,3}` | 带 manifest | 压力测试 split | 未用于主表 |
| `datasets/fewshot6_512` / `other5_512` | 160 / 152 | few-shot / 外部风格 | 仅鲁棒性实验 |

## 5 已知缺口与错记

1. `docs/NEXTGEN_RUNBOOK.md#L20-L25` 把 `data/train_d5_taesd/.latent_cache/packed` 写成"18k packed latent"，实测该目录不存在，且 `d5` 命名与 18k 无关——**失实，需改**。
2. `DATA.md`（根目录）把 `wikiart512_5style` 描述为"D5-512 源 / 同 5 风格"、把 `r5` 描述为"与 D5 同 5 风格"——与实测风格表冲突（见第 0 节）。
3. 750/600/150 的拆分在 `DATA.md` §6.1 被写成"5 内容图 × 30 × 5"，实际是 150 内容图 × 5 目标风格——**表述错误**。
4. 18k 缺 latent 是路线 C/P0 的前置阻塞项。

## 6 当前数据与缓存清单（2026-11-04）

> 本节记录 V2 主线在用的数据与离线缓存。路径以远端 `/mnt/i/Latent_Style/WEAVE` 为准；本机仓库没有 `data/`、`runs/` 产物，数字来自 3060 实测，**未在本机复核**。

| 对象 | 路径 | 形态 / 规模 | 用途 | 是否入库 |
|---|---|---|---|---|
| 训练集（5k D5） | `datasets/d5/train`（`distinct5_512/train`） | 5 风格 × 1000 张，512 长边 | V2/V3 唯一训练集 | 否（`datasets/` 被 `.gitignore` 忽略） |
| 512 原生 packed latent | `data/train/.latent_cache/packed/packed/*.pt` | 5 个文件，每个 1000 × **4×64×64** float32（约 313MB，合计 ~1.6G） | V2 `v2/data512.py` 直接读取，**不池化、不重编码**；同时提供 content latent 与源风格 id | 否（`data/` 被 `.gitignore:57` 忽略） |
| 逐样本 latent / 像素备份 | `data/train/<Style>/*.pt`、`data/train_d5/images_hwc_uint8.bin` | 1×4×64×64 / 3.93GB uint8 | 历史编码与重编码用，训练不读 | 否 |
| 评测集 | `data/test`（= `datasets/d5/test`） | 150 张 = 5 风格 × 30 | 600 跨风格主表 + 150 同风格 Identity | 否 |
| SWD 风格统计库 | `runs/v2/swd/library_5styles_512*.pt`（V3S 用的是 `library_5styles_512_enc.pt`） | 5 风格的小波/patch 描述子统计 | `v2/objective_swd.py` 的 SWD/MMD 目标；由 `tools/build_style_library.py` 生成 | 否（`runs/` 被 `.gitignore:24` 忽略） |
| 风格编码器权重 | `runs/v2/swd/style_encoder_512.pt` | 512 风格 token 编码器 | 给 V3S 提供参考风格特征；由 `tools/build_style_encoder.py` 生成 | 否（同上，且 `*.pt` 被忽略） |
| DINOv2 参考特征缓存 | `dino_cache/`（`refs_test` 150 样本） | patch tokens | 评测与参考图诊断 | 否 |
| 生成图 / 审计中间产物 | `runs/v2/**/images`、`runs/audit/*.json` | png 与 json | 图不入库；**纯文本/JSON 指标按需 `git add -f`**（`runs/` 默认被忽略） | 部分（仅 json/md/jsonl） |

`.gitignore` 覆盖检查：`data/`（第 57 行）、`runs/`（第 24 行）、`*.pt`、`*.png|jpg` 均已忽略，因此**数据本体、权重与生成图不会误入版本库**；需要在库内保留的只有脚本、配置、文档与 `runs/audit/*.json` 这类小体积指标文件（需显式 `-f`）。
