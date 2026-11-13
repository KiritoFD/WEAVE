# WEAVE / ICME：RTX 3060 12GB 新 Baseline 复现执行方案

> 审计日期：2026-11-13；状态：**源码和模型下载路径已核查，未在真实 3060 GPU 上运行**。
> 坚持三个层级：A. 源码可取得；B. 从架构/补丁看有希望落入 12GB；C. **12GB 实机 smoke + 完整输出通过**。本报告只完成 A/B 的核查；C 必须在用户的 3060 上执行。

## 0. 决策：不要把“有代码”当成“12GB 原样可复现”

| 方法 | 来源 | 代码 | 必需权重 | 官方是否证明 12 GB | 本次处理 | 结论 |
|---|---|---|---|---|---|---|
| **StyleGallery** | CVPR 2026, https://github.com/iiiiiiiword/StyleGallery | ✅ `demo.py`, `demo_accelerate.py`, `pipeline.py` | SD1.5、DINOv2-B；加速版 LCM/Hyper LoRA | **否**。官方只写 RTX 4090 | **P0：适配后实机验收，单参考，FP16，关 Depth/SAM、启 checkpoint** | **可执行的 12GB 适配方案；非已证实通过** |
| **Attention Distillation** | CVPR 2025, https://github.com/xugao97/AttentionDistillation | ✅ `ad.ipynb`, `webui/runner.py`, `pipeline_sd.py` | SD1.5 | **否**。未见官方 12GB 实测说明 | **P1：SD1.5、512、混合精度、checkpoint** | **优先做 12GB 可行性测试** |
| **SMS** | ICCV 2025, https://github.com/showlab/SMS | ✅ | SD1.5 + 目标风格 LoRA | **明确约 17 GB，建议 24 GB** | **不在 12GB 机器上安排完整原版复现** | 可引用、借鉴；24GB 另测 |
| **StyleSSP** | CVPR 2025, https://github.com/bytedance/StyleSSP | ✅ | SDXL、ControlNet、IP-Adapter、BLIP2 等 | **未证实** | 不投入 12GB 主线；若尝试需重构顺序 offload，不能宣称原版 | 暂排除 |
| **OmniStyle** | CVPR 2025 | ✅ 推理与权重 | 大模型 | **官方 38–46 GB 档** | 不在 12GB 上跑 | 暂排除 |
| AesPA / StyTr2 / SaMAM / CUT / SD-Turbo | 已有 WEAVE Test750 统一实测 | 已有 | 已有 | 不必重试 | 保留原图/原数据/原速度 | 主表继续使用 |
| StyleShot / StyleID | 历史高风格但严重掉内容 | 已测 | 已有 | 不必重试 | 主表可删除，Related Work 仍引用 | 不投入预算 |

**不要将 StyleGallery 的 12GB 低显存适配版称为“原论文原始配置复现”。** 需要区分：`SG-official`（FP32/Depth）与 `SG-LCM-12G`（FP16/no-depth/SAM-off/checkpoint）；后一种是基于作者公开加速入口的有限配置改动。

## 1. StyleGallery 源码深审（决定技术路线）

仓库：https://github.com/iiiiiiiword/StyleGallery
CVPR 2026 论文：https://arxiv.org/abs/2603.10354
作者环境：Ubuntu 20.04 / CUDA 12.6 / Python 3.10.18 / PyTorch 2.6.0 / RTX 4090。

### 1.1 原实现的三段

1. `cluster_match()`：SD1.5 对源图/参考图作 DDIM 反演并抽取 UNet 特征；GPU PCA/KMeans 聚类；DINOv2-B 对局部簇做语义匹配。`processed_data` 磁盘缓存可复用。
2. `style_transfer()`：UNet Self Attention 抽取 Q/K/V，在 `num_optimize_steps` 个时间步内直接反传优化生成 latent；内容和局部风格 loss 混合。
3. `latent2image()`：使用 SD1.5 VAE 解码成 512×512 RGB。

### 1.2 为什么不能直接在 12GB 跑官方命令

- `demo.py` 用 `StyleGallery.from_pretrained(...).to(device)`，没有指定 `torch_dtype=torch.float16`，默认常为 **FP32**。
- `demo_accelerate.py` 虽然加载 **FP16**，但 `style_transfer(enable_gradient_checkpoint=False)` 是默认值。注意这一步是**对 latent 的梯度优化**，不是普通无梯度的文生图。
- `--use_depth` 是 `action='store_true',default=True`，因而在 CLI 中**无法正常关掉深度分支**。
- `pipeline.py` 顶层导入 `pretrained_models.dpv2.depth_anything_v2.dpt`，但官方仓库 `pretrained_models/` 仅含占位文件。即使不想用深度，也会先导入报错。
- FP16 下聚类的 `torch.pca_lowrank` / 小型 Transformer 可能发生 fp16 数值或 dtype 不一致；补丁把这两个小模块固定为 fp32，但不改风格优化目标。
- 参考图越多，style feature、Self-Attention 参考缓存越大。**12GB 第一轮只用 1 张参考图/目标风格**，不直接搬作者多参考设置。

### 1.3 已附 12GB 兼容补丁

`patch_stylegallery_12g.py` 做 7 项受控改动：

1. Depth Anything V2 改为**可选导入**；关闭深度时不再要求安装它。
2. 默认 `use_depth=False`；保留 `--use_depth` 显式开启路径。
3. 聚类 PCA 的 feature 和 latent 转 fp32。
4. 语义匹配的小 Transformer 与其输入都为 fp32。
5. 新增 `STYLEGALLERY_DINO_CPU=1`：显存不足时把 DINOv2-B 放 CPU（会慢很多）。
6. `demo_accelerate.py` 调用 `style_transfer()` 时启用梯度检查点。
7. **不**修改 UNet、Self-Attention loss、scheduler、时间步数与算法权重。

### 1.4 环境、模型和下载（在 RTX 3060 的 WSL2/Linux 内执行）

> 推荐一个**独立 conda 环境**，不要与 WEAVE 的 torch/TRT/SD-VAE 安装混用。GPU 使用前 `nvidia-smi` 确认有至少约 11GB 空闲、没有其他任务。

```bash
# 将下载包中的 12G 脚本放到 3060 上，例如 ~/weave_baselines/
# 环境安装 + 权重下载 + 补丁
bash ~/weave_baselines/setup_stylegallery_12g.sh ~/StyleGallery
conda activate sg12
cd ~/StyleGallery
```

脚本按作者版本安装 Torch 2.6 / Diffusers 0.36 等，并从 Hugging Face 下载：

- SD1.5：`stable-diffusion-v1-5/stable-diffusion-v1-5` → `pretrained_models/runwayml_stable-diffusion-v1-5/`
- DINOv2-B：`facebook/dinov2-base` → `pretrained_models/facebook/dinov2-base/`
- LCM LoRA：`latent-consistency/lcm-lora-sdv1-5` → `pretrained_models/lcm-lora-sd1.5/`
- Hyper-SD（备选）：`hf download ByteDance/Hyper-SD Hyper-SD15-4steps-lora.safetensors --local-dir pretrained_models`

优先 LCM，避免无谓下载 Hyper 与 DepthAnything。PyTorch wheel 官方同时提供 torch2.6.0 + torchvision0.21.0 `cu126`，适合上述版本。

作者 `requirements.txt` 列了很多仅用于 Gradio / 人脸增强的包，最小批评测路径不需要一起装；安装脚本使用依赖闭包更小的子集。若启动日志出现真实缺包，再追加并在环境锁文件记录；**不要随手升级整套 diffusers/transformers**。

### 1.5 Step 0：12GB 实机 Gate（必需，不能跳过）

```bash
cd ~/StyleGallery
conda activate sg12
nvidia-smi
bash ~/weave_baselines/run_stylegallery_smoke_12g.sh \
  content/city_scene/0002.png style/VanGogh/VanGogh010.png
```

这只是**缩减到 12 优化步、1 iteration 的显存验证**，不是最终论文质量设置。

验收内容：

- 存在 `outputs/sg12_smoke/result_lcm.png`，可打开，512×512，非整黑/整白。
- GPU 无 OOM；VRAM 峰值最好低于 11.5GB，给碎片与偶然波动留余量。
- `--use_depth` 默认关闭且没有 SAM；整个步骤可在单张 12GB RTX 3060 上完成。
- 记录分阶段 wall-time（聚类/匹配、风格优化、解码）及精度版本；**本报告没有 GPU 实测，不保证此 Gate PASS**。

如因 CUDA OOM：先重试 `STYLEGALLERY_DINO_CPU=1 bash ...`（DINO CPU 只是预处理），再确认 checkpoint 已启用、环境无其它 GPU 任务；仍 OOM 则该方法在 12GB **列为不适配/复现失败**，不能声称完成，完整原版留给 4090。

如因数学/版本错误失败：不能靠改目标损失或降分辨率来宣称原论文已复现；需要单独记录为 `engineering_failure` 和修改代码/版本的范围。

### 1.6 Step 1：作者加速质量设置（smoke 之后）

StyleGallery 原 `demo_accelerate.py` 两套预设：

| 配置 | steps | iters | lr | c_ratio | 参考图 | 类型 |
|---|---:|---:|---:|---:|---|---|
| **LCM 官方加速入口** | **28** | **3** | **0.25** | **0.33** | 单张 | **主比较（SG-LCM-12G）** |
| Hyper-SD 官方加速入口 | 28 | 3 | 0.15 | 0.38 | 单张 | 备选 |
| `demo.py` 官方常规 | 150 | 1 | 0.05 | 0.26 | 单/多张 | 24GB/4090 优先，12GB 不承诺 |

```bash
cd ~/StyleGallery
conda activate sg12
python demo_accelerate.py \
  --mode lcm \
  --model_name pretrained_models/runwayml_stable-diffusion-v1-5 \
  --lora_path pretrained_models/lcm-lora-sd1.5 \
  --content_image /absolute/path/source.jpg \
  --style_images /absolute/path/style_reference.jpg \
  --output_folder outputs/sg12_full \
  --noise_steps 15 --num_optimize_steps 28 --iters 3 \
  --lr 0.25 --c_ratio 0.33 --seed 42
```

若 28×3 在 12GB 峰值安全，则该作者加速入口的单参考版本可列为 **12GB 可复现**；若作者设置不行而仅 12×1 可行，只能标 `SG reduced compute`，不能当正式公允 Baseline。

### 1.7 Step 2：冻住参考集，跑 WEAVE Test750

WEAVE 仓库中有 `eval_protocol/tasks_750.json`，明确列出 150 来源图像、5 目标画派、600 cross / 150 same，并带 `source_rel_path` 等字段。

在 WEAVE 的真实 dataset 中，每目标画派从 **`data/train/<style>/`** 选取固定的 1 张参考画，固定 Seed 42，不在 `data/test` 或评价参考图里挑选。

```bash
# 先离线生成一张参考/目标画派的任务表（可以在本地或 3060 运行）
python ~/weave_baselines/prepare_weave750_sg_manifest.py \
   --weave-root /mnt/i/Latent_Style/WEAVE \
   --tasks eval_protocol/tasks_750.json \
   --ref-root data/train --refs-per-style 1 --seed 42 \
   --out ~/sg750_manifest.csv --refs-out ~/sg750_refs.json

# 先运行 10 个任务检查完整 pipeline，使用绝对路径
conda activate sg12
python ~/weave_baselines/batch_stylegallery_12g.py \
  --repo ~/StyleGallery --manifest ~/sg750_manifest.csv \
  --out ~/sg750_lcm --limit 10 --steps 28 --iters 3

# 通过后全量、串行、断点续跑（文件存在则跳过）
python ~/weave_baselines/batch_stylegallery_12g.py \
  --repo ~/StyleGallery --manifest ~/sg750_manifest.csv \
  --out ~/sg750_lcm --steps 28 --iters 3
```

批量脚本逐图写 `sg750_lcm/<index>_<id_hash>.png`、单图日志和 `execution.jsonl`，**失败会停止**，不会用 identity 或上一轮结果冒充生成。若 GPU 碎片问题，追加 `--cpu-dino` 重试（作为另一个明确记录的预处理配置）。

**重要：批量脚本为保证 12GB 稳定，每张图重新启动 Python 并加载模型。它测得的 `cold_wall_s` 包括初始化/读取权重，不能直接与 WEAVE 的 warm end-to-end ms/img 横比。** 论文速度必须另用长驻进程、同设备、batch1，并同时记录 `preprocess+matching+optimization+decode` 和 `warm optimization+decode`。

所有参考选择、任务来源、参数、源码 commit、日志和模型哈希写进 provenance。不用生成图像进行选参。

## 2. Attention Distillation：12GB 独立备选

- 官方代码：https://github.com/xugao97/AttentionDistillation
- 该仓库明确提供 SD1.5 的 **Style/Appearance Transfer**，入口为 `webui/runner.py::run_style_transfer` → `ADPipeline.optimize()`；不需要作者另行训练的专用权重。
- UI 默认 `num_steps=200`，`lr=0.05`，`content_weight=0.25`，精度通常 `bf16`。
- 官方 `runner.py` 给 style transfer 的 `enable_gradient_checkpoint=False`；附带安装脚本使其变为 True。

```bash
bash ~/weave_baselines/setup_attention_distillation_12g.sh ~/AttentionDistillation
conda activate ad12
cd ~/AttentionDistillation
python ~/weave_baselines/run_ad_smoke_12g.py \
  --content /absolute/path/source.jpg \
  --style /absolute/path/ref.jpg \
  --model pretrained_models/stable-diffusion-v1-5 \
  --steps 20 --precision bf16
```

注意：这是 **20 步 smoke**；正式比较要恢复作者的 200 步，若 12GB 实机不行，不能用 20 步失败/成功代替论文默认成绩。AD 是单张参考迁移，和 WEAVE 的五类 Style-ID 不同，需按同一冻结参考方案进行 Cross600。

## 3. 必须复用的 WEAVE 统一评测协议

- **图像与任务**：原始 `tasks_750.json`，Cross600 与 Same150 单独列；输出统一到当前 WEAVE 评测的 RGB、尺寸、预处理。StyleGallery 自身 512×512，不需重复 VAE 解码。
- **质量**：DINO-S、DINO-C、LPIPS-Alex、CSD Top-1、Target Margin、G、CLIP-S/Style Affinity；沿用 `src/eval` 实现，禁止新建不同指标。
- **对齐**：使用 `task_id` 精确关联，不按图片位置/遍历顺序盲配。仅完整 750 且 0 failed 的 arm 可进入主表。
- **指标泄漏**：风格参考图库取 train-only；与训练/评测 DINO-S references 做去重审计。不要通过目标评测分数挑选最佳参考图片。
- **参考条件公平性**：StyleGallery/AD 是 image-guided；WEAVE 是 Style-ID。主表明确“单参考图片”相对“类条件”，避免暗示它们输入相同。
- **耗时**：同时报告作者版本 warm generation、冷启动与匹配预处理成本；禁止拼接不同版本/设备时延。速度不能仅以 GPU kernel 耗时替代实际完整推理。
- **失败回退**：不得用更多参考、改用其它 backbone、降低图像分辨率或手工提示词而不标记。这些可以作为**新配置**，不能称原配置复现。

## 4. 决策门（按此决定是否投入全量）

| Gate | 条件 | 行动 |
|---|---|---|
| G0 代码及权重齐全 | pip 环境、SD1.5、DINOv2-B、LCM LoRA 均成功载入 | 执行单图 |
| G1 **12GB 可运行** | 512×512 输出有效、无 OOM、显存有余量 | 开始 10 图 |
| G2 风格有效 | 10 图至少肉眼可辨风格变化，无全黑/重复严重 | 开始 50 图 |
| G3 50 图稳定 | 0 OOM、0 缺文件、0 NaN、runtime 可接受 | 决定是否跑 750 |
| G4 750 完整 | 600+150 均有 PNG、task_id 严格一致 | 与 N2/F 统一评测 |
| G5 完整对照 | 质量 × 速度 × 规模 × 参考条件 | 进入 ICME 主表 |

## 5. 不建议浪费时间的路线

- **SMS**：官方 README 明确优化要 ~17GB，24GB 推荐。不能在 12GB 上承诺。先用论文与代码对照研究 Score、频谱正则、语义梯度，不妨碍 WEAVE 的低开销主线。
- **StyleSSP**：默认脚本使用 SDXL、BLIP2-flan-t5-xl、ControlNet、IP-Adapter 等，作者没有提供 12GB 完整复现证据。要在 12GB 上跑需大规模 CPU/offload/8bit 改造，优先级低于 StyleGallery/AD。
- **OmniStyle2 / GST**：公开代码或可用权重尚不完整；直接说明无法复现，不耗预算。

## 6. 已核查权威链接

- StyleGallery code & README: https://github.com/iiiiiiiword/StyleGallery
- StyleGallery paper: https://arxiv.org/abs/2603.10354
- AD code: https://github.com/xugao97/AttentionDistillation
- SMS code + **17GB**: https://github.com/showlab/SMS
- StyleSSP code: https://github.com/bytedance/StyleSSP
- SD1.5 model: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- DINOv2-B: https://huggingface.co/facebook/dinov2-base
- LCM LoRA: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
- Hyper-SD LoRA: https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SD15-4steps-lora.safetensors
- PyTorch 2.6 CUDA 12.6 wheels: https://pytorch.org/get-started/previous-versions/

**最后结论**：本次已经在官方源码层面完成 12GB 适配的可执行设计与脚本静态检查；由于没有访问真实 RTX 3060 + 外部模型权重，尚未完成硬件显存验收。任何“已确认 12GB 可跑”的更强表述都不真实。

### AD Test750 批量命令（通过 20 步与 200 步实机验收后）

```bash
conda activate ad12
python ~/weave_baselines/batch_ad_12g.py \
  --repo ~/AttentionDistillation \
  --manifest ~/sg750_manifest.csv \
  --runner ~/weave_baselines/run_ad_smoke_12g.py \
  --out ~/ad750_sd15 --steps 200 --precision bf16
```

与 SG 一样，AD 的批量入口是 **重启每图进程的稳定性优先实现**，冷启动时延不能当作发表时的 warm GPU inference speed；其本质是确保全量结果图可完成评测。正式计时需另外固定模型常驻状态及显存峰值。

## 3. 3060 实测记录（2026-11-13，WSL2 / RTX 3060 12GB）

环境：系统 python3（torch 2.10+cu128 / diffusers 0.41 / transformers 5.15），未建独立 venv；补丁 `patch_stylegallery_12g.py` 全部命中（optional depth import、no-depth 默认、FP32 PCA/clustering/Transformer、DINO 可选、gradient checkpointing、`vae.enable_slicing()` 兼容）。

权重（均本地已有，未重复下载）：
- SD1.5 diffusers：`/mnt/i/Latent_Style/weights/sd15-diffusers`（45GB 全仓含冗余，实际组件 ~5GB）
- DINOv2-B：`/mnt/i/Latent_Style/weights/dinov2-base`
- LCM-LoRA：`/mnt/i/Latent_Style/weights/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors`

### 3.1 单用例测速（Early_Renaissance 内容 × Ukiyo_e 风格，seed 42）

| 配置 | 结果 | wall | 峰值显存 |
|---|---|---:|---:|
| 12×1（smoke） | ✅ | 57s | ~4.7GB |
| **28×3（官方 lcm）** | ✅ | **66s** | ~4.7GB |
| 50×1（LCM 调度器上限） | ✅ | 59s | ~4.6GB |
| **150×1** | ❌ | — | `LCMScheduler` 拒绝：`original_steps(50)×strength(1.0) < num_inference_steps(150)`；官方 lcm 入口本就用 28×3，150 属 `demo.py` 常规档（需 24GB 级配置，不在本复现范围） |

并发测试（同一官方 28×3 配置，不同图对）：

| 并发 | 结果 | 总耗时 | 吞吐 | 峰值显存 |
|---|---|---:|---:|---:|
| ×1 | ✅ | 66s/张 | 0.91 张/min | ~4.6GB |
| ×2 | ✅ | 149s/2张 | 0.80 张/min | 7.9GB |
| ×3 | ✅ | 220s/3张 | 0.82 张/min | **11.7GB（95%）** |

**结论：并发无加速**（瓶颈是 UNet 前向+反传算力而非加载），×3 已贴近 OOM 边界。750 全量采用**串行冷启动**，预计 ~14h。

### 3.2 已挂全量任务

- manifest：`sg750_manifest.csv`（750 任务，source=`data/test/<style>/`，refs=`datasets/distinct5_512/train_512clean/<style>/` 每画派固定 1 张、seed 42、**train-only 无泄漏**；`data/train/` 只有 .pt latent 不能作参考图，故改用 train_512clean）
- 运行：tmux `sg750`，`batch_stylegallery_12g.py --mode lcm --steps 28 --iters 3 --seed 42`，输出 `sg750_run/`（png + per-task log + execution.jsonl），失败即停不跳过
- 注意：`--model_name` 必须用软链名 `runwayml_stable-diffusion-v1-5`（下划线）；连字符会被 diffusers 当 HF repo id 联网拉取而失败

### 3.3 常驻进程优化 runner（`batch_stylegallery_fast.py`，2026-11-13 深夜启用）

**测速口径约定**：论文/表格中 StyleGallery 的单图耗时一律引用**官方脚本冷启动口径 ~76–78s/张**（`batch_stylegallery_12g.py` 逐任务子进程，含模型加载+重复聚类特征计算）。常驻 runner 的加速是**我方工程优化**，其耗时单独记录、不冒充 baseline 原始速度。

优化内容（算法不变，详见脚本 docstring）：
- 模型常驻：SD1.5/UNet/VAE/DINOv2/Transformer 只加载一次（原脚本每任务重载 ~10s）
- per-image 聚类特征磁盘缓存：`match_content_style_clusters` 的单图部分（combined_feature/圆/dinov2_feature）与配对无关，150 内容+5 风格共 ~155 次计算代替 ~1500 次
- 调度器按阶段切换 DDIM↔LCM、LoRA 每任务 fuse/unfuse，保持官方顺序

**已知官方代码怪癖（如实记录）**：`match_content_style_clusters` 每次调用新建**随机初始化 Transformer**，官方逐任务子进程下其权重依赖 RNG 状态（缓存命中与否都会改变匹配结果）。fast runner 固定 `manual_seed(42)` 单例 Transformer —— 全程确定且自洽，属于对官方非确定行为的显式固定，已在脚本注释声明。

实测（同一 manifest、同输出目录、可断点续跑）：

| runner | 稳态耗时 | 备注 |
|---|---:|---|
| 官方逐任务子进程 | ~76s/张 | 19 任务均值；论文引用口径 |
| fast_persistent | **37–55s/张**（缓存命中 ~37s，新内容图 ~50s） | 峰值显存 ~5.0GB |

预期 ETA：150×~50s + 600×~37s ≈ **7.5–8h**（加每 200 任务一次重启开销）。

运行方式：tmux `sg750f`，`until DONE; do python3 batch_stylegallery_fast.py ... --max-tasks 200; done`；失败任务记录 `*.error.log` 并继续，重启轮自动重试。
