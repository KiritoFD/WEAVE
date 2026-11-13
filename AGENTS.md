# WEAVE — Agent Memory（必读）

## 决策协议（2026-11-14 起，覆盖一切历史文档的相反约定）

### 解码器：一律用 S10，不用原版 SD-VAE decoder

- **所有评测/出图解码使用 S10**（`deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt`，10.66M 剪枝 decoder，自带 `post_quant_conv`）。
- 历史文档里"正式质量用标准 SD-VAE 口径；S10 仅作部署对照"（docs/v8/README.md 等）**已作废**——实测 S10 在同协议下全面更优，属必须铭记的结论。
- **原版 decoder 文件已删除**：`G:\eval_cache\hf\hub\models--stabilityai--sd-vae-ft-ema`（含 encoder+decoder）。如需重新编码 latent 要重新下载；解码永远不需要它。

### S10 使用契约（错了会出垃圾图，历史上踩过）

```python
from src.utils import load_s10_decoder, decode_s10
s10 = load_s10_decoder("cuda")          # StudentDecoder(pqc, dec).half()
rgb01 = decode_s10(s10, z_scaled)       # z 须为 repo 存储的缩放 latent，内部自动 /0.18215
```

1. 输入是**未缩放 latent**：repo 存的 `z` 是 scaled（`encode()*0.18215`），喂 decoder 前必须 `/scaling_factor`——`decode_s10` 已封装，**不要绕过它直接调模型**。
2. `post_quant_conv` 用 checkpoint 自带的，**恰好应用一次**（StudentDecoder 内部已含，勿再手动套）。
3. 不调 `AutoencoderKL.decode()`/`vae.decode()`——那是原版 decoder 路径，文件已删。

### 为什么 S10 更好（证据，勿再质疑）

- Q2A+Oracle1 × 750 任务配对评测（`deliver/q2a_oracle_scores_baseline_protocol.json`，cross600）：
  - CSD Top-1：old_s10 **.3767** vs sdvae .3450（**+3.2pp**，bootstrap CI 排除 0）
  - CSD Margin：−.0411 vs −.0551；LPIPS .5034 vs .5149；same150 Top-1 .96 vs .94
  - 代价：DINO-S/C 各 −.010（风格略降），但 Top-1/Margin/LPIPS 净胜
- `docs/vae/S10_READY_TO_USE.md`：SD0 Encoder + 旧 S10 直接用，Affinity .5616 > 官方 SD0 decoder .5548，零重训。（docs/v6 同名文件已改为指针 stub，权威在 docs/vae）
- `docs/v6/Q2A_ORACLE_DECODER_RANKING.md`：old_s10 在 4 变体中 Top-1 排名第 1。
- S10B（`deliver/checkpoints/s10b_standalone/`）与 s10_p5_lowlr **字节级同一权重**，只是为 SD0 未缩放 latent 打包的独立加载器；本仓库 SD1.5 latent 用 `s10_p5_lowlr/decoder.pt` + `src.utils.load_s10_decoder`。

## 仓库布局（2026-11-14 重组后）

- **活跃代码**：`src/`（唯一门面层：model/infer/eval/utils）+ `exp/v8`、`exp/v9_local`、`exp/v10`（实验包，`python -m exp.*` 从仓库根运行）。均已 git 跟踪。
- **tools/** = 历史代码归档，但内含**钉住子包**（`vae_suite/decoder_prune.py`、`v6*`、`v7*`、`pseudo_gt`、`12h/csd_model.py` 等）被 src 依赖，**不要移动/重构**——清单见 `tools/README.md`。
- `tools/legacy_root/` = 论文期根目录平铺模块（model.py/trainer.py/run.py…）+ 一次性脚本；自举包，flat import 仍可用；根 `__init__.py`/`__main__.py` 是兼容 shim。
- `tools/legacy_utils/` = 原 `utils/` 包，全仓以 `tools.legacy_utils.*` 引用。
- **`paths.py` 在仓库根，位置定义 `WEAVE_ROOT`，不要移动。**
- `archives/` = 冷存（_tmp、EXPERIMENT_ARCHAEOLOGY、agent 工作区、exp 历史输出等；gitignore）。
- 文档入口：`docs/README.md`（索引）→ `docs/HIGHLIGHTS.md`（两大进展：S10 decoder、Q2A+Oracle）→ 阶段目录。根目录 README/REPRODUCE 是论文口径，与当前研究线并存。
- 无关项目 `foi26/`（复旦校赛）已移出仓库至 `G:\foi26\`，与本仓库无关。

## 当前进行中的工作

- **V10 已收尾**（2026-11-14）：真实画作条件 KDE score 路线——score 级验证全过（方向 .302 vs r1 .054），750 口径最优臂 E4（cross600 Top-1 .3400 vs LD0 .3283，dino_s .4415/affinity .5616 全场最高）但未超 LD4；DSM < KDE（保留闭式）。主表与结论 `docs/v10/V10_RESULTS.md`；代码 `exp/v10/`，产物 `runs/v10_local_4070/`（untracked）。
- **V10.1 局部匹配已判负**（2026-11-14）：StyleGallery 式"内容条件检索目标 + 距离损失"在恢复代理上反向对齐（q 检索 −0.25~−0.53 vs KDE +0.30）——处方式目标形态错误，KDE 密度梯度才是对的形态；详见 `docs/v10_1/V10_1_RESULTS.md`。
- **V10.2 特异性验证已收尾**（2026-11-14）：**恢复对齐收益是通用的，不是画派的**——混合画派 KDE 拿到目标 KDE 的 95%（特异性 5%）；q 条件均值拉动 +.49 但特异性 0.9%；LD4 机制换装差异 Score（C-TM/C-TS）Top-1 +0.67~0.84pp vs LD0，未达 LD4。**v11 重启门槛四项全未达标**；latent 统计特征空间的画派特异信号已近枯竭，下一步要么学习式编码器（v11，密度目标）要么换特征空间。详见 `docs/v10_2/V10_2_RESULTS.md`。方法论新增：**Recovery Alignment ≠ Style Specificity**（score 路线必须同时报告 A1/A2/A3 三控制）。
- **V11 已延期**（底稿 `docs/v11/WEAVE_V11_LOCAL_STROKE_MEMORY_PLAN.md`）：P0–P2 已被 v10.1 吸收并判负；P3–P5（学习式 Attention 分支）重启前提 = v10.2 门槛，未达标；若重启，监督用密度提升而非距离回归。
- **评测口径（2026-11-14 用户决定）**：任务级评测/决策一律用完整 test750（读数 cross600），不用 dev 预览集；LD 基线 = v9 dev+test eval 行合并重算。
- 3060 被他人占用做 StyleGallery 复现，**不要碰**。

## 环境备忘

- 本地 GPU：RTX 4070 Laptop 8GB（实际可用 ~6GB）；评测解码/指标需分进程，勿同时驻留多模型。
- 评测指标管线：`src/eval/measure.py`（DINOv2-small/CSD-ViT-L/CLIP/LPIPS，冻结 referee）。
- tasks：`eval_protocol/tasks_750.json`（750=Cross600+Same150，5 画派）；V9 dev/test 按 md5(source_stem)%5 划分（dev=170/test=580，互补并集=750）。
- **图像数据已全部本地化**：`datasets/wikiart_new_512/`（10,545 张，从原图重建，sha256 校验 7,678 字节级一致 + 2,867 jpeg 编码器版本漂移——像素相同，逐文件状态在 `weights_local/wikiart_new_512/rebuild_status.jsonl`）；`datasets/distinct5_512/train/`（5,000 张，hardlink 自 `G:\GitHub\WEAVE\train_512clean`）；`data/test/` 150 源图 + 150 参考图已就位。
- **J 盘原图**：`J:\WEAVE_data\{train,test}` 是 wikiart dump 原图（数字 id 命名，8100+2445=10,545 张，与 `weights_local/wikiart_new_512/manifest.jsonl` 的 id 一一对应）。已通过 junction 挂到仓库：`datasets/wikiart_raw/train`、`datasets/wikiart_raw/test`（NTFS junction，删了链接不影响 J 盘数据；`datasets/` 整个目录 gitignore）。另有 `J:\WEAVE_data\wikiart_new_512_images.tgz`（6.1GB 原始打包，与解压后内容重复，一般不用）。
- 已知偏差：本地曾用 VAE fallback 解码（~8.6/255 MAE vs TRT），**现已全部改 S10**；S10 vs 原版 decoder 的差异是协议级选择，不是误差。
