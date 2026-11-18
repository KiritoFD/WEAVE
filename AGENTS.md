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

- **活跃代码**：`src/`（统一门面与基础设施层：S10 解码、自包含 Haar/LCN/Oklab 结构层 `src.utils.structure`、3060 自适应微批分块 `tier_micro_accum`、清理 `expandable_segments` 的进程隔离裁判执行器 `src.eval.runner`、无碰撞错标签与协议辅助 `src.utils.protocol`）+ `exp/v8`、`exp/v9_local`、`exp/v10`..`exp/v20`（实验包，`python -m exp.*` 从仓库根运行，索引见 `exp/README.md`）。全仓活跃代码受 `tests/test_no_absolute_paths.py`（`KNOWN_DEBT = set()`）与 `tests/test_markdown_links.py` 保护。
- **性能与论文中心**：`docs/performance/`（唯一权威数据台账 `metrics.{csv,json}` 含 **257 条**实验臂与外部基线 / **98 个帕累托点**、总台账 `MASTER_LEDGER.md`、多维矢量图表 `figures/*.svg`、750 生成图库 `images/`）；`docs/icme2027/`（独立论文 `PRISM` 完整 6 页 `IEEEtran` 工程 `paper.tex/pdf`）；`docs/next1118/`（`PRISM v2 / V19` 与 `PRISM-Flow / V20` 理论与实测报告 `V19_RESULTS.md`、`V20_RESULTS.md`）。
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
- **V10 / V10.1 / V10.2 / V11 / V12 / V13 全线关闭**（2026-11-14）：六条风格路线（手工统计 Score、检索目标、DINO 密度 Score、GCL 内容闸门 ×5、DWT 域对比学习 E_s）全部判负。三条方法论铁律：**Reference Retrieval ≠ Style Distribution Gradient**；**Recovery Alignment ≠ Style Specificity**；**Separation Margin ≠ Generative Direction**；**DWT Token ≠ Style Carrier**。完整证据链见各自 RESULTS 文档（docs/v10/ v10_1/ v10_2/ v11/ v12/ v13/）。
- **当前最优交付模型族（2026-11-18，V19–V20 / `PRISM` 全量实测收官）**：
  - **`PRISM-1` (`V19_S2_prism1_v2`, `exp/v19/`)**：`1.50M` 活跃潜参数纯单步前馈（`17.0 ms/img` = `58.7 img/s` @ 4090，`~29 ms` @ 3060），**100% 免测试期源标签 `sid`**，Cross600 **CSD Top-1 = `44.33%`**（`DINO-C = 0.7440`, `LPIPS = 0.5057`，反事实错标签 `0.83%`，全盲 Same150 `99.3%`；1-Shot 单参考图达 **`45.50%`**；$\lambda=1.3$ 达 **`50.17%`**），在全部五项风格与内容指标上严格支配商业 `Seedream 4.5`（`37.30%`）与 `StyTr2`（`39.50%`）。
  - **`PRISM-Pro` / `PRISM-Pro+` (`V19_T4` / `V20_A1_t4_patchswd`)**：穿过可微 `S10` 的无归一化 `VGG-16+Oklab` + 多尺度 `ConvNeXt+RGB Patch-SWD` 各向异性求解器，Cross600 **CSD Top-1 = `59.00%` → `64.33%`**（`DINO-C = 0.6783` / `0.6392`, `LPIPS = 0.5166` / `0.5332`）。
  - **`PRISM-Flow` (`V20_D1..D7`, `exp/v20/`)**：纯训练集内部 `kNN-30` 核心流形原型 + 仅 $16\times 16$ 瓶颈层注入结构语义的 Coarse-16 `SLFUNet` 分段视界流（`76.6–134.1 ms/img`，12GB 3060 训练峰值仅 `4.06 GB`），全面超越扩散跨注意力强基线 **`StyleGallery`**（`DINO-S = 0.5604`, `Affinity = 0.6432`, `Top-1 = 52.50%`）：`V20_D1` 达 **`DINO-S = 0.5789`、`CLIP-S = 0.7576`、`Style Affinity = 0.6683`、`CSD Top-1 = 64.00%`**（`Minimalism = 100.0%`）；`V20_D2` 达 **`DINO-S = 0.5672`、`Affinity = 0.6584`、`Top-1 = 60.50%`**（错标签控制 `2.33%`，因果差 `+58.17pp`）。
  - 完整证据链见 `docs/next1118/V20_RESULTS.md`、`docs/next1118/V19_RESULTS.md`、`docs/performance/MASTER_LEDGER.md` 与 `docs/icme2027/paper.pdf`。
- V11b（波let Flow）代码/修复版在 `exp/v13/`（8 bug 修复，sigma √2 精确验证），3060 上有远程重建脚本。
- **评测口径（2026-11-14 用户决定）**：任务级评测/决策一律用完整 test750（读数 cross600），不用 dev 预览集；LD 基线 = v9 dev+test eval 行合并重算。
- **V15 / V16 / V17 全线定论（2026-11-15～17）**：
  - V14 多尺度扫描确认 224px DINOv2 16×16 patch（NCM `.793` >> DWT `.544`）为最佳语义载体（`docs/v14/V14_RESULTS.md`）。
  - V15 Group A（16 臂）+ Scaling（`dim96→384`, 34.2M）+ V16 `A_strict` 证实：**Self-Paired Generative Capacity ≠ Cross-Style Editing**——扩容使 $\tau_0=0$ 生成 Top-1 升至 `49.3%`，但在 $\tau_0=0.7$ SDEdit 编辑点因保源增强使 Cross600 Top-1 从 `10.67%` 跌至 `6.67%`；57.8% Impressionism 不平衡使底座在 `ER` 与 `Ukiyo-e` 上先验坍缩为 `0%`；唯 `Style-Bridge FM`（`PT2_P2`）在无额外推理开销下拉升了目标判别。
  - V16 B/C 与 V17 P0–P4 证实：CFG（`w=2,4`）、Signed RGB-H2（`H4` `11.00%`）、五类均衡自配对 FM 残差（`F1` `11.33%`）均受困于 $\tau_0=0.7$ 的 59% 源画派锁定；但 **V17 `P2_S4`（12k 步 + DINO patch 上下文 + `u_tgt − u_src` 无压缩差分）达 `13.33%`，反超 `B_teacher`（`11.83%`）+1.50pp**，解决了单步学生拟合难题。
- **V18 / V19 突破已完成（2026-11-18）**：V18 确立 Gate-0 去风格结构（`18.3%` 探针）与 $\tau_0=0$ 自配对控制流的 `18.67%` 几何天花板（`docs/v18/V18_RESULTS.md`）；V19 建立正交商—纤维对偶 $(\Pi_\perp, \Pi_\parallel)$ 与各向异性 `S10-VGG` 谱传输（`exp/v19/`，`docs/next1118/`），750 全量评测与输出图库已归档至 `docs/performance/`。

## 环境备忘

- **远程主力算力**：`ssh 4090`（RTX 4090 24GB，`/root/Workspace/WEAVE`）与 `ssh 3060`（RTX 3060 12GB Windows Server，禁用 `expandable_segments`，微批由 `src.utils.tier_micro_accum` 自动限幅至 $\le 6\text{GB}$）；**本地 GPU**：RTX 4070 Laptop 8GB（仅轻载，重训练/评测一律上远程并严格分进程运行 `src.eval.measure`）。
- 评测指标管线：`src/eval/measure.py`（DINOv2-small/CSD-ViT-L/CLIP/LPIPS，冻结 referee）。
- tasks：`eval_protocol/tasks_750.json`（750=Cross600+Same150，5 画派）；V9 dev/test 按 md5(source_stem)%5 划分（dev=170/test=580，互补并集=750）。
- **图像数据已全部本地化**：`datasets/wikiart_new_512/`（10,545 张，从原图重建，sha256 校验 7,678 字节级一致 + 2,867 jpeg 编码器版本漂移——像素相同，逐文件状态在 `weights_local/wikiart_new_512/rebuild_status.jsonl`）；`datasets/distinct5_512/train/`（5,000 张，hardlink 自 `G:\GitHub\WEAVE\train_512clean`）；`data/test/` 150 源图 + 150 参考图已就位。
- **J 盘原图**：`J:\WEAVE_data\{train,test}` 是 wikiart dump 原图（数字 id 命名，8100+2445=10,545 张，与 `weights_local/wikiart_new_512/manifest.jsonl` 的 id 一一对应）。已通过 junction 挂到仓库：`datasets/wikiart_raw/train`、`datasets/wikiart_raw/test`（NTFS junction，删了链接不影响 J 盘数据；`datasets/` 整个目录 gitignore）。另有 `J:\WEAVE_data\wikiart_new_512_images.tgz`（6.1GB 原始打包，与解压后内容重复，一般不用）。
- 已知偏差：本地曾用 VAE fallback 解码（~8.6/255 MAE vs TRT），**现已全部改 S10**；S10 vs 原版 decoder 的差异是协议级选择，不是误差。

## V15 4070 后训练（2026-11-16 完成）

- 方案/结果：`docs/v15/WEAVE_V15_POSTTRAIN_4070LP.md` → `docs/v15/WEAVE_V15_POSTTRAIN_4070LP_RESULTS.md`
- 基线 `W_dim96_conv_base` 本地复现**像素级验证通过**（z_final/rgb 与 3060 差 ≤2/255）；quick dino_s 残差 ±0.003 是本地 DINO 裁判权重版本差，非协议错误。
- 代码：`exp/v15/eval_4070lp.py`（评测+cfg/shift 控制）、`exp/v15/posttrain_4070lp.py`（balance/band/guided）、`exp/v15/h2_apply_4070lp.py`、`exp/v15/h2_train_4070lp.py`。
- 结论：balance/band/小批量对照三种后训练全部中性（±噪声内）；R1-guided 微调**崩坏模型**（dino_c .75→.26，判 FAILED）；V11b 旧 ckpt 与 RGB 重训 H2 在当前 edit_bands 强度下都是无操作（像素改动 <1/255）。
- 显存：训练峰值 ≤58MB，评测 ~3.7GB；**同卡并发 GPU 任务 ≤2 个**（>3 个会 OOM）。
