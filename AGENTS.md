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

- **活跃代码**：`src/`（唯一门面层：model/infer/eval/utils）+ `exp/v8`、`exp/v9_local`（实验包，`python -m exp.*` 从仓库根运行）。两者均已 git 跟踪。
- **tools/** = 历史代码归档，但内含**钉住子包**（`vae_suite/decoder_prune.py`、`v6*`、`v7*`、`pseudo_gt`、`12h/csd_model.py` 等）被 src 依赖，**不要移动/重构**——清单见 `tools/README.md`。
- `tools/legacy_root/` = 论文期根目录平铺模块（model.py/trainer.py/run.py…）+ 一次性脚本；自举包，flat import 仍可用；根 `__init__.py`/`__main__.py` 是兼容 shim。
- `tools/legacy_utils/` = 原 `utils/` 包，全仓以 `tools.legacy_utils.*` 引用。
- **`paths.py` 在仓库根，位置定义 `WEAVE_ROOT`，不要移动。**
- `archives/` = 冷存（_tmp、EXPERIMENT_ARCHAEOLOGY、agent 工作区、exp 历史输出等；gitignore）。
- 文档入口：`docs/README.md`（索引）→ `docs/HIGHLIGHTS.md`（两大进展：S10 decoder、Q2A+Oracle）→ 阶段目录。根目录 README/REPRODUCE 是论文口径，与当前研究线并存。
- 无关项目 `foi26/`（复旦校赛）已移出仓库至 `G:\foi26\`，与本仓库无关。

## 当前进行中的工作

- **V9-LD 已收尾**（2026-11-14）：LD0–LD4 latent 消融结果见 `docs/v9/V9_LD_RESULTS.md`——LD4（mask+全方向）是唯一净胜臂（test580 Top-1 .3556）；代码 `exp/v9_local/`，产物 `runs/v9_local_4070/`（untracked，大文件不入 git）。
- 3060 被他人占用做 StyleGallery 复现，**不要碰**。

## 环境备忘

- 本地 GPU：RTX 4070 Laptop 8GB（实际可用 ~6GB）；评测解码/指标需分进程，勿同时驻留多模型。
- 评测指标管线：`src/eval/measure.py`（DINOv2-small/CSD-ViT-L/CLIP/LPIPS，冻结 referee）。
- tasks：`eval_protocol/tasks_750.json`（750=Cross600+Same150，5 画派）；V9 dev/test 按 md5(source_stem)%5 划分（dev=170/test=580，互补并集=750）。
- 已知偏差：本地曾用 VAE fallback 解码（~8.6/255 MAE vs TRT），**现已全部改 S10**；S10 vs 原版 decoder 的差异是协议级选择，不是误差。
