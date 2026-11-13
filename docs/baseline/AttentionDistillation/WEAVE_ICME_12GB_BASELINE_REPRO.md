# Attention Distillation (AD)：RTX 3060 12GB 复现执行方案

> 审计日期：2026-11-13；状态：**代码路径已核查（webui/runner.py → ADPipeline.optimize），未在真实 3060 上运行**。
> 继承 `WEAVE_ICME_12GB_BASELINE_REPRO.md` 的 A/B/C 三层判定：A 源码可得 ✅；B 架构上无官方 12GB 证据，需适配；C 必须实机 smoke。

## 0. 结论先行

| 项 | 状态 |
|---|---|
| 代码 | ✅ https://github.com/xugao97/AttentionDistillation （CVPR 2025） |
| 权重 | 仅需 SD1.5（diffusers 布局），无额外训练权重 |
| 官方 12GB 证据 | **无**。UI 默认 `num_steps=200, lr=0.05, bf16`，`enable_gradient_checkpoint` 官方 False（本仓库附带脚本改 True） |
| 定位 | **P1 备选**：先做 12GB 可行性 smoke，过 Gate 才进 10/50/750 |

## 1. 与 StyleGallery 的差异

- AD 直接优化 SD1.5 UNet attention 特征做风格迁移，**没有聚类/DINO 匹配阶段**——预处理更轻，但 200 步 latent 优化的反传图同样可观。
- 单参考图输入；正式比较沿用 `prepare_weave750_sg_manifest.py` 生成的同一份 manifest（train-only、seed 42 冻结参考）。
- 速度口径与 SG 相同：批量脚本逐图冷启动，`cold_wall_s` 不可与 warm 推理横比。

## 2. 环境与权重

```bash
bash ~/weave_baselines/setup_attention_distillation_12g.sh \
     ~/AttentionDistillation ~/weave_baselines/sg12env \
     --sd15 /path/to/StyleGallery/pretrained_models/runwayml_stable-diffusion-v1-5
conda activate ad12   # 或 source ~/weave_baselines/ad12env/bin/activate
```

- SD1.5 优先**复用 StyleGallery 已下载的 diffusers 目录**（省 ~4.3GB）；不存在时脚本自动 `hf download`。
- `HF_ENDPOINT=https://hf-mirror.com` 可用于国内网络。

## 3. Step 0：12GB smoke（必需）

```bash
python ~/weave_baselines/run_ad_smoke_12g.py \
  --repo ~/AttentionDistillation \
  --content /abs/source.jpg --style /abs/ref.jpg \
  --model pretrained_models/stable-diffusion-v1-5 \
  --steps 20 --precision bf16
```

注意：**20 步只是显存验证，不是论文默认**。论文/UI 默认 200 步；若 200 步 OOM 只能记 `SG reduced compute` 类似的受限配置，不能宣称原版复现。

## 4. 全量（smoke 通过后）

```bash
python ~/weave_baselines/batch_ad_12g.py \
  --repo ~/AttentionDistillation \
  --manifest ~/sg750_manifest.csv \
  --out ~/ad750_sd15 --steps 200 --precision bf16
```

逐图子进程、失败即停、断点续跑（`<index>_<hash>.png` 存在则跳过），provenance 写 `execution.jsonl`。

## 5. 风险与降级路径

- 若 200 步 bf16 在 12GB OOM：依次试 `enable_gradient_checkpoint=True`（官方 runner 有开关）、`--precision fp16`、最后才考虑步数削减并**明确标注 reduced**。
- 若 API 与本脚本假设不符：以 clone 时固定的 commit 为准修 `run_ad_smoke_12g.py` 的导入路径，改动范围记入 provenance。
