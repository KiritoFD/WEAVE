# Hist2Style 复现执行记录（WEAVE / ICME baseline）

- 论文：Hist2Style: Histogram-Guided Stylization with Bilateral Grids, CVPR 2026（Adobe Research + UC Berkeley）
- 仓库：https://github.com/dgalor/hist2style （2026-11-14 经 ghfast.top 镜像克隆到 3060 `/mnt/i/Latent_Style/hist2style`）
- Checkpoint：仓库自带 `demo/model-epoch=1500-val_loss=2.79.ckpt`（5.9MB，1.5M 参数），**无需训练、无需额外下载**
- 复现机器：3060（与其他 baseline 同机同时序口径）；模型极小，与 StyleGallery 批任务共享 GPU 无冲突

## 定位（写论文时注意）

Hist2Style 是**颜色直方图引导的局部仿射调色**模型（Bilateral Grid 预测局部 LUT），
表达能力上限是"空间自适应颜色编辑"，不改变笔触/结构。它是**轻量对照**，
用于分离"颜色匹配"与"笔触/结构迁移"的贡献（Histogram / Hist2Style / WEAVE 消融），
不替代 StyleGallery 等艺术风格迁移主 baseline。

输入信息量差异必须注明：Hist2Style 用 **1 张参考图**（Y'CbCr 直方图）；
WEAVE 用 Style-ID 条件。两者条件通道不同，不是纯算法对比。

## 环境

- 系统 python3（torch 2.10+cu128 / torchvision 0.25 / diffusers 0.41），未建独立 venv
- 额外依赖：`pip install lightning==2.6.4`（`demo/model.py` 继承 `pl.LightningModule`，仅加载权重用）
- kornia/omegaconf/lpips **不需要**（仅训练/评测脚本用）

## Smoke 验证（2026-11-14）

```
python3 run.py --content <test图> --style <train参考图> --output-dir outputs/h2s_smoke
→ 1/1 OK，19.1s wall（含模型加载），输出 PNG 正常
```

## 750 全量

- manifest：复用 `StyleGallery_src/sg750_manifest.csv`（750 任务，source=`data/test/`，
  refs=`datasets/distinct5_512/train_512clean/` 每画派固定 1 张、seed 42、train-only 无泄漏）
- runner：`batch_h2s_750.py` —— 直接调用官方 `run.py` 的 `load_model / style_histogram /
  stylize / configure_performance`，仅把成对目录枚举换成 manifest 迭代；模型常驻、
  风格直方图按路径缓存（5 张参考图只算 5 次）
- 输出：`h2s750_outputs/{index}_{hash}.png` + `execution.jsonl`（命名规则与 SG 一致，
  可跨 baseline 对齐）；断点续跑，失败记 `*.error.log` 继续
- 运行：tmux `h2s750`，`until DONE; do batch_h2s_750.py --max-tasks 250; done`

## 测速口径（重要）

| 口径 | 单图耗时 | 说明 |
|---|---:|---|
| 官方 `run.py` 单次调用 | ~19s/张（含 ~0.6s 权重加载+首次 cudnn benchmark，实测 warm 后 ~1s 推理） | 论文引用的保守口径 |
| `batch_h2s_750.py` 常驻 | **~0.8–1.1s/张**（首任务 6.4s warmup） | 我方工程优化，单独记录 |

论文表格建议报官方单次调用口径；常驻 runner 仅用于快速产出 750 结果。

## 备注

- `eval.py`（EvaluationSet 2584 对基准）依赖下载外部数据集与 FID inception 权重，
  本次复现目标是 WEAVE Test750，不跑 eval.py。
- `train.py` 需 ~54GB 训练数据，仅当需要验证训练复现时才考虑（本任务不需要）。
