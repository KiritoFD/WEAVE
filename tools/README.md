# tools/ — 历史代码归档 + 钉住依赖

> 2026-11-14 重组约定：**src/ 是唯一的项目门面层**；tools/ 是历史实验代码的归档地。
> 但其中一部分子包被 src/exp **活跃依赖**，属于"钉住"状态——不要移动、不要重构，
> 除非先读完 `src/` 对应 shim 的 import 链。

## 钉住子包（load-bearing，src/exp 运行时依赖）

| 子包 | 被谁依赖 | 内容 |
|---|---|---|
| `vae_suite/decoder_prune.py` | `src.utils.load_s10_decoder` | **S10 剪枝 decoder**（StudentDecoder/build_decoder）；同文件还有 12h 训练脚本（3060 硬编码路径，冻结出处） |
| `v6/models_v6.py` | `src.model.unet` | C3UNet / ConditionalScoreNet |
| `v6/canonicalize.py` | `src.model.canon`, `v7_common` | ContentCanonicalizer (C1/C2) |
| `v6_next/probe_style.py` | `src.model.critic` | LatentStyleNet（R1 critic） |
| `v7/models_v7.py` | `src.model.transport` | SingleStepTransportNet |
| `v7/v7_common.py` | `v7r/v7r_common` | checkpoint 加载器 / B4_CKPT、R1_CRITIC 路径常量 |
| `v7r/v7r_common.py` | `exp.v8.*` | SD_POOL、V7-R 共享 helpers |
| `v6_next/r2_eval.py` | `src.utils.features` | R2 750 任务评测管线（Windows 上经 `src/utils/_local_eval_refs.py` 兜底） |
| `v5/route_c.py` | `src.utils.vae` | VAE/TRT 加载与 decode（TRT 引擎部署路径） |
| `pseudo_gt/matching.py` | `_local_eval_refs`, `v6/canonicalize` | DINOv2 加载/编码 |
| `12h/csd_model.py` | `_local_eval_refs`（flat import） | CSD-ViT-L referee |
| `evaluation_v2/features.py` | `tools.sbflow` | sbflow 评测特征 |
| `eval_fast.py` | `v5/route_c` | MetricEngine/协议构建 |

其它 v6_24h、v6_next（w_guided/gate 等）经上述模块间接被拉入。

## 归档区（不再保证可运行）

| 目录 | 内容 |
|---|---|
| `legacy_root/` | **论文快照期的根目录平铺模块**（model.py/trainer.py/run.py/config_schema.py/wavelet.py 等 22 个 + 一次性脚本）。自举 `__init__.py` 使 flat import 继续可用；`python -m weave` 经根目录 shim 仍可跑 |
| `legacy_utils/` | 原 `utils/` 包（inference/run_evaluation/dataset/training…）。全仓以 `tools.legacy_utils.*` 引用 |
| `legacy_v2/`、`legacy_v4/` | v2/v4 时代的模型代次（`python -m tools.legacy_v2.train`） |
| `sbflow/` | 独立 Schrödinger-bridge 子项目（`python -m tools.sbflow.*`，内部 import 已改为 tools.sbflow.*） |
| `oneoff/`、`viz/`、`configs_retired/` | 一次性脚本 / HTML 可视化 / 退役配置 |
| `v9/`、`trt/`、`local_reader/`、`art_vae/` 等其余 | 各历史实验阶段的支撑代码 |

## 2026-11-14 import 变更摘要

- `utils.X` → `tools.legacy_utils.X`（约 50 个文件，sed 统一替换）
- 根目录平铺模块引用 → `tools.legacy_root.X`（tools/ 与 tests/ 内；legacy_root 内部保持 flat + 自举）
- `sbflow.*` → `tools.sbflow.*`，`parents[1]`→`parents[2]`（tests 内 `parents[2]`→`parents[3]`）
- `paths.py` 保持在仓库根（其位置定义 `WEAVE_ROOT`），未移动
