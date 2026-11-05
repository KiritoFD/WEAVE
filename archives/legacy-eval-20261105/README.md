# legacy-eval-20261105

**不可用于比较 / NOT COMPARABLE.**

这些脚本产生过 DINO-S/CLIP-S/LPIPS 数字，但它们的编码器、分辨率、参考图策略或
两段式流程与 canonical 协议不一致。迁移前已确认没有正式产物或主链路依赖它们
（仅文档与 `tests/test_no_absolute_paths.py` 的路径白名单提及）。

Canonical 协议唯一入口：

```
bash tools/12h/eval_path_trt.sh <tag> <ckpt> <out_dir> 8
```

常量集中在仓库根 `eval_protocol.py`；锚点复跑（`epoch_0010.pt`，750 对）
得到 dino_s=0.4902256206522385（与历史 0.49023 逐位一致）。

| 文件 | 原用途 | 为何不可比 |
|---|---|---|
| `_repro_dino_eval.py` | 两段式 DINO 复算驱动（硬编码 `I:\` 路径） | 调 `utils/compute_dino_metrics.py`，参考图策略/流程与 `eval_fast` 不同 |
| `eval_other5.py` | 生成后二次调 DINO 脚本 | 同上，且不带 `protocol/backbone/reference_policy` 声明 |
| `eval_other5_dup.py` | `scripts/scripts/` 下的陈旧副本 | 与 `eval_other5.py` 重复且内容可能漂移 |
| `run_dino_on_pack_ablations.py` | 对打包消融目录批量跑 DINO | 输入不是 canonical 750 划分 |

仍保留但**不属于** canonical 链路：

- `utils/compute_dino_metrics.py`：被 `v4/` 冻结诊断脚本 `import load_dino`，作为特征加载库保留；它本身不是评测入口（无 `--image-size`）。
- `utils/run_evaluation.py`：通用评测器，被 `tools/pack_subpack.py`、`tools/prepare_submission.py` 引用（提交打包依赖）；其 DINO/指标数字不得与 canonical 口径混用。
