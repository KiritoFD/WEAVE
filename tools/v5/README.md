# WEAVE V5 执行入口

`M:\WEAVE\docs\116plan.txt`（已同步到本仓库 [docs/116plan.txt](../docs/116plan.txt)）定义 24 小时、三路线、串行决策树。所有 GPU 实验在 `ssh 3060` 的 `I:\Latent_Style\WEAVE`（WSL 路径 `/mnt/i/Latent_Style/WEAVE`）执行；git 仍在本地维护。

## 已执行 / 正在执行

| 阶段 | 脚本 | 产物 | 裁决 |
|---|---|---|---|
| P0 条件泄漏修复、验证集冻结、真实 D5 多尺度描述缓存 | `p0_prep.py`, `p0_condition_swap.py` | `runs/v5/p0_report.json`, `runs/v5/cache/*` | 条件只给 Style-ID；旧权重对 ID 几乎不敏感 |
| P1 Latent Oracle O1/O2/O3 | `p1_oracle.py` | `runs/v5/p1_oracle/oracle_matrix_report.json` | 全部失败；无约束也无法提高 CSD ΔT |
| B1 Latent Style Encoder + Oracle 重测 | `b1_latent_style_encoder.py` | `runs/v5/b1_style_encoder/b1_report.json`, `latent_style_encoder.pt` | 验证分类 87.7%，但重测 ΔT 仍 <+0.01；切 Route C |
| Route C 真实分布 RF 先验 | `route_c.py` + `route_c_pipeline.sh` | `runs/v5/route_c/*`, `final_report.json` | 串行：训练 → 纯噪声检查 → 低/中噪声补救 → 200 扫描 → 750 双 seed → 总结 |
| 被剪枝分支实现 | `branches/route_a.py` + `branches/run_pruned_branches.sh` | `runs/v5/branches/*` | M1/M2/B2/解蒸馏均实现并 smoke test；本轮按决策树不启动 |

## 串行启动（当前已由 tmux `v5c` 运行）

```bash
cd /mnt/i/Latent_Style/WEAVE
bash tools/v5/route_c_pipeline.sh
```

冻结副本：`runs/v5/route_c/route_c_pipeline_frozen.sh`，避免运行中改写 bash 造成字节偏移。训练使用 15,545 张真实 D5 latent 常驻 GPU、2.28M 参数、有效 batch 256、峰值约 11.4 GiB，满足「尽量吃满 10G+ 显存」。

## 本地回传与报告

```bash
python _tmp/fetch_v5_results.py
python _tmp/v5_wait_and_fetch.py   # 等 V5 ROUTE C ALL DONE 后自动回传
```

`tools/v5/summarize_v5.py` 在远端生成 `docs/v5/V5_116PLAN_EXECUTION_REPORT.md`，回传后即为 24h 交付报告。

## 剪枝分支（仅在对应 gate 通过时手动启动）

```bash
bash tools/v5/branches/run_pruned_branches.sh
```

- `train --mode m1`: 直接残差风格算子，零初始化恒等。
- `train --mode m2`: 无配对终点监督可微 ODE。
- `b2`: `local selfsim` vs `coarse geometry` 内容约束 Pareto。
- `distill`: Oracle 优化解 → M1 摊销蒸馏。
