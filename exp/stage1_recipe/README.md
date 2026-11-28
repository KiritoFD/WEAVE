# CLEAVE stage-one recipe study

Stage 1 deployment status (2026-11-28): code and plan are deployed in the existing NVMe workspace `I:\Latent_Style\WEAVE`. Remote compile and plan mode pass. Asset preflight is `ready=false`: P2, V36 cache/memmaps, VGG16 and ConvNeXt weights are missing. E4500, P1 and SD0 files exist, but P1 size and SD0 hash need verification. Stage 1 has not started.

```powershell
# CPU-only: no CUDA query, no model construction, no worker
python -m exp.stage1_recipe.queue --config docs/system/optimization/training/stage1_config.json
python -m exp.stage1_recipe.evaluation.audit --config docs/system/optimization/training/stage1_config.json --out runs/stage1_recipe/preparation/s0_cpu_audit.json
python -m exp.stage1_recipe.queue --config docs/system/optimization/training/stage1_config.json --preflight
```

在 `I:\Latent_Style\WEAVE` 执行 `powershell -ExecutionPolicy Bypass -File exp/stage1_recipe/deploy_3060.ps1 -Mode Audit`。当前审计列出缺失的P2、V36 cache/memmap和离线VGG/ConvNeXt权重；补齐后重跑Audit，只有CPU checkpoint/cache检查通过才算达到S0起跑条件。执行入口还会独立检查3060空闲状态、获取锁并冻结16h绝对deadline。

| 代码 | 职责 |
|---|---|
| `config.py` | CPU配置、11阶段预算、路径与基本契约 |
| `queue.py` | 严格串行、独占/存活、阶段截止、Seen-Dev选择、R0/I0/F0 |
| `training/e_objectives.py` | 固定代理缓存、A/B/C输入契约与B画法倍率 |
| `training/reference_e.py` | E双EMA、梯度诊断、profile、matched/equal-time与完整恢复 |
| `training/oneshot_p.py` | P分组LR、同步batch96、双EMA、完整恢复 |
| `training/profile.py` | future GPU独立worker吞吐/显存标定；少于600更新拒绝全矩阵 |
| `evaluation/audit.py` | 默认CPU来源/严格Fast5门控；显式已生成RGB的LPIPS补算 |
| `evaluation/generate.py` | 分进程生成/裁判、明确EMA、输出签名及完整任务检查 |
| `evaluation/selection.py` | 训练侧筛选、括区插值、源图配对CI和采用门 |
| `evaluation/probes.py` | 固定Seen-Dev同色盘/灰度响应及少量样例；不自动证明质量 |
| `evaluation/final.py` | 完整正式汇总、采用判定、各数据集Pareto导出 |

主要输出：`manifest.json`、`configs/`、`workers/`、`stages/`、`training/<arm>/{latest,matched,equal_time}.pt`、`screens/`、`r0_plan.json`、`final_selection.json`、`evaluations/`、`formal_index.json`、`final_report.json`、`RESULTS.md`、`pareto_points.json`及矢量/交互图。恢复同命令不会扩展deadline；第二种子和新阶段延长写入不同run，不混淆证据。

LPIPS补算示例（需先具备完整旧输出；默认CPU实际LPIPS也会加载权重，不属于纯文件审计）：

```powershell
python -m exp.stage1_recipe.evaluation.audit --repair-lpips --repair-target all --out runs/stage1_recipe/lpips_repair --device cpu
```

若需从其他既有RGB目录补算，传`--repair-spec`；清单schema由`audit.py`校验。缺资产会在加载网络前报错，不生成替代图。

正式图复现入口（只在确有完整结果后运行）：

```powershell
python docs/performance/plot_cleave_pareto.py --stage1-results runs/stage1_recipe/pareto_points.json
```

当前评测helper绑定规范V36缓存目录，`cache_root`不得悄悄指到另一套先验；非规范目录明确拒绝。实际运行前应保留依赖的V29/V32源码和已验证离线环境；CPU准备不保证远端环境已完整。
