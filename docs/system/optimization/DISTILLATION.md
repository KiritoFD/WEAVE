# A0 / P-Core 计算量优化：串行执行与验收

> **2026-11-28 结项归档**：全部阶段已执行完毕（`status = complete`）。**A0 TRT FP16 (`1.362×`, E B4 `16.35 img/s`) 6/6 无损通过并正式采用**；`enc20` 与 `P3 CFG` 蒸馏均未通过完整 `test750` 非劣门控，**正式关闭 Encoder 压缩与 CFG 蒸馏训练**。完整定量数据、95% CI 与速度表见 [**蒸馏实测总报告**](DISTILLATION_RESULTS.md)。

本轮按用户已批准的 P0–P4 继续推进。与此前纯 PyTorch 工程优化分开记录：这是兼容性蒸馏研究，不提前宣称质量无损。所有生成仍用固定 S10B；不覆盖原 A0、E-Core、P-Core 或 S10 权重。

## 当前进度（2026-11-27）

- 原 Optimized 的完整 test750 证据已经存在；代表点 E λ=.60 的 Cross600 Top-1=.3717、DINO-C=.7854、DINO-S=.4370；P s=1.00 为 .6850、.5555、.5572。
- P0/P1 报告已回传；A0/S10 FP16、full/mixed INT8 与两档剪枝速度已测。E-Core 导出通道与 TAESD 下载问题已修复并完成补测；它们仍是探针，不代替完整质量验收。
- P2–P4 已实现；3060 七阶段短运行已通过：两档 Encoder/CFG 训练与权重重载、CFG 实际 7 次单分支调用、TRT B4 引擎的 2 图尾批处理。短运行不作质量或收敛证据。19.23M 短训练峰值显存约 7.98GB。
- 本地 11 项相关测试通过。实际运行进度以远端 `runs/v35_3060_opt/distillation/status.json` 和各阶段日志为准。

### 最新核验（2026-11-28 06:59 CST）

两档 Encoder 与 CFG 训练已完成；CFG complete=12,000 steps、4877s。CFG 教师缓存完整 512 train +64 calib，1,152 个 rollouts 文件非空，A0 cache SHA256 与来源一致；每条轨迹实际 7 次教师调用。test750 测速已完成 Optimized/TRT E/P 的 B1、B4。暖机驻留 host RGB→host RGB 口径下：Optimized E B1/B4=112.56/83.30 ms/img，TRT E=67.25/61.15（B4 1.36×）；Optimized P B1/B4=223.18/184.53，TRT P=177.40/160.87（B4 1.146×，稍低于 1.15 的采用线，待质量结果和汇总报告核对）。测量样本是实测 30–149 次，文件 IO 排除；Windows 报告设备字符串为 `NVIDIA Graphics Device`，非精确型号枚举。TRT P 的速度结果是 P4 中只替换 A0/S10 的候选，不等价于早期 P0 全链路 hybrid TRT。

CFG 的训练 batch loss 有波动，不单凭 loss 判断收敛；到 step600 的 128 条训练侧校准 velocity relative RMSE 从初始 .30456 降至 .24892、cosine 从 .95334 升至 .97104。完整训练虽按12,000步完成，但 best checkpoint 与学生自回归 rollout 的 test750 质量仍需 P4 判定。

P4 于 06:47 的首次启动在 `generate:trt:e` 失败：Windows SYSTEM 的 `Path.read_text()` 默认 ANSI 解码 UTF-8 test750 源清单，重音作者名被损坏。清单包含的 150 张源图随后逐条确认均存在。已在 `acceptance.py` 改用统一 `read_json` UTF-8 读取；新增回归测试把默认解码模拟为 GBK，确认重音路径仍正确，测试通过（1 passed）。修复已同步远端，原队列于 06:57 从失败阶段恢复；06:59 日志确认 `trt_e_l050` 已实际开始写分片，92/750 tasks。该臂还需完成至750，再完成另一 E 档。旧错误仍保留在追加日志中，按新启动时间识别即可。

已将同一自动复查设为 **07:20 CST**，根据恢复后实测生成吞吐重算 ETA；目前未宣称任何候选通过 test750。证据在 `runs/v35_3060_opt/p4_snapshot.json`、`missing_audit.json` 与 `eta_check.json`。

### 后台运行与 ETA 检查（2026-11-27 23:33 CST 核验）

实际运行机器为 **3060 Windows**，由计划任务 `WEAVE-Distill-P0-P4` 以 SYSTEM 身份托管，**不是 tmux**。断开 SSH 后任务继续；当前队列处于训练数据审计/缓存 `cache_encoder`（3/36），训练尚未开始。日志追加保留旧错误，必须结合本次启动时间判断。

P0 曾于 21:42 在 PyTorch 编译处失败：venv 继承系统 Triton，但 TinyCC 默认查找错误地落在 venv 的 platlib。22:42 已恢复，22:52 完成 P0，修复已通过实际编译；P1 于 23:07 完成。入口按已安装 Triton 的实际路径定位编译器。另修复 Windows 上 `os.kill(pid, 0)` 会终止进程的问题，改用只读进程句柄查询；新增存活检查测试已通过。

23:05、23:30 自动复查已执行，下一次为 **11 月 28 日 00:00 CST**。当前仍处于 CPU 图像身份审计：训练原图根目录共 79,433 张 JPG（约 39.1GB），其中存在 1.3–1.45 亿像素图片；这些是扫描候选，最终仍只选 8,192+128 张。worker 39844 的累计读取量于 23:32–23:33 从 3.689GB 增至 3.829GB，CPU 时间同步增加，确认进程在推进；未见异常退出。当前审计无逐图进度日志，因此尚不能精确估时，不把低 GPU 利用率误判成卡死。训练预算为两档 Encoder 各至多 2h、CFG 至多 4h，缓存/编译/评测另计。调度记录在本机 `runs/v35_3060_opt/eta_check.json`；阶段完成或失败时检查产物和串行交接，不另起并行 GPU 工作。

E-Core 导出已改为直接取真实源图的 conditioning 张量（`c64` 实为 4 通道），HF 镜像端点已移到导入 Hugging Face 前设置。当前缓存进程早于补丁启动，因此补测由 `finalize` 前的独立子进程兜底执行；若此前重启，缓存阶段入口也会先补测。`*_before_recheck.json` 留存原报告，`stages/*_recheck.json` 防止重复。原 P0 hybrid 含 P-Core TRT，P4 的 `trt` 候选仅替换 A0/S10，二者不可混称。

## 串行顺序

| 阶段 | 实现 | 数据与产物 |
|---|---|---|
| P0 | TRT FP16、INT8 敏感度、真实编辑 latent 响应 | `p0_trt_report.json`；INT8 仅探针 |
| P1 | TAESD 规范转换探针、两档 A0 剪枝速度 | `p1_report.json`；下载不可用单独标记 |
| P2 | 原权重 Taylor 排序剪枝；Latent + Haar + 固定 S10 一致性蒸馏 | 每档至多 2h；`enc20/enc10/{last,best}.pt` |
| P3 | 四分支教师真实 rollout；接收连续 guidance 的单分支学生 | 至多 4h；`cfg/{last,best}.pt` |
| P4 | 各档完整 test750、错目标控制、B1/B4 实测、来源聚类 bootstrap | `p4/*_quality.json`、`p4/speed/*.json` |

每个训练、生成、裁判进程退出后才开始下一进程。任一必需阶段失败，队列立即停止，保留日志及 checkpoint。组件分别过完整 CI 门控后，才运行组合后端验证。固定时间上限是预算，不是预计收敛时间；缓存、编译和评测另计。

两档名称是实验标签，实参分别为 **19.230M**（96/192/384/384）与 **11.264M**（64/128/256/320），不是精确的 20M/10M。

## 数据和目标

入口为 [串行队列](../../../exp/deploy_v35/distill/run.py)。训练根目录必须显式传入，源码中不写机器绝对路径。

```powershell
python -m exp.deploy_v35.distill.run --train-root <训练图目录> --out runs/v35_3060_opt/distill_smoke --smoke
python -m exp.deploy_v35.distill.run --train-root <训练图目录> --out runs/v35_3060_opt/distillation
python -m exp.deploy_v35.distill.run --config runs/v35_3060_opt/distillation/config.json
```

默认选 8,192 张训练图与 128 张训练侧校准图。按解码后的 512px RGB 精确 hash、dHash 和缩略图误差剔除测试作品的精确/近重复，再做独立校准划分。此审计不等价于 Artist-Disjoint。分片缓存保留图片、原 A0 BF16 的 scaled mean latent 与来源清单；不使用测试图训练或选 checkpoint。

Taylor 排序用固定 S10 的重建误差提供非零重要性信号，仅用于初始化排序。学生训练目标始终是复现教师坐标及教师 S10 响应，不优化一套新的 VAE 重建坐标。S10 输入恰好除以 .18215，checkpoint PQC 恰好应用一次。

CFG 缓存覆盖 s=.60/.85/1.00/1.50，保留教师首步 midpoint 的中间状态；6 步实际采集 7 次四分支速度场。学生接收每次实际 ws/wc/wsp、rescale phi、gamma、layout 参数，直接预测已合成并 rescale 的速度。学生仍执行同一积分器，实际 7 次单条件评估。离线教师状态拟合不保证学生自己的 rollout 稳定，必须通过 P4。

## 质量与采用

校准集只选一次 Encoder checkpoint；随后冻结候选 hash、test750 协议和噪声。P4 复用已有同协议 Optimized 基线，生成器退出后另进程加载裁判。Cross600、Same150、逐画派、source-lock 与错目标控制均保留。

非劣要求：来源聚类 bootstrap 95% CI 的 C/S 下界 ≥−.005、Top-1/Same150 下界 ≥−2pp、LPIPS 上界 ≤+.005。均值达标而 CI 不足时记为 inconclusive。当前实现修正了旧门控仅检查均值的问题。采用还要求同机同口径 B4 端到端加速至少 15%。

测速是常驻模型、warm、host RGB→host RGB（含 H2D/D2H，不含文件编解码）；不能称 process cold。B1 延迟与 B4 吞吐分别保存，不能把 batch 均摊时间当单图响应时间。

完结后生成 `runs/v35_3060_opt/distillation/REPORT.md` 和 `docs/performance/data/3060_distillation_summary.json`。完整实测点由 [统一 Pareto 绘图入口](../../performance/plot_cleave_pareto.py) 合入 SVG/HTML；质量通过的内部点可参与内容—风格前沿，失败点保留为诊断标记，3060 与 4090 的速度不混合计算前沿。
