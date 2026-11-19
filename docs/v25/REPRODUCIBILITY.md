# V25 最小复现手册（DOC-01，2026-11-20）

> **任务**：[agent2.md §八](../19_pull_together/agent2.md) A09/B09/DOC-01 交付物之三。覆盖：训练（Round 2 / 机制 A / 机制 B）、评测（run_measure 管线）、教师生产（reference_teacher_v2 全部开关）、导出推理（双接口 B05 状态）、4090/3060 启动手册、checkpoint manifest。
> **统一口径**：全部质量读数 = `eval_protocol/tasks_750.json`（Cross600 读数 + Same150 恒等控制）× `S10` 解码（`deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt`，**不得绕过 `src.utils.decode_s10`**）× 进程隔离四裁判。任何"删组件"式归因必须走 A01 配对 CI（[audits/a01_round1_ci.md](audits/a01_round1_ci.md)）。
> **命令均在仓库根执行**（`paths.py` 定义 `WEAVE_ROOT`，勿移动）；远程根目录：4090 = `/root/Workspace/WEAVE`（Linux+tmux），3060 = 仓库 checkout（Windows+schtasks）。

## 一、训练复现

### 1.1 共享前置（机制 A/B 与 Round 2 评测前必跑一次）

```bash
# 4090；产出 runs/v25/cache/train_latents_4400.pt（577MB，含 teacher_spec）
python -m exp.v25.precompute_train_latents --pool ../pseudo_gt_24h/orig_mean_fp16
```

教师三元组全程锁定（[GPU4090_RESULTS.md](GPU4090_RESULTS.md) 共享前置）：`ckpt=slf_core_v2.pt + anchor=B4(canon(z0,c1_gentle)) + solver=D6H + branches=cs_j + gate=original`；`z_p1_tr` 来自 `s2_prism1_v2 λ=1.0`。
**血缘警告**（[audits/a08_science_review.md](audits/a08_science_review.md) §二）：训练教师锚 = B4，评测主配置锚 = T4，两者 distinct-scale——跨锚读数禁止互相归因；补救重评用 [../../exp/v25/r12_anchor_reeval.py](../../exp/v25/r12_anchor_reeval.py)。

### 1.2 Round 2 配方消融（r2_train_ablation）

```bash
python -m exp.v25.r2_train_ablation \
  --pool ../pseudo_gt_24h/orig_mean_fp16 --device cuda \
  --updates 1500 --seed 20263500 \
  --arms Tr_T0,Tr_T1,Tr_T2,Tr_T3
```

- 配方↔分支规则内置（T0 无 drop_c/AB、T1 drop_c 30%、T2 AB 25%、T3 两者）；评测只走 cs_j（[../../exp/v25/r2_train_ablation.py](../../exp/v25/r2_train_ablation.py) `--arms/--updates/--seed`）。
- 产物：`runs/v25/r2_tmatrix/slf_Tr_T*.pt` + `runs/v25/r2_train_ablation.json`；评测种子 20261119（与 R1 的 20261185 不同——跨 Round 配对 CI 须同种子重跑一侧，见账本 U9）。
- **R4 型覆盖风险**（[audits/b06_recovery_report.md](audits/b06_recovery_report.md) §1.3）：`--arms` 子集重启会整体覆盖结果 JSON，永久丢先前臂记录——重启必须用完整臂列表（幂等收敛）。

### 1.3 机制 A（空间门控，判负档案）

```bash
python -m exp.v25.mechanism_a --arm all --gamma 0.55 \
  --updates 1000 --cal-updates 300 --lam-c -1 \
  --pool ../pseudo_gt_24h/orig_mean_fp16 --device cuda
```

- `--arm all` = A0_global / A1_heuristic(κ 网格 `--kappa 0.5,1.0`) / A2_learned / A3_smooth 顺序训练+评测；`--lam-c -1` 触发 300-update 自动标定（实测探针 Δ=+0.037，λ_c=0.3）。
- 产物：`runs/v25/mechanism_a/`（门控 EMA 权重 `gate_*.pt` + `mechanism_a.json` 含逐臂 edit_energy）。
- 训练无 ckpt 复用（B06 R5 型：重启即全量重训）；判负定案与限制见 [ABLATION_LEDGER.md](ABLATION_LEDGER.md) F2/§四。

### 1.4 机制 B（学生蒸馏，B0 判负-修正中）

```bash
# B0 = PRISM-1 热启动 + γ 残差分支 361K；默认 updates B0=3000 / B1=6000（--updates 覆盖）
python -m exp.v25.mechanism_b --scale B0 \
  --gammas 0,0.25,0.33,0.40,0.55,0.63,0.70 \
  --updates 3000 --pool ../pseudo_gt_24h/orig_mean_fp16 --device cuda
```

- **重试协议**（已记录，未执行）：≥3000 updates + 扫测施加 same-task bypass（旧 :382 缺 bypass 的 Same150 崩塌是伪迹）+ lr 降档；先跑 R2 修复重评（R12）再谈重训。
- γ=0 精确回退回归断言内置于脚本（residual==0 逐位，构造性回退）。

## 二、评测（run_measure 管线）

标准链：**实验脚本生成 `runs/v25/raw/<arm>/`（RawWriter 分片 + meta.json）→ 进程隔离裁判 `src.eval.measure` → `measured/<arm>/eval.json` + `summary_arm.json`**。

```bash
# 方式一：runner 封装（评测侧默认；不存在 eval.json 时才跑 measure）
python -c "from pathlib import Path; from src.eval.runner import run_measure_subprocess; \
  run_measure_subprocess(Path('runs/v25/raw/<arm>'), Path('runs/v25/measured/<arm>'), arm_name='<arm>')"

# 方式二：直接调裁判（write_png=False 即 --no-png 正式口径）
python -m src.eval.measure --raw runs/v25/raw/<arm> --out runs/v25/measured/<arm> \
  --device cuda --micro-batch 8 --no-png
```

- 校验：eval.json 必须 `n_total=750 / n_cross=600 / n_same=150`（否则 `summarize_measured_arm` 抛错，[../../src/eval/runner.py](../../src/eval/runner.py)）。
- 已知半完成窗口（B06）：meta 有效但 measure 被 kill → 重启**不会自动补跑 measure**（H1，崩溃于读 summary）；eval.json 半写需手工删除后补跑（H2）。分诊：

```bash
python -m src.v25.result_writer runs/v25 --stale-min 30        # flags: orphan_raw / summary_missing / ...
```

- 统计复算（全部 CI 数字）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  python -m tools.v25_cpu.a01_noninferiority     # n_boot=1500, seed=2026, 150 源聚类, 容忍带+守卫
```

## 三、教师生产（reference_teacher_v2 全部开关）

```bash
python -m exp.v25.reference_teacher_v2 \
  --pool ../pseudo_gt_24h/orig_mean_fp16 --device cuda \
  --content-w 1.2            # 2.2 | 1.2 | 0.7 | all（三档 w_c 对应 T-conservative/medium/expressive）
  --n-cand 2400 --n-keep 2000 \
  --steps 24 --batch 12 --n-swap 400 \
  --mid-cos-lo 0.80 --mid-cos-hi 0.92 --style-gain-min 0.10 \
  --init src --anchor-alpha 1.0 --trust-region None --color-guide 0.0 \
  --ref-cache-dir runs/v25/cache/ref_feats --force
```

- **结构性重设计三开关**（C1 诊断后新增，[../../exp/v25/reference_teacher_v2.py](../../exp/v25/reference_teacher_v2.py)）：`--init ref_anchor`（z_opt 以参考侧统计初始化，替代恒等盆地）、`--trust-region`（内容项改有界信任域）、`--color-guide <w>`（latent 空间参考色彩 fiber 引导项）；`--fast` 走批量缓存加速。
- 双指标闸门：`keep iff mid_cos ∈ [0.80,0.92] AND style_increment ≥ 0.10`（A03 §五参数块；两档 w_c=1.2/0.7 在旧目标下 keep=0，属盆地形态问题——见账本 E7/F6）。
- **缓存血缘铁律**：steps/lr 必须纳入缓存键（事故：`ref_teacher_pairs_v2_texpr.pt` 按档名复用命中 24 步旧包）；接入 [../../src/v25/artifact_registry.py](../../src/v25/artifact_registry.py)（B03）登记 `params_hash` 后再复用。
- 4090 实测速率 ≈10s/对（0.4s/步）；3060 物理 20–40s/对不可行（B02 FLOP 地板）——教师生产一律 4090。

## 四、导出推理（双接口 B05 状态）

- **接口规约已交付**：[../../src/v25/config.py](../../src/v25/config.py) —— 用户接口仅 `PRISMCore.transfer(content, style, strength=0.5)` 与 `PRISMRef.transfer(content, reference, strength=0.5)`；高级配置冻结在 `TransferConfig`/`RefConfig`，`override` 仅留研究人员。
- 引擎注册表 `ENGINE_REGISTRY`：core = `prism1` / `d6_flow` / `hybrid`（**`DEFAULT_CORE_ENGINE="hybrid"`**，B0 判负后集中切换点）/ `mecha_a`，`student` pending；ref = `o3_ref`（默认 wj=2.2 t45 canonical）。
- 强度映射只消费 [audits/a04_strength_curves.json](audits/a04_strength_curves.json)（UNMEASURED 区间 clamp；反向旋钮 τ_min/τ0 已在 spec 标注 `direction="inverse"`）；配置 hash 复用 `param_hash`（与 A02 solver_hash 同一序列化约定）；历史臂经 `restore_historical_arm` 精确重建。
- **能力边界（如实标注，未实现即 NotImplementedError）**：当前全部 available 引擎仅支持 test750 协议批形态（N=750，`capabilities=protocol_batch_750`）；单样本推理 / 显式 style 覆盖 / 任意单参考图编码路径**尚未集成**；最终单样本双接口冒烟未跑（未跑项在案）。

## 五、4090 启动手册（Linux + tmux）

1. **一律脚本文件方式**：写 `xxx.sh` 后 `tmux new -s <sess> -d 'bash xxx.sh'`——不用内联长引号命令（复合引号损坏教训）；`pkill` 模式必须用 `[x]` 括号技巧防自匹配（`pkill -f '[m]echanism_b'`）。
2. 会话链：`v25r1`（Round 1 消融）→ `v25r2q`（Round 2 自动接力）→ `v25pre`（共享前置）→ `v25mechAall`（机制 A）→ `v25tchq`（教师队列）。每臂增量写 JSON（`updated_utc` 停更 = stage 内崩溃定位）。
3. **OOM 攻坚配置**（已固化）：autocast bf16 + 分块带梯度解码（decode_sub=4）+ 候选间清理 + `expandable_segments`；注意外部服务（`xy/DiT/bundlecheck`，Qwen-VL）会间歇占用 4.6GB 级显存——启动前 `nvidia-smi` 核查。
4. python 需 `-u` 防缓冲假死；日志 `runs/v25_*.log` 单文件重定向便于 tail。
5. 评测复算/A05 导出等登记命令示例：

```bash
ssh 4090 "cd /root/Workspace/WEAVE && OMP_NUM_THREADS=2 python -m tools.v25_cpu.a02_teacher_lineage --check-b4-t4 --device cuda > runs/v25/cache/a02_b4_vs_t4.json"
ssh 4090 "cd /root/Workspace/WEAVE && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -m tools.v25_cpu.a05_feature_geometry --export-u > runs/v25/cache/a05_export_u.log"
```

## 六、3060 启动手册（Windows + bat + schtasks）

1. **bat 模板**（吸收全部教训：SSH 直启被回收、schtasks 内联 `/TR` 引号转义损坏 exit=2、httpx 网络瞬断）：

```bat
@echo off
cd /d %~dp0
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
python -u -m exp.v25.r3_oneshot_minimal --pool ..\pseudo_gt_24h\orig_mean_fp16 >> runs\v25_r3.log 2>&1
```

`schtasks /TR` 只指 bat 文件路径，绝不内联引号参数。

2. **失败识别**（[audits/b06_recovery_report.md](audits/b06_recovery_report.md) §3 检查项）：`schtasks /query /tn v25r3 /v /fo LIST` → `Last Result`：0=正常 / 2=bat 路径或转义损坏 / 267009=正在运行 / 267011=未运行过 / 0xC0000409·0xC0000005=python 运行时崩溃；配合日志 mtime 停更（>10–15 min 且进程不在 = 死亡；mtime 停但进程在 = 忘 `-u`）与 JSON `updated_utc` stage 定位。
3. 日志尾模式：`httpx.ConnectError` → 补 `HF_HUB_OFFLINE=1`；`CUDA out of memory` → 3060 12GB 微批（`src.utils.tier_micro_accum` 自动限幅 ≤6GB）；无 traceback 戛然而止 → 进程被回收。
4. 恢复顺序：`result_writer --json` 看 flags（决定 raw 重生成 or 只补 measure）→ 查 Last Result 与日志尾 → 手工修复后让队列继续（R2 复用闸保证幂等，勿重复启动已成功臂）。原 `O3_rgb_cnx_gated.pt` 为保底 fallback，不删。

## 七、Checkpoint manifest

| 权重 / 资产 | 路径 | sha256 前 16 位（已登记者） | 状态 |
|---|---|---|---|
| Round 2 Tr_T0 | `runs/v25/r2_tmatrix/` | `e348be4e014ec6d6` | 已登记（[GPU4090_RESULTS.md](GPU4090_RESULTS.md)） |
| Round 2 Tr_T1 | `runs/v25/r2_tmatrix/` | `da883c8a6060c162` | 已登记 |
| Round 2 Tr_T2 | `runs/v25/r2_tmatrix/` | `936c71e3d63761b5` | 已登记 |
| Round 2 Tr_T3 | `runs/v25/r2_tmatrix/` | `a9abd0de3ed8f7a1` | 已登记 |
| B4 锚底座 | `runs/v6/P1_B4_distill50/P1_B4_distill50.pt` | `2716436fe0ee` | 已登记（本地实测） |
| T4 锚 | `exp/v22/t4.py` T4_HASH | `04007b4f0355`（config hash，非文件 sha） | 已登记（漂移测试守卫） |
| solver 身份 | D5 `07904c2bb16b` / D6 `e14eb16ca1e9` / D6H `f52948ae3451` / DualSWD 臂 `0d88c4ae06b5` / 曲线B t066 `fa46f171706e` / P13 锚臂 `9885c0e7d0f7` | —（config hash） | 已登记（[audits/a02_teacher_lineage.md](audits/a02_teacher_lineage.md)） |
| PRISM-1 | `runs/v19/students/s2_prism1_v2.pt` | 待登记（4090 独占） | **待登记** |
| V20 SLFUNet | `runs/v20/slf_flow/slf_core_v2.pt` | 待登记（4090 独占） | **待登记** |
| V22 冠军 | `runs/v22/tmatrix/slf_T4Ssnap1500.pt` | 待登记 | **待登记** |
| V23 DualSWD | `runs/v23/dual_swd/slf_DualSWD_*.pt` | 待登记 | **待登记** |
| V24 O3 | `runs/v24/oneshot_val/O3_rgb_cnx_gated.pt` | 待登记 | **待登记** |
| 机制 A 门控 / B0 学生 | `runs/v25/mechanism_a/gate_*.pt`、`runs/v25/mechanism_b/student_B0.pt` | 待登记 | **待登记** |
| 教师缓存 | `runs/v25/cache/train_latents_4400.pt` | 待登记（含 teacher_spec，但锚 sha 未回填） | **待登记** |

**登记方式**：[../../src/v25/artifact_registry.py](../../src/v25/artifact_registry.py)（B03）——`register` 高频成本 ≈39.23ms/次（含 git 探测）或 1.814ms/次（设 `WEAVE_GIT_SHA` 后）；4090 独占文件由远程 `sha256sum` 回填后写入注册表；所有新锚/教师缓存必须随张量存 `LineageRecord.to_dict()`，加载不匹配即重生成（a02 约束 8）。`steps/lr` 必须进 `params_hash`（F8 事故）。

## 八、测试与护栏

```bash
python -m pytest tests/v25 -q                                          # 214 passed（2026-11-20 基线）
python -m pytest tests/test_markdown_links.py tests/test_no_absolute_paths.py -q   # 全仓护栏
python -m pytest tests/v25/test_stats.py -q                            # 16 passed（统计等价逐位断言）
```

- 统计基础唯一来源 [../../src/eval/metrics.py](../../src/eval/metrics.py)（`bootstrap_source_ci`/`paired_cluster_ci`）；A01 包装层单测断言与基础层 1e-12 等价。
- CPU 隔离纪律：`CUDA_VISIBLE_DEVICES=""`、`OMP/MKL/OPENBLAS_NUM_THREADS=2`；重训练/评测一律上远程并分进程运行。

## 九、未跑项（如实登记，不预写结论）

机制 B1（6000 upd）、C1 结构重设计全量教师生产与 C1/C2 训练、A07 端到端成本模型、B07 回归套件整合、B08 跨平台运行器整理、最终单样本双接口冒烟、R12 锚对齐重评（脚本就绪未跑）、Tr_T3 配对 CI。完成前**不得**在任何文档预写其成功结论。
