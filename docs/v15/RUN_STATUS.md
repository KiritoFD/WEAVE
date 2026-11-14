# V15 Work Group A — 运行状态

**2026-11-15 起跑于 3060**（tmux session `v15`，pane 内 **cmd 直跑** `_tmp/run_v15_tmux.bat`，无重定向、pane 即进度）。

> **infra 教训**：3060 的 tmux pane 里 **MSYS bash 会静默挂死**（进程不启动、无输出、无退出）——首跑假成功即由此而来。tmux 里一律用 `cmd /c ...bat` 跑 Windows 原生进程。另：`schtasks` 遗留任务（`v15pipeline`）会造成幽灵进程/撞车，起跑前必须 `schtasks /Query` 清点并删除同名任务。

## 起跑前验证（全部通过）

1. **冒烟矩阵**（`exp/v15/smoke_all.py`，18 变体 × 2 updates，单进程）：4 容量(16/32/64/96) × 2 上下文(conv/att) × 2 Where(base/where) + spatial_film + router —— 全部 PASS，loss 有限且下降。
2. **集成测试**：sequencer 真实命令路径 200 updates → config/curve/ckpt_200/final/eval_noise 五件套齐全，loss 1.66→1.43。
3. 冒烟矩阵抓出并修复 6 个 bug（见 git log f9a5ec8…本系列）：
   - `fm_loss` 引用未定义 `device`（NameError，3080 首跑即崩）
   - sequencer 缺 `import json`（第一臂写状态即崩）
   - encoder `ResBlock` 调用漏 `cond` 参数
   - where 边缘图 gx/gy 形状 (B,64,63) vs (B,63,64) 不可加
   - SpatialFiLM `global_fc` 输入维度不匹配 decoder 各级 ch
   - SpatialRouter 对已 flatten 张量调 Conv2d

## 训练配置

- 数据：`I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16`，train split（md5(pid)%10<9）= **13,992 latents**（ER 1319 / Imp 8089 / Min 997 / Roc 2264 / Uki 1323）。
- 每臂 12k opt updates，有效 batch 128（micro×accum）。
- **显存计划**（不用 allocator flag）：dim96×where 两臂预设 micro=16/accum=8（冒烟峰值 micro=32 时 10–11.5GB）；其余 32/4。任何残余 OOM 自动 micro 减半重试一次（有效 batch 不变，协议安全）。
- 超时单臂 6h；状态机 `runs/v15/group_a_status.json`，每臂日志 `runs/v15/logs/<tag>.log`。

## 首臂实测（W_dim16_conv_base）

- **micro 表提速后：0.075s/update → 12k updates ≈ 15 分钟/臂**（base 臂；where 臂按 2–4× 估计）。
- loss 1.60 → 0.946 @2600 steps，与 micro=32 逐点重合（梯度等价验证）。
- GPU 1606MiB / 73% / 121W（micro=32 时 689MiB / 45% / 85W）。

## micro 表（有效 batch 恒为 128，协议不变）

| 臂 | micro/accum | 依据（冒烟每样本激活） |
|---|---|---|
| 全部 base 臂（10 条）| 128 / 1 | 激活 <25MB/样本 |
| dim32 × where（2 条）| 64 / 2 | ~80MB/样本 |
| dim64 × where（2 条）| 32 / 4 | ~150–180MB/样本 |
| dim96 conv where | 32 / 4 | ~242MB/样本 |
| dim96 att where | 16 / 8 | ~290MB/样本 |

**协议约束**：micro ≤ 有效 batch。要填到 ~10GB 显存需将有效 batch 提到 512+，属预注册协议变更（需重标定 lr、噪声水平改变）——除非用户明示，否则不做。

## 查看

- `tmux attach -t v15`（3060）
- `runs/v15/group_a.log` — sequencer 主日志
- `runs/v15/group_a_status.json` — 每臂状态
- `runs/v15/logs/W_*.log` — 每臂 stdout/stderr
- `runs/v15/W_*/train_curves.jsonl` — 每 200 opt step 一行

## 用时表（2026-11-15 用户决断：4k updates/臂、两阶段跑序、朴素训练）

**变更记录**：12k→6k→4k（loss 拐点 ~1–2k 步，12k=110 epochs 过度）；跑序改为先全部 base 臂（大→小）再全部 where 臂（大→小）；pool cache 与 `cudnn.benchmark` 已回退——后者与 CFG drop（p_drop=0.15，unc 前向 batch 1..20 随机）交互导致每新 shape 全量 autotune，nvidia-smi 呈突发-停顿循环，去除后恢复线性 0.328 s/upd·98% util。

| 阶段 | 臂 | 配置 | 用时/臂 |
|---|---|---|---|
| 1 朴素 | dim96 conv/att base | 128/1 | 22 / 26 min（**实测 0.328 s/upd**）|
| 1 朴素 | dim64 conv/att base | 128/1 | ~13 / ~15 min |
| 1 朴素 | dim32 conv/att base | 128/1 | ~7 / ~8 min |
| 1 朴素 | dim16 conv/att base | 128/1 | ~5 / ~6 min |
| 2 where | dim96 conv_where | 32/4 | ~60–70 min |
| 2 where | dim96 att_where | 16/8 | ~110–130 min |
| 2 where | dim64 conv/att where | 32/4 | ~35 / ~37 min |
| 2 where | dim32 conv/att where | 64/2 | ~11 / ~13 min |
| 2 where | dim16 conv/att where | 128/1 | ~7 / ~8 min |

**seed 0 总计 ≈ 7.5h**（每臂 +~3 min 冷启动 pool 加载）。不足则从 ckpt_3000/ckpt_4000（含 opt+sched 状态）resume 延长。
