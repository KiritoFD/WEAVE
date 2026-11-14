# V15 Work Group A — 运行状态

**2026-11-15 起跑于 3060**（tmux session `v15`，bash 启动脚本 `_tmp/run_v15_tmux.sh`）。

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

- 0.155s/update → 12k updates ≈ **31 分钟/臂**（dim16 base；where/dim96 臂按 2–4× 估计）。
- loss 1.60 → 0.98 @ 1800 steps，下降正常。
- GPU 689MiB / 45%。

## 查看

- `tmux attach -t v15`（3060）
- `runs/v15/group_a.log` — sequencer 主日志
- `runs/v15/group_a_status.json` — 每臂状态
- `runs/v15/logs/W_*.log` — 每臂 stdout/stderr
- `runs/v15/W_*/train_curves.jsonl` — 每 200 opt step 一行

## 预计

16 臂（seed 0）串行约 **12–18h**。完成后跑统一诊断（Where Probe + 内容干预 + 750 评测）再决定 seed 1 复跑与 Group B/C/D。
