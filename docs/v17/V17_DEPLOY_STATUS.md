# V17 部署状态（2026-11-16/17）

## 实现清单（全部提交，23/23 单元测试绿）

| 模块 | 内容 |
|---|---|
| exp/v17/common.py | 路径/原子 IO/sha/seed/heartbeat/类均衡索引 |
| exp/v17/data.py | NFC+去重+Test750/参考排除审计（复用 v16.pool_samples）|
| exp/v17/checkpoint.py | 全状态 ckpt（model/opt/sched/RNG×4/scaler），20步==10+resume+10 严格断言 |
| exp/v17/eval750.py | 通用 750 评测：配对噪声→S10→RawWriter→measure→summary/confusion→DONE.json |
| exp/v17/probe.py | P0：t 曲线 R_style/FM MSE/cos 对齐；CFG w∈{1,2,4}+early/late；五标签互换 |
| exp/v17/teacher.py | P1：T0-T4（V16 Teacher 派生；T1 校准化方向、T2 Target−Source、T3 每目标低频 cap、T4=T2+T3）|
| exp/v17/dino_cache.py | 池源 DINO patch token 缓存（B×384×16×16， referee 同源 encode_dino）|
| exp/v17/adapter.py + losses.py | S0-S8 学生族 + Phase I/II 损失（E_rel 恒等式断言、零教师安全、gate 仅作用于范数项）|
| exp/v17/train_student.py | 12k updates 训练 + 1000 步全状态 ckpt + holdout 拟合 |
| exp/v17/h2.py | P3：H1 逐画派/逐子带 H2 统计 → H2arm 解析符号增益（idwt2 正确形态）→ H3/H4 零初始化学习头 + 合成污染复原监督 |
| exp/v17/fm_residual.py | P4：F0/F1/F3 类均衡 FM 残差；F2 桥接监督（自配对拒绝合同）|
| exp/v17/run.py | 统一入口 --config/--seed/--resume auto/--dry-run/--eval full750，11 个 family 分派 |
| exp/v17/build_manifest.py | 52 主臂 + 重复池（S5/S6/H4/F1 × seeds1-32）+ long-run {24k,48k} |
| scripts/v17/queue.py | 串行队列：DONE 准入（hash）、重试≤2、24h floor/36h target、原子 queue_state.json |
| tests/v17/*.py | 23 断言：resume 等价、logit bias 梯度不变、centered 零和、恒等式、DWT 往返、F2 拒绝、DONE 准入等 |

## Canary 状态（3060 实测）

- **dry-run**：全部 family PASS（probe/cfg/teacher/student S0+S2/h2/fm）
- **真实 T0 教师对**：338s，2000 cross 对（含 mask，S1 BCE 合同就绪）
- **真实训练 canary（2 updates）**：S0 ✓ / H3 ✓ / F0 ✓（各含全状态 ckpt 落盘）
- **评测链 canary**：2-update 学生 → 750 生成 → measure（CSD 全裁判）→ wrong-label → **DONE.json 写入 ✓**（916s）

## 环境决策

- **Windows 原生 Python + psmux**（非计划的 WSL2）：理由——V16 全部基线在此环境产生，消除 §6.1 的跨 OS 复现风险；CSD/DINO/资产全部在此环境验证可用。

## 队列启动（2026-11-17 01:05，用户决断：停 scaling，直接拉起 V17）

- scaling v2 终止于 S6（S1-S6 数据保留，S7_dim512 / B8k_dim96 按用户决断放弃——"已没收益"）。
- 队列首个真实臂 **P0_cfg2 在 5.7 分钟完整落地**（750 生成 + CSD 全裁判 + DONE 准入），queue_state 的 valid_minutes 修复后正确计数。
- 修复记录：`_cfg_sample` 导入位置；队列准入扩展到非评测臂（result.json ok）；分钟数 UTC 解析（此前 mktime 伪 480 分钟触发假 36h 停止）。
- 36h 时钟自 2026-11-16T17:05 UTC 起算；监控：queue_console.log / queue.log / queue_state.json / 每臂 progress.json。

## 监控

- 本地：`runs/v17/orchestrate.log`（启动链）
- 3060：`runs/v17/queue_console.log`（队列主日志）、`runs/v17/queue_state.json`（原子状态）、`runs/v17/logs/*.log`（每臂）、每臂 `progress.json` heartbeat
- 36h 时钟自队列启动起算；24h floor / 36h 停止新增 repeat 臂。
