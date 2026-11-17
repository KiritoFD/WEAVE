# V17 进度与结果报告（T+13.4h，2026-11-17）

## 队列状态

有效任务时间 **13.37h / 36h 目标**；串行队列 + 1h 监控循环健康运行。

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 条件探针 | 6/8 完成 + 2 手动补齐 | cfg_early/late 工件已补（队列状态仍标 failed） |
| P1 教师 T0-T4 | 5/5 pairs 就绪 | T1/T3/T4 曾失败（缺 bias 文件 / `calib` 键），已修复并手动补跑（各 2000 cross pairs） |
| P2 学生训练 | 8/9 完成 | S0-S6/S8 train 完成（74–105min/臂），S7 train 待补（依赖 T4 pairs，现已就绪） |
| P2 学生评测 | 补跑中 | 首轮 8 个 eval 臂因 student_dir 路径 bug 秒级失败，已修复并在独立 tmux `recover` 补跑 |
| P3 RGB-H2 | H1 完成 / H3 训练中 | H2arm 曾因双重 decode 失败，已修复（output_rgb）待补 |
| P4/P5 | 未启动 | 队列自动推进 |

## 关键科学结果（已完成部分）

### P0：条件强度与标签跟随（cross600，n=600）

| 臂 | CSD Top-1 | DINO-S | DINO-C | LPIPS |
|---|---:|---:|---:|---:|
| V15 baseline (B_base) | 0.1067 | 0.4606 | 0.7506 | 0.2700 |
| CFG w=2 | 0.1100 | 0.4606 | 0.7507 | 0.2706 |
| CFG w=4 | 0.1117 | 0.4603 | 0.7501 | 0.2732 |
| CFG early(t≤.35) w=3 | 0.1050 | 0.4607 | 0.7507 | 0.2700 |
| CFG late(t>.35) w=3 | 0.1117 | 0.4604 | 0.7506 | 0.2717 |
| t0=0 五标签互换 | 0.200–0.215（随机）| ~0.545–0.553 | ~0.30 | ~0.81–0.86 |
| t0=.7 五标签互换 | 0.103–0.105 | ~0.460 | ~0.750 | ~0.267–0.272 |

**结论**：
- CFG 把 R_style 放大 w=2→4，Top-1 只买 +0.5pp（0.110→0.112），且 early/late 分段无差异 → **现有条件方向的幅度不是瓶颈**（复现计划 §8.1 预警）。
- t0=0 五标签互换全部落在随机 0.20 附近（标签零跟随）；t0=.7 标签互换与 baseline 同水平 → **固定 eps 下换网络标签不能改变目标画派响应**。

### P2：学生 holdout 方向复现（clean holdout，cross 子集）

| 臂 | 相对 MSE ↓ | 方向 cosine ↑ |
|---|---:|---:|
| S0（V16 L2 复现）| 1.179 | 0.249 |
| S1（cosine + mask）| 1.177 | 0.249 |
| S2（+DINO 上下文）| 1.489 | 0.238 |
| S3（+类中心化）| 1.577 | 0.239 |
| S4（+target-source）| 1.742 | 0.256 |
| S5（+幅度校准）| 0.945 | 0.241 |

对照 V16 学生：relMSE 0.935 / cos 0.256。**S0/S1 与 V16 几乎逐位一致**，DINO 上下文反而恶化 relMSE、方向无改进；S5 幅度项把 relMSE 拉回 0.945（与 V16 相当）但 cosine 仍 0.24。按计划 §8.3 判读：S2>S1 不成立（缺上下文不是主因）、S3/S4 仅 micro 变化、S5 只修幅度——**教师方向不可更好复现，方向本身是瓶颈**。

### P3：H1 画派频谱统计（clean train，每派 100 幅，H2 三方向 RMS）

| 画派 | H2 RMS（dir1/dir2/dir3） |
|---|---|
| Early Renaissance | 0.117 / 0.127 / 0.045 |
| Impressionism | 0.094 / 0.092 / 0.041 |
| Minimalism | 0.085 / 0.088 / 0.027 |
| Rococo | 0.086 / 0.088 / 0.034 |
| Ukiyo-e | **0.144 / 0.139 / 0.063** |

Ukiyo-e 高频能量最高、Minimalism 最低——与 V11b 真画 H2 分析（Uki 0.111 最高）一致，**不存在统一"应压制 H2"的低频画派**（计划 §0.1）。

## 工程修复记录（T+10h→T+13.4h）

1. **T1/T3/T4 教师失败**：缺 `r1_logit_bias.pt`（改为从独立 T0 holdout 生成）、`lf_caps_from_pairs` 引用不存在的 `calib` 键（改用 holdout）→ 5 个 pairs 全部就绪。
2. **P2_*_eval 秒级失败**：student_dir 默认路径 `OUT/Sx/Sx_s0` 错误（实际 `OUT/P2_Sx_train`）→ 已修，8 个 eval 臂在 tmux `recover` 补跑。
3. **S7_train 失败**：T4 pairs（seed=0 补跑）与 dino_cache（seed=20261117）只有 22/500 id 直接匹配 → 改为 canonical pid 匹配（去前缀/casefold），S7 待补。
4. **P3_H2arm 失败**：op 已返回 RGB，evaluate_arm 又 decode 一次 → 加 `output_rgb=True`，待补。
5. **监控循环**：1h cadence 稳定运行（首轮 decode/path 问题已修），台账随臂落地自动扩至 59 条。

## 下一步

- recover 会话完成 9 个 eval/H2arm 臂 → 学生全表入台账 → 与 V16 B/C 系列对比
- S7 train 单独补（~100min，等主队列 P4 空档或并发安全时）
- P3 H3/H4/H5/H6/H7 + P4 F0-F3 队列自动推进
- 36h 目标不变；全部主臂完成后进预注册 Seed 重复池
