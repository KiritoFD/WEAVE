# V17 任务级结果（作数口径：cross600，官方裁判，2026-11-17）

**口径**：全 750 生成（V15 base + 学生残差，与 V16 B_adapter 同协议）→ S10 → 官方裁判 `src.eval.measure`（CSD-ViT-L / DINOv2-small / CLIP / LPIPS）→ cross600 汇总。**不用 holdout relMSE/cosine 作结论**（那些只是训练代理，见旧版 V17_PROGRESS_T14H.md 已作废的 P2 段落）。

## P2 学生 cross600（n=600）

| 臂 | CSD Top-1 | DINO-S | DINO-C | LPIPS | 说明 |
|---|---:|---:|---:|---:|---|
| V15 base (P0 baseline) | 0.1067 | 0.4606 | 0.7506 | 0.2700 | 冻结基座 |
| S0_eval（V16 L2 复现）| 0.1167 | 0.4634 | 0.7444 | 0.2764 | 正确标签 |
| S0_eval_wrong（错标签控制）| 0.1067 | 0.4624 | 0.7452 | 0.2759 | **正确−错误 = +1.0pp** |
| S1_eval（cosine+mask）| 0.1167 | 0.4633 | 0.7442 | 0.2765 | 与 S0 几乎逐位相同 |
| S2_eval（+DINO patch 上下文）| 0.1233 | 0.4651 | 0.7415 | 0.2801 | Top-1 +1.67pp vs base（超 B_teacher 11.83%） |
| **P2_S4_eval（+DINO + `u_tgt−u_src`）** | **0.1333** | **0.4663** | 0.7386 | 0.2845 | **V17 全场最高（直接启发 V18/V19 弦差分学生）** |
| P2_S6_eval / S6_wrong（+幅度压缩头） | 0.1150 / 0.1100 | 0.4619 | 0.7486 | 0.2717 | 幅度头限制了有效步长 |
| S8_eval（hard-gate 消融，死头污染）| 0.1050 | 0.4607 | 0.7507 | 0.2700 | `amplitude` 头在训练期零梯度、推理期 `clamp(0,2)` 截断退回基线 |
| P3_H4_full_eval（Signed RGB-H2） | 0.1100 | 0.4615 | 0.7480 | 0.2749 | 事后子带增益无法跨越源画派锁定 |
| P4_F0_eval（FM 非中心残差，8k）| 0.1100 | 0.4620 | 0.7516 | 0.2696 | 5 类均衡自配对 FM |
| P4_F1_eval（FM 类中心化残差，8k）| 0.1133 | 0.4622 | 0.7497 | 0.2714 | 类中心化仅 +0.67pp vs base |
| P4_F3_eval（FM 中心化+早段加权）| 0.1117 | 0.4613 | 0.7501 | 0.2706 | 早段加权无额外收益 |

*代码核查修正：`S8_eval` 的 `config_resolved.json` 经核实确为 `S8`（非此前猜测的配置覆盖），其退回基线是因为 `train_student.py` 未将 `self.amplitude` 乘入损失路径导致该线性层保持零梯度随机初始化，推理时被 `scale.clamp(0, 2)` 截断；`S3/S4` 无此头，`S5/S6/S8` 屏蔽死头后主干 `delta` 完好可评。完整跨代研判与下一步 V18 计划见 [../WEAVE_COMPREHENSIVE_SUMMARY_AND_NEXT_PLAN_20261117.md](../WEAVE_COMPREHENSIVE_SUMMARY_AND_NEXT_PLAN_20261117.md)。*

## 判读（初步，随补跑更新）

1. **学生残差确实带来 +1pp 级 Top-1**：S0/S1 0.1167 vs V15 base 0.1067；**错标签控制 0.1067 = base** → 这 +1pp 是标签特异的（正确标签优于错标签），不是任意扰动的副作用。
2. **DINO 上下文有独立增益**：S2 (0.1233) > S1 (0.1167)，+0.67pp，且 DINO-S 也升（0.4651）——**部分支持 P2 的"缺上下文"假设**（计划 §8.3 的 S2>S1 判据成立）。注意 DINO-C 微降 0.003、LPIPS 升 0.004（内容/失真小代价）。
3. **方向幅度仍小**：全部臂 Top-1 在 0.105–0.123 区间，远未接近实用（StyleShot 0.926 / V10 LD4 0.352），与"教师方向 cosine≈0.25"的代理读数自洽——方向弱是瓶颈。

## 工程修复（作数结果的必要前提）

- 学生评测 op 曾把残差加在源 latent 上 → 垃圾结果（top1 0.008 / dino_c 0.952）；已改为 `base=V15生成 + delta(base, canon(z), tid)`（对齐 V16 B_adapter）。
- 内联 measure 曾因生成器+裁判共存 OOM（12GB 卡）→ 改为**生成进程退出后单独 measure_only.py**（process-split），已端到端验证（S2 1600s 成功）。
- 全部失败臂的 raw payload 已作废重生成；14 个 P2/P3 eval 臂在 tmux `recoverfull` 串行补跑中（~6h）。

## 提示

`docs/performance/metrics.json|csv` 每次监控周期自动重建，含全部 cross600 作数条目 + 帕累托旗标；新臂落地即入表。`V17_PROGRESS_T14H.md` 中的 holdout 代理表仅为训练诊断，不代表任务级结果。
