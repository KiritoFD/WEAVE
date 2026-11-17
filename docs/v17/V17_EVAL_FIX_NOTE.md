# V17 评测链修复记录（2026-11-17，作数结果）

## 问题链（全部已定位）

学生评测臂（P2_*_eval）首轮全部产生**垃圾指标**：S0_eval top1=0.008、dino_c=0.952（输出≈源图）。根因是 `arm_student_eval` 的 op **把残差加在原始源 latent 上**（`z + delta(z)`），而学生训练时学的是 `base + delta(base, content, tid)`，其中 base=V15 生成结果、content=canonicalizer(z)。修复后（对照 V16 B_adapter 评测）S0_eval cross600：**top1=0.117、dino_s=0.463、dino_c=0.744、lpips=0.276**（vs V15 base 0.1067 → +1pp，真实但小）。

第二轮队列跑 S1-S8 eval 全部 rc=1（`measure failed`）。根因：**14 个僵尸 python 进程并发占满 11976/12288MiB 显存**（此前手动补跑 + 队列并发 + 进程未退）。全部 taskkill 清理后 GPU 回到 115MiB。

## 工程规则（固化）

1. 学生评测必须走 `base = v16_generate(v15, z, tid, eps); delta = net(base, canon(z), tid); out = base + delta`（已修入 run.py）。
2. **禁止并发跑多个 3060 eval/train 进程**：单卡串行。任何手动补跑前必须确认队列空闲；`taskkill /f /im python.exe` 会误杀队列，须按 PID 精确杀。
3. 监控循环只拉状态/JSON，不在 3060 启动任何 GPU 进程。

## 当前状态

- 主队列健康推进：P3 H4 训练中（12066MiB 单进程）。
- S1-S8 eval + S7 train + H2arm eval 尚未落地（被 OOM 中断/重试耗尽），**待 H4/H5 完成后单进程串行补跑**。
- 台账 61 条，V17 有效臂仅 P0 + S0_eval。
