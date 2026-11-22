# V26 最终模型配方 (L6 收官)

- **最佳候选**: `S0_vgg_global` ｜ 归类: **GLOBAL_REF_ONLY** ｜ Top-1: 16.17%
- **推理协议**: label-free (sid 槽位恒零), 条件 = (内容 latent, 参考图全局纤维 + adapter), 2 分支 CFG (wj=2.2, tau0=0.45), 10 步 RK2, S10 解码
- **训练配方**: 见 runs/v26/oneshot_val/<best>/meta (strict-cross 参考采样 + accum-RNG 修复 + VGG-mean swap λ_sw=0.5 λ_c=1.0)
- **解码器**: S10 (10.66M) — L2 归因结论见 ABLATIONS
- **诚实声明**: F2 类臂训练期教师使用真画派标签生成跨画派目标; 推理完全免标签
- **种子复验**: S0_vgg_global 三份独立读数 16.17%/16.17%(seed23)/16.00%(seed45); F2_global_teacher 16.00%/16.17%(seed23)/16.00%(seed45) — 聚类配对 bootstrap CI 全部含 0 (S0−F2 Δ=+0.10pp CI95 [−0.98,+1.01]pp), 两候选统计不可区分, 归类依据 = Top-1 相同下的最简机制 (无 SWD/无 RefAttn)
- **SG-25 OOD**: 见 RESULTS 八 (3060 125 对, swap_dist/δs 与 C2nt 基线同口径)
