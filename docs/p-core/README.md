# P-Core｜目标流形上的条件重绘

> 44,540,532 参数，PCoreSLFUNet base_dim=208、cond_dim=384；当前 C2 3500-step EMA。它不是 E 的替代品，也不是44M完整系统。E 与 P 的因果协作见 [E-Core](../e-core/README.md)。

## 1. 架构与训练目标

输入 z_t、t、16×16 Q32、Y16 LCN、目标画派 tid、K8 路由的417-D风格统计、gamma/layout_scale 与独立 condition dropout。只在粗尺度瓶颈约束布局，不保留直接传递源图高频的64/32尺度 skip；不使用 V33 的32×32 Adapter、Q64+ 或参考 Value。输出同形状速度场。

训练使用 Noise-to-Data FM：
\[
z_t=(1-t)\epsilon+t z_{\rm gt},\qquad
v^*=z_{\rm gt}-\epsilon.
\]
以 Pseudo-Huber(delta=.05) 回归速度而非 latent 终点 MSE；使用历史 Core-80% 采样、独立 content/style dropout（30%/15%）、15% Pool C 反向真画配对。V35 同构44M对照 3500 updates、effective batch128（64×2）、swd_micro=6，峰值约18.95GB，41.5–41.6分钟/臂（4090）。训练和评测裁判隔离。完整 loss 权重与采样以执行配置/源码为准，不将摘要视为所有超参的复现脚本。

## 2. A3：解析起点，而不是新网络

在 E 的 draft 上分解亮度投影 P_Y z_E 为低频 L_low 与高频 L_high；保留高频分量，对低频亮度矩与色度投影 P_C 做目标统计仿射校准。单次19×19可分离高斯分解，零学习参数。由 z_A 形成：
\[
z_{\tau_0}=(1-\tau_0)\epsilon+\tau_0 z_A.
\]
tau0=0 时 A3 不应影响输出。不要将 A3 无条件改写 E 原生输出。

| 44M 固定权重，P1 | ΔC | ΔS | ΔTop-1 |
|---|---:|---:|---:|
| A3 vs E draft @s085 | −.0014 | +.0089 [.0069,.0109] | +3.33pp |
| A3 vs E draft @s100 | +.0106 | +.0086 [.0066,.0104] | +2.83pp |

历史 V34 在固定 C=.50/.61/.70，A3 对同一 C2 的 S 增益约+.0104/+.0107/+.0090。扩容主要提高目标判别，A3 更直接推动等内容风格前沿。来源：[V34 Stage A/C/D](../archive/historical/v34/V34_FRONTIER_REPORT.md)；[P1](../archive/historical/v35/data/p1_ecore_report.json)。

## 3. Dual-CFG 与时间积分

四个分支分别为 v_sc、v_s0、v_0c、v_00；独立风格/内容 dropout 支持下式：
\[
v=v_{00}+w_c(v_{0c}-v_{00})+w_{s,j}(v_{sc}-v_{0c})+w_{sp,j}(v_{s0}-v_{00}).
\]
按进度调节 ws_j 与 wsp_eff；末端可用 rescale。现役 P8 配置 ws_mul=1、wc_mul=1.2、dtau0=-.05、alpha_y=alpha_c=1、6名义积分步，调度入口为 make_v35_schedule。源码积分器首步 midpoint、其余二阶多步，所以名义6步静态推导为7次四分支调用（28单条件等价 NFE），实际需 hook 核实，不能沿用旧24-NFE ledger。纯噪声独立档与 s150 不同：P0 的 s150 tau0=.36，不是纯噪声；具体参数以最终调度为准。

## 4. 容量、训练数据与时长：保留什么

| 受控证据 | 结果 | 决策 |
|---|---|---|
| 历史自配对 Flow 0.13M→34.2M | tau=.7 编辑 Top-1 10.67→6.67%，纯噪生成提高 | 扩容可强化错误恢复，不等于所有扩容无效 |
| V34 C0→C2（22.29M→44.54M，同配方 Q32） | s150 Top-1 76.00→91.17%，等 C=.50 +5.11pp | 当前生成容量有效 |
| V34 C1/C3（32×32联合训练） | @C=.61 相对同容量 Q32 低 .0137/.0159 S | 结构信息不必越多越好 |
| V35 Pool C 0%→15% | s125 Top-1 78.17→80.83%；s150 88.83→91.83%，Same94→98% | 保留，主要守住强风格端 |
| P5 3500→7000 EMA | s100 Top-1 69→64%，训练 loss 降至.7735 | 冻结3500 EMA |

P5 原 checkpoint 没有 optimizer/online，使用重新预热 AdamW 续训；因此只证明这次续训无益，不能证明自然优化轨迹的绝对峰值。Stiefel 的独立收益与限制见 [Conditioning](../conditioning/README.md)。

## 5. 当前重绘全谱（P8/P9a）

| s | C | S | Top-1 |
|---|---:|---:|---:|
| .60 | .7813 | .4636 | 38.83% |
| .65 | .7443 | .4831 | 47.83% |
| .68 | .7328 | .4903 | 48.33% |
| .70 | .7276 | .4924 | 49.00% |
| .72 | .7133 | .5002 | 51.33% |
| .75 | .6875 | .5101 | 55.67% |
| .85 | .6514 | .5261 | 57.83% |
| 1.00 | .5502 | .5597 | 69.67% |
| 1.25 | .4678 | .5749 | 80.83% |
| 1.50 | .3435 | .5919 | 91.17% |
| tau0=0（独立配置） | .4295 | .5883 | 70.33% |

表内 .60–.72 来自 P9a，.75–1.50 来自 P8，均为完整 test750 的 Cross600。最终 s085 ER/Uki 25.8/26.7%，明显落后 Imp/Min 93.3/92.5%；s150 回升至90.8/79.2%。完整等内容比较见 [System](../system/README.md)。

## 6. 停止探索的路径与资产

- V34 D_merged 高方差桥接训练将35%样本速度换成 (z1-z_start)/(1-tau0)，残差在 tau=.6–.8 被放大约2.5–5倍，loss 约.82→1.25，Top-1 损失13–17pp。保持干净 FM，用 A3 作为解析起点。
- V33 Q64+/高分辨率控制、V30 区域损失、V31 EditFlow/NeutralBridge 未推动现役前沿；不再用复杂注入补偿错误训练目标。
- 最终权重：runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt（EMA，3500步）。V32 22M、C0–C3、P3/P4、Unified58M 均进入论文复现冷档，不删唯一副本。
- [训练与加载入口](../../exp/v35_final/) · [积分器](../../exp/v33_semantic_control/models.py) · [V34训练](../../exp/v34_frontier/)；PCoreSLFUNet 依赖的 exp/v29_biglittle 源码未随当前 main 提交，复现前须从实际节点补齐。
- 原始证据：[P3](../archive/historical/v35/data/p3_stiefel_poolc_report.json) · [P5](../archive/historical/v35/data/p5_duration_report.json) · [P6](../archive/historical/v35/data/p6_hyper_report.json) · [P8](../archive/historical/v35/data/p8_final_report.json) · [P9](../archive/historical/v35/data/p9_paper_audit_report.json)。
