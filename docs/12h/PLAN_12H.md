# WEAVE：更新后的 12 小时实验方案

已检查最新主分支 8b4a76a7929e（2026-11-03）。既然 eval 已经复现，不再投入时间重复基线评测，直接进入模型瓶颈诊断和训练实验。

我的判断是：目前最值得解决的不是模型太小，而是非配对 Flow Matching 的训练目标是否可学习，以及模型是否在真实 ODE 轨迹上学到了有效的风格变换。扩大容量应当作为这一假设的对照实验，而不是默认方向。

## 一、最新代码带来的调整

| 已确认情况 | 实验决策 |
|---|---|
| 已有 tools/diagnose_saturation.py | 直接补正后运行 A2/A3/A4，不重新开发 |
| train_sample_limit、pairing_mode=fixed 已接入 | 可以进行严格的 32/128/512 小集过拟合实验 |
| identity_prob 已实现真正的 identity 监督 | 不需要再修改 Dataset |
| t_sampling_legacy_compat 已支持显式关闭 | 直接对比 Uniform 与真实 Logit-Normal |
| 内部门控早停默认开启 | 诊断和容量扫描必须关闭自动早停，仅记录梯度变化 |
| eval_fast.py 已提交 | 使用已复现的协议和指标作为固定评测基准 |
| Route A 输出监督已扫描多个配置 | 本轮不重复原有 λ 扫描 |
| Route C 两 seed 收益接近零 | 暂停原版 DINO Reference Encoder 路线 |
| Route D 纹理模块两 seed 收益接近零 | 暂停单纯增加纹理模块 |
| Route B 的模块已写但未真正接入主训练 | 保留一次独立机制验证，不直接启动全量训练 |

新一轮仍使用已经复现的论文协议：750 all-pairs、DINOv2-small、CLIP ViT-B/32、8-step ODE、对应的 AdaIN 配置。同时输出 600 跨风格、150 同风格拆分结果，避免同风格样本掩盖风格迁移能力。

历史 0.4916/0.4918 可以作为参考，但模型晋级应当比较刚复现的相同协议、相同代码基线，不能跨不同评测主干比较。

## 二、12 小时预算分配

| 阶段 | 时间 | 核心任务 |
|---|---|---|
| P0 | 0:00–0:30 | 诊断代码正确性、运行环境、评测锁定 |
| P1 | 0:30–2:00 | 小集过拟合、标签歧义、TF/ODE 轨迹诊断 |
| P2 | 2:00–5:00 | 训练目标：配对、identity、时间采样、损失设计 |
| P3 | 5:00–8:30 | 模型容量：深度、宽度、Style Memory、条件注入 |
| P4 | 8:30–10:30 | 针对性组合、梯度隔离、真实轨迹监督 |
| P5 | 10:30–12:00 | 多 seed 复验、完整评测、决策报告 |

以现有 3060 记录估计，普通 Flow-only 模型 batch 160 约 0.45–0.50 秒/step；2,000 steps 约 15–17 分钟。较大模型与可微输出监督的时间必须另外实测。因此主实验采用 600-step 快筛 → 2,000-step 复验。

## 三、P0：先修三个诊断问题（0:00–0:30）

| 位置 | 当前实现问题 | 必须修正 |
|---|---|---|
| analysis_a3 | 使用原始 target-content | 必须使用 FlowMatchingObjective._build_training_target() 生成的真实监督 |
| analysis_a2 | 在 t=0.5 固定一个预测，却比较多个不同目标的速度 | 改为 t=0，固定风格条件，测目标标签方差 |
| analysis_a4 | 所谓 AdaIN scale 实际是对 style_latent 乘系数 | 必须调用真实 integrate_transport()，否则实验结论错误 |

另外 checkpoint 加载目前使用 strict=False，正式诊断应要求预期参数全部匹配。同时完成环境检查：使用服务器现有 torch 2.10.0+cu128，不升级；确认显存空闲、冻结 eval 配置、固定参考图选择和推理 batch。本阶段不重新训练基线。

## 四、P1：找出为什么几百步就饱和（0:30–2:00）

### P1-A：受控过拟合

| ID | 数据规模 | 配对 | 训练预算 | 判断目标 |
|---|---|---|---|---|
| A1 | 32 | 固定 | 2,000 steps | 能否拟合极小数据集 |
| A2 | 128 | 固定 | 2,000 steps | 拟合难度是否快速增加 |
| A3 | 512 | 固定 | 2,000 steps | 容量或目标冲突是否出现 |
| A4 | 128 | 每 epoch 重采样 | 2,000 steps | 随机监督是否导致平台期 |

使用现有 train_sample_limit 和 pairing_mode，不需要另建 Dataset。注意：不同训练子集需要相应缩小 microbatch；不能把 batch 设得大于子集长度，并使用 drop_last 导致没有有效训练步。

所有实验记录：L_LL, L_LH, L_LH, cos(v_θ,u)，‖v_θ‖/‖u‖，分别测训练集与固定验证样本。

### P1-B：不训练的机制诊断

| ID | 测量 | 要回答的问题 |
|---|---|---|
| A5 | Top1/4/8 目标速度条件方差 | 非配对监督的不确定性有多大 |
| A6 | 五个时间分桶的 velocity error | 是否只学会了接近真实目标的 x_t |
| A7 | Teacher-forcing vs 自回归 ODE | 推理轨迹是否脱离训练分布 |
| A8 | 固定内容，交换 style ID | 网络究竟有多依赖风格条件 |
| A9 | 真实 ODE、仅网络、不同 AdaIN 位置 | 风格提升来自网络还是后处理 |

核心判断：若 A1 可以近乎拟合、A4 无法降低误差，且 A5 标签方差很高，则监督目标优先于模型容量。若 A1 都无法有效拟合，再优先改主干结构。若 teacher-forcing 正常、自由 ODE 明显退化，则后续应加真实轨迹监督，而非单纯增加层数。

## 五、P2：训练目标实验（2:00–5:00）

采用相同初始化、数据量、optimizer 更新数。第一轮每组 600 steps；表现最好的 3 组延长到 2,000 steps。

| ID | 方案 | 实现 | 核心假设 |
|---|---|---|---|
| T0 | 原始 FM | 不变 | 对照 |
| T1 | Cross-only | identity_prob=0 | 同风格随机目标可能干扰训练 |
| T2 | 20% 真 Identity | identity_prob=0.2 | 稳定内容保持而非同域随机迁移 |
| T3 | Top1 配对 | 使用已有 pairing cache | 减少标签方差 |
| T4 | Top4 配对 | 同上 | 保留一定多样性 |
| T5 | 无桥噪声 | bridge_sigma=0 | 减小早期学习难度 |
| T6 | 早期偏置采样 | 真正的 Logit-Normal，μ=-1 | 加强推理起点 |
| T7 | Huber | loss_type=huber | 减少离群配对影响 |
| T8 | 内容锚定 HF 端点 | 新目标实现 | 解决不相关绘画的逐像素 HF 监督冲突 |

T8 是最重要的新设计。现有目标为 H* = H_random style，改成 H* = (1-α)H_c + α·StyleTransform(H_c, S_s)，其中 S_s 是训练集统计得到的目标风格信息，变换作用于内容自身的 HF，而不是复制另一张画的空间 HF。先测试 α=0.5, 1.0。若 T8 显著降低训练难度并提高风格指标，说明以前主要是在要求模型学习一个不合理的空间映射；若只降 FM loss 而不涨 DINO-S，需要进一步修改风格监督，不能把 FM loss 的改善当成功。

## 六、P3：模型容量和条件结构（5:00–8:30）

### P3-A：容量扫描

| ID | Width | Depth | 目的 |
|---|---|---|---|
| M0 | 64 | 4 | 基线 |
| M1 | 64 | 8 | 加深 |
| M2 | 48 | 12 | 深而窄 |
| M3 | 96 | 4 | 加宽 |
| M4 | 96 | 8 | 综合扩容 |
| M5 | 128 | 4 | 更宽的对照 |
| M6 | 96 | 6 | 中间规模 |

先全部使用原始 T0 监督，600 steps 快筛。然后把最有希望的两种模型配上 P2 最优目标，各进行 1,200–2,000 steps 复验。对于大模型，降低 microbatch、增加梯度累积，保证有效 batch 与总样本曝光量一致，不得直接拿 b160 和 b32 的相同步数比较。

### P3-B：Style Memory 容量

model.py 构造 StyleConditioner 时把 num_memory_tokens=256 写死，因此修改 config.json 的 style_attn_num_tokens 不会完成真正的容量消融。应先把 token 数变为真实模型构造参数，然后测试：

| ID | Memory tokens | 主干 | 目的 |
|---|---|---|---|
| S0 | 256 | 64×4 | 当前实现 |
| S1 | 64 | 64×4 | 判断大量记忆是否冗余 |
| S2 | 32 | 64×4 | 风格表征下限 |
| S3 | 64 | 96×4 | 把参数预算转给主干 |

另外，当前 Cross-Attention 使用未归一化的 ReLU² attention，改变 token 数会同时改变 attention 输出幅值，因此必须加一个输出幅值归一化对照。

### P3-C：条件注入（最多两组）

| ID | 改动 |
|---|---|
| S4 | Style-AdaLN 替代单纯 Cross-Attention |
| S5 | Style-conditioned HF velocity head |

不在本轮重做 raw spatial HF、单纯放大 gate、HH head 或旧版 DINO reference 分支。

## 七、P4：组合与真实生成轨迹（8:30–10:30）

| ID | 实验 | 启动条件 |
|---|---|---|
| R1 | 最佳训练目标 + 最佳主干 | P2/P3 出现独立收益 |
| R2 | HF residual 读取 h.detach() | 共享主干梯度竞争明显 |
| R3 | epoch 4 后冻结主干，只训练风格分支 | 风格早期达到峰值随后退化 |
| R4 | 2-step 可微 ODE rollout + 局部 HF 分布损失 | TF/ODE 轨迹差异明显 |
| R5 | 4-step rollout 对照 | R4 确实有效才执行 |
| R6 | Route B 的独立伪目标质量测试 | T8 证明内容锚定端点有价值 |

R4 与已失败的早期 SWD 实验区别：它监督模型从内容出发生成的真实轨迹终点，而不是训练时包含真实目标的 x_t。建议 L = L_FM + λ_s L_HF patch(x̂1, y_s) + λ_c L_structure(x̂1, x0)，先在 latent 小波域实现局部 patch 分布监督，避免立即加载 VGG/DINO。Route B 暂不直接合并主模型，先要求伪目标自身同时具备风格变化、内容保持和不复制参考图布局的证据。

## 八、P5：最终复验（10:30–12:00）

从 R/T/M 中选择最佳两个模型，加上固定基线：补 seed 7、123（35 min）、固定 eval 协议完整评测（30 min）、逐风格分析与误差归因（15 min）、整理曲线和结论（10 min）。最终报告不仅给最佳 epoch，还要给从 200 到 2,000 steps 的指标曲线。若 600 steps 最好、2,000 steps 退化，需报告清楚是过度平均、风格梯度失效，还是主干与风格支路协同适应。

## 九、3060 12GB 工程实施

### 1. 训练配置

为每组使用独立继承配置，保留原始配置不动：

{ "_base": "../config.json", "training": { "seed": 42, "batch_size": 96, "accumulation_steps": 1, "stop_after_global_steps": 600, "save_step_milestones": [200,400,600], "full_eval_each_epoch": false, "internal_probe_enabled": true, "internal_probe_every": 3, "internal_early_stop_enabled": false, "resume_checkpoint": "", "resume_prefer_local_checkpoint": false, "torch_compile": false }, "checkpoint": { "save_dir": "runs/12h/T01" } }

注意 num_epochs 仍决定 CosineLR 的周期，必须根据每个实验实际 steps/epoch 同步设置，不能只改 stop_after_global_steps 而让学习率调度不一致。Post-training 若使用旧 checkpoint，还必须设置 resume_optimizer=false、resume_training_state=false，并确认日志中的实际 LR。

### 2. CUDA 12.8 与 Compile

保留 BF16 AMP、TF32、cuDNN benchmark；检查 fused AdamW 是否生效；保留当前已验证的 channels_last；扫描阶段 torch.compile 默认关闭；gradient checkpoint 仅大模型或 ODE rollout 必要时开启；不使用 max-autotune。要专门测 compile 时做一次 100–200 step benchmark（backend=inductor, mode=default, fullgraph=false, dynamic=false），记录 Eager 与 Compiled 的冷启动时间、稳态 step time、峰值显存、数值差异。当前训练器要求 channels_last=false 才能打开 compile。

### 3. 必须修正的工程细节

| 文件 | 修改 |
|---|---|
| tools/diagnose_saturation.py | 修复真实目标、真正的 AdaIN 和严格 checkpoint 加载 |
| model.py | Style Memory token 数配置化；可选 HF residual detach |
| flow.py | 新增内容锚定 HF 目标模式 |
| config_schema.py | 新开关显式声明、校验参数范围 |
| trainer.py | 保留 step checkpoints；扫描期关闭主动早停 |
| tools/eval_fast.py | 锁定已验证的推理 batch、backbone、参考图协议 |
| runs/registry.jsonl | 记录配置 SHA、代码 SHA、seed、曝光量、有效 batch、结果路径 |

特别注意：历史记录发现推理 batch 2 与 32 的生成图存在可观测差异，即便 eval_fast 更快，也必须使用已完成复现验证的固定 batch 和数值路径。

## 十、12 小时后如何做决策

| 实验现象 | 下一代主要方向 |
|---|---|
| 固定配对能拟合，随机配对不能；T8 显著有效 | 重构训练目标，优先发展内容锚定风格操作 |
| T0 与 T8 都欠拟合；加深/加宽明显有效 | 扩大有效模型容量 |
| 加大模型只降低 FM loss，不提高风格指标 | 监督信号瓶颈，不再盲目扩容 |
| 风格早期最优，后期下降；R2/R3 有效 | 分离内容主干与风格分支的梯度所有权 |
| Teacher-forcing 好、ODE rollout 差；R4 有效 | 转向轨迹一致的输出监督 |
| Style Memory 压缩无损，扩大主干有收益 | 减少记忆表参数，把容量转给 spatial 计算 |

优先级：P1 诊断 → T8 内容锚定目标 → 深度/宽度扫描 → R2 梯度隔离 → R4 真实轨迹监督。

## 十一、评测判据（2026-11-04 最终口径，覆盖此前一切判据）

> **唯一官方口径 = 750 all-pairs 全量混合（600 跨风格 + 150 同风格，全部计入）。**
> 与论文主表同协议，跨 run 比较一律用这一张表。
> **750 是主判据；600 跨风格 / 150 同风格只作为解释性拆分，不得单独当晋级依据。**

唯一合法锚点：`base_ep10`（`exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt`）+ CLIP ViT-B/32 + b8 + 8 步 ODE + AdaIN 2.0 + 256px，产物 `runs/12h/anchors/base_ep10_600/`：

| 子集 | n | dino_s | dino_c | clip_s | lpips | 用途 |
|---|---|---|---|---|---|---|
| **750 all-pairs** | **750** | **0.48587** | **0.82902** | **0.70940** | **0.25850** | **主判据（唯一晋级依据）** |
| 600 跨风格 | 600 | 0.39572 | 0.82493 | 0.67779 | 0.26267 | 解释性拆分（附注） |
| 150 同风格 | 150 | 0.84650 | 0.84540 | 0.83587 | 0.24178 | 解释性拆分（附注） |

### 候选门槛（750 口径，四条全过才算候选，任一条不过即无效）

| 条件 | 门槛（锚点 ± 原约定绝对 delta） |
|---|---|
| dino_s | ≥ 0.48587 + 0.015 = **0.5009** |
| dino_c | ≥ 0.82902 − 0.015 = **0.8140** |
| clip_s | ≥ 0.70940 + 0.010 = **0.7194**（才算有风格收益） |
| lpips | ≤ 0.25850 + 0.010 = **0.2685** |

旧判据全部作废：`dino_c > 0.78 / lpips < 0.33`（错误基线推出）以及基于 600 跨风格锚点平移出的 `.4107/.8099/.6878/.2727`。所有数字必须带 `eval_batch`，不同 batch 不得横比。

### A1–A4 历史判定（基于已作废的 600 口径，仅留档，不得再引用为结论）

| arm | 判定 |
|---|---|
| A1 | 无效（dino_c .567629、lpips .448064 不过） |
| A2 | 无效（dino_c .743400、lpips .312708 不过） |
| A3 | 无效（四条全不过） |
| A4 | 无效（四条全不过） |

