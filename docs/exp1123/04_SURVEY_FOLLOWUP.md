# 4090 综述复核实验

## A．冻结 Stage 1

权重为 `d1_reflow_v2.pt`，每臂四次模型前向，统一 S10、任务顺序和种子。Same150 全部经过网络，指标中的 RGB MAE 以原 latent 的 S10 重建为参照。

| test750 对照 | Cross600 Top-1 | DINO-C | DINO-S | LPIPS | ER / 120 | Uki / 120 | Same150 Top-1 | Same150 RGB MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全局均匀 Euler | 53.17% | .5665 | .5395 | .5936 | 11 | 2 | 65.33% | .0546 |
| 拟合四步时间网格 | 52.50% | .5657 | .5396 | .5936 | 12 | 1 | 66.00% | .0543 |
| Q32 调起点 | 55.17% | .5462 | .5468 | .5970 | 13 | 1 | 60.67% | .0613 |
| Q32 调起点和局部步长 | 54.83% | .5498 | .5453 | .5959 | 12 | 1 | 62.67% | .0601 |

- ER 与 Uki 各有 120 个 Cross600 任务。冻结全局对照合计 13/240。历史记录的 96.67% Same150 使用了输出旁路。
- 固定时间网格拟合是 S4S 启发的简化对照。用 96 个训练样本和三档 γ，拟合同权重 16 步输出。最优节点为 [0, .125, .375, .75, 1]，测试 Top-1 变化 −0.67pp，按来源配对 bootstrap 95% CI 为 [−1.50, 0.00]pp。[S4S](https://proceedings.mlr.press/v267/frankel25a.html)
- 空间控制使用 Q32 局部特征残差作为结构显著性代理。起点调节延长或缩短局部噪声保留。局部时钟臂再重分配四步的空间步长，总积分长度保持 1，模型仍接收全局标量时间。单改起点 Top-1 +2.00pp [+.17, +4.00]pp，DINO-C −.0202。联动相对单改起点 Top-1 −0.33pp [−1.17, +.50]pp，DINO-C +.0036，ER/Uki 合计少命中 1 例。
- 路由用前 24 个测试任务和可精确对齐的合成水平翻转。Oracle 纹理 MAE 为 0，真实 Q32 等边际 MAE .01360，等边际乱序 .02119。精确目标 patch 的 Top-1 为 12.01%，平均分配质量 2.82%。样本包含重复来源，翻转对应只检验受控几何，跨题材对应仍需验证。
- A/B 每画派 16 个来源任务，固定噪声、画派和 γ。对灰度并标准化亮度的同类供体提取 fiber，经 Stage 1 的 `u_vivid` 注入。ER/Uki 的供体 fiber 余弦分别为 .589/.988，输出 RGB MAE .0426/.0028，灰度频带变化方向余弦 −.945/−.742。ER 有明显响应，但响应与供体方向相反。Uki 的供体经该编码后接近，难以隔离编码和 transport 的瓶颈。这个探针测的是 Stage 1 的 fiber 干预，训练后的部件分支需要独立复测。

原始数据见 [时间网格、路由及 test750](data/followup/step_a.json) 和 [参考 A/B](data/followup/reference_ab.json)。

在另 24 张不同测试源的翻转实验中，以相同 80.08% 覆盖率比较部分一对一匹配。加入相对位置距离的 Partial Fused Gromov–Wasserstein 把全 patch 精确命中率从 30.08% 提高到 53.69%，保留 patch 的纹理 MAE 从 .00567 降到 .00238。全覆盖 Q32 逐 patch 取最大值的命中率为 10.40%。这支持继续验证几何约束，跨画作部件标注与训练注入增益仍待验证。[Partial-FGW 原始结果](data/followup/partial_fgw.json)

## B．ER/Uki 真实终点三臂

每类 128 张训练画作和 16 张独立画作 holdout，按已有 v18 绘画身份划分及同名 RGB 一一匹配。原画和灰度抗混叠平滑代理分别由同一个 SD1.5 encoder 编码。灰度代理按 96×96 下采样再恢复到 512×512，从代理重新提取 Q32 和亮度条件。

三臂共享去风格输入、噪声、Q32、Stage 1 初始化及 350 次更新，训练 γ 保留两个端点及中间值。γ=0 端点均为同一个内容教师。γ=1 分别用原 P1 加锚点的 Q32 教师在同一批源上重生成、真实同画作终点、以及同画派无固定点置乱后的同一批真实终点。所有训练使用纯噪声起点，评测使用 4 NFE 和 S10，全量跑真实 test750 与去风格 holdout。该试验隔离终点和配对效应，去风格方法采用确定性降采样灰度代理，参考注入仍沿用闭集 Q32。与 OmniStyle2 的生成式去风格配方存在实现差异。[OmniStyle2](https://arxiv.org/abs/2509.05970)

| 终点 | Cross600 Top-1 | DINO-C | DINO-S | LPIPS | ER / 120 | Uki / 120 | Same150 Top-1 | 去风格 holdout latent MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 教师 | 36.67% | .3704 | .5564 | .7190 | 15 | 0 | 42.67% | .7511 |
| 同画作真实对齐 | 49.00% | .3609 | .5694 | .7380 | 2 | 0 | 54.00% | .6377 |
| 同类真实乱配 | 50.67% | .3287 | .5839 | .7882 | 0 | 0 | 55.33% | .7072 |

- 对齐相对教师的 ER/Uki Top-1 为 −5.42pp，来源配对 95% CI [−8.54, −2.50]pp。对齐相对乱配仅多命中 2 个 ER 任务，差异为 +0.83pp [0.00, +2.16]pp。总体 Top-1 对比为 −1.67pp [−3.83, +0.50]pp。
- 对齐相对乱配保住了部分内容，Cross600 DINO-C +.0321、LPIPS −.0502，且去风格 holdout 的 latent MSE 最低。对齐相对教师在 ER/Uki 子集的 DINO-C 同样提高 .0243，但风格命中减少。代理输入去除了约 98% 的局部灰度高频能量，训练/holdout 的灰度 DINO 布局余弦为 .744/.721。三臂在真实 test750 的 DINO-C 均低于采用原空间桥的冻结 Stage 1。该结果支持配对对代理域恢复和内容保留有帮助，当前配方未达到 ER/Uki 的扩训条件。
- 以 k=10 的 CSLS 修正后，对齐和乱配总体分别为 44.00% 和 41.17%，两个真实终点臂的 ER/Uki 仍均为 0/240。
- 新的 Part-Bank Rollout 未启动。揭示的路由弱点先用同覆盖率 Partial-FGW 受控对照验证，既有 Stage 2a v2 队列继续保留。随机正交 SWD、梯度余弦课程和三频带损失留待新训练放行后验证。

[三臂原始数据](data/followup/step_b.json) · [配对区间与 CSLS](data/followup/evaluation_audit.json) · [四组样例](data/followup/endpoint_qual.png)

追加冻结对照只将 Stage 1 推理起点由 α=.55 改为 α=0，权重、真实 test750、Q32、种子、四步积分和 S10 保持一致。Cross600 的 DINO-C 从 .5665 跌至 .1953，DINO-S 从 .5395 跌至 .3513，CSD Top-1 从 53.17% 跌至 32.67%。这证实单独切换纯噪声起点已足以引发严重内容退化。三臂用纯噪声继续训练后 DINO-C 回升到 .3287–.3704，仍明显低于原 Stage 1。真实终点对原起点配方的净效应仍需原起点下的同预算消融。[纯噪声冻结对照](data/followup/pure_noise_control.json)

修正版通过 `endpoints --sharp-bridge` 启动独立三臂。训练与四步评测均恢复 α=.55。去风格源采用原分辨率亮度双边滤波和预设替代色度，逐张要求灰度 DINO 布局余弦 ≥.88、局部灰度高频能量保留 ≥65%。ER/Uki 各 128 张训练图和 16 张留出图。每 120 个训练样本含 18 个恒等样本、40 个 Imp/Min/Roc 教师回放样本与 62 个 ER/Uki 干预样本，三臂共享采样。每臂 350 步，评估真实 test750 时关闭 Same150 旁路。三臂结束后更新原 CSD、CSLS 与按来源配对的审计。

## 评测与投稿记录

- 按训练缓存的两个六步教师配方查 Cross600 留出评测，强风格教师 `A_p1/e2full_d5` 的 ER/Uki 为 24/120 和 5/120，高内容教师 `A_p1/csj_d6h` 为 32/120 和 8/120。Stage 1 是 11/120 和 2/120，教师本身偏弱，Reflow 后进一步损失。Stage 2a v2 s1500 将两项提升到 77/120 和 61/120，同时 ER/Uki 子集 DINO-C 从 .6040 降到 .5158。无旁路实测 Same150 为 124/150（82.67%），DINO-C .7298、LPIPS .3094。原旁路成绩 145/150（96.67%）。[教师与 Stage 2a 审计](data/followup/teacher_and_stage2a_audit.json)

- 原 CSD 主表冻结。附加 CSLS 用固定 150 张真实源图校准原型密度，k=5/10/20。文件名解析初审发现 92 个测试源画家名中 89 个与训练池重叠，只有 12 个 Cross600 任务来自未命中训练画家名单的来源。参考画家与训练池候选重叠 27/30，需用规范化作者元数据复核。当前结果不能支撑 Artist-Disjoint 声明。[评测审计](data/followup/evaluation_audit.json)
- 外部同名论文为 Yoshai 与 Shaked 的《PRISM: Distribution-Gated Flow Matching for Controllable Unpaired Image Translation》。该文以源特征到目标分布的距离构造分布门控。WEAVE 的当前分支使用 Q32 内容条件与四步 Reflow。保留已冻结 ICME 文档，新稿及可编辑返修版引用该文并明确区分方法。[外部 PRISM](https://arxiv.org/abs/2608.06240)

## 复现

从 4090 上的仓库根目录，以 `cu121` 解释器依次运行。输出写入 `runs/v27_mainline/followup/`，原始样本和逐任务评测分别用既有 `raw/` 与 `measured/` 下的 `followup_*` 子目录。

```bash
python -m exp.v27_mainline.followup.frozen
python -m exp.v27_mainline.followup.reference_ab
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m exp.v27_mainline.followup.partial_fgw
python -m exp.v27_mainline.followup.endpoints --encoder "$SD15_VAE_SNAPSHOT"
python -m exp.v27_mainline.followup.audit
```
