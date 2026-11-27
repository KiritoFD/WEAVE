# 03 · RTX 4090 全权实验执行预注册方案与论文升级路线图

本文档规定在 `RTX 4090 (24GB)` 上实施 `exp/v19/` 的完整分阶段流水线、门控判据、零样本跨域实验、定性原图导出规范以及 [`docs/icme2027/`](../icme2027/) 论文回填方案。

---

## 一、实验硬件、数据隔离与红线契约

1. **硬件与运行规范**：
   - 目标机器：`ssh 4090`（`RTX 4090 24GB`，仓库路径 `/root/Workspace/WEAVE`，15,545 规范化池 `/root/Workspace/pseudo_gt_24h/orig_mean_fp16` 已就绪并通过校验）；
   - 严格遵循用户铁律：**绝不在本地 4070 跑重负载**；远程**不设 `expandable_segments`**；全程在 `tmux` 会话 `weave-v19` 中运行独立子进程流水线（生成与 `measure.py` 四裁判严格分进程隔离）。
   - 全仓路径可移植性：所有新增代码位于 `exp/v19/` 与 `tests/v19/`，使用 `paths.py` 与相对路径，提交前必须通过 `python -m pytest tests/test_no_absolute_paths.py`。

2. **四道防泄漏铁闸（Audit Firewall）**：
   - **数据铁闸**：`V19` 的全部训练组件（$\phi_\eta$ 风格纤维编码器、`VGG-16` 多尺度判别头 $F_\psi$、潜空间亮度轴 $\hat{w}_Y$、类色度矩先验、4,000 对蒸馏轨迹）**100% 仅从剔除了 `tasks_750` 与 `references_v1` 的 4,400 张训练池采样**；超参/先验偏置仅在 300 张独立 `holdout` 上标定。
   - **裁判铁闸**：训练与教师求解全过程**零导入** `CSD-ViT-L`、`CLIP-ViT-B/32`、`LPIPS-Alex`；`DINOv2-small` 仅在缓存阶段对**源图 $x_s$** 静态提取一次无梯度输入特征 $(\tilde{P}_{16}(x_s), \Pi_\parallel(x_s))$，绝不对任何生成图求导。
   - **免源标签铁闸**：主推实验臂（`V19_T_pro` 与 `V19_S_prism1`）在推理 Test750 的 Cross600 跨风格任务时，**严禁读取 `task["source_style_id"]`**！（对于 Same150 恒等控制，同时报告①协议级同标签门控结果 与 ②完全盲测连是否同画派都不告知的纯连续弦结果，做到极度透明诚实！）
   - **因果铁闸**：最优单步与多步臂必须同步运行固定随机种子的**反事实错标签控制（`Wrong-ID`）**，实测 $\Delta_{\mathrm{causal}} = \mathrm{Top1}(y_t) - \mathrm{Top1}(y_{\mathrm{wrong}})$。

---

## 二、4090 分阶段实验矩阵与时间预算（总计约 95 分钟）

| 阶段 | 模块 / 实验臂 | 核心验证目标与超参设计 | 预估耗时 (4090) | 晋级 / 判据门槛 |
|---|---|---|---|---|
| **Stage 0** | `canary` + `fiber_cache` | 验证可微 `S10` 梯度有限性、潜色度正交投影 $\langle P_Y z, P_C z\rangle=0$、提取 4,400 train + 300 holdout + 150 test 的 $\Pi_\parallel(x)$ (795D) 并训练连续风格编码器 $\phi_\eta$ | ~6 min | Canary 全绿；$\phi_\eta$ 在 300 holdout 上 5 分类准确率 $\ge 78\%$ |
| **Stage 1** | `train_rgb_critic` (**Gate-1**) | 在 4,400 训练图（原图 + `S10` 往返图，1:1:1:1:1 均衡采样）上提取无归一化 `VGG-16` 多尺度一阶/二阶矩 + Oklab 色度统计，训练 $F_\psi$ 并在 300 holdout 校准偏置 $\hat{b}_{\mathrm{ho}}$ | ~8 min | Holdout 总体 Top-1 $\ge 78\%$，且 **`ER` 与 `Ukiyo-e` 单类召回率均 $\ge 68\%$**（对比旧 `R1` 提升 $>20\text{pp}$） |
| **Stage 2** | **多步教师消融矩阵** (`Test750`)<br>• `V19_T1_luma_aniso`（仅换各向异性信任域 + 旧 R1）<br>• `V19_T2_s10_vgg`（可微 `S10` + `VGG` 均衡双向教师 + 各向异性信任域，免 `sid`）<br>• `V19_T3_hybrid_pro`（潜色度预对齐 $z_0^\perp$ + `S10-VGG` + 潜局部方向混合，免 `sid`） | 拆解“各向异性色度放开”与“像素域无 GroupNorm 多尺度判别器”各自的独立净收益，冲击 **`42%–48%+` Cross600 Top-1** | ~22 min<br>（含 3×31s 评测） | 至少一臂 Cross600 Top-1 **`> 40.5%`**（正面压过 `StyTr2` `39.50%`），`Ukiyo-e >= 18%`，`DINO-C >= 0.72`，`LPIPS <= 0.51` |
| **Stage 3** | **离线均衡轨迹合成** (`distill_pairs`) | 用 Stage 2 胜出的最强教师在训练池 800 张均衡源图 × 5 目标上生成 **4,000 对**高质量跨域终点 $(z_0^\perp, z^\star)$ | ~18 min | 无 NaN，轨迹验收率 $>92\%$ |
| **Stage 4** | **免 `sid` 纯单步学生消融** (`Test750`)<br>• `V19_S1_global_chord`（$\Pi_\parallel$ 免 `sid` 连续弦 + 全局 AdaGN）<br>• `V19_S2_spatial_prism1`（+ 潜色度预对齐 + $\tilde{P}_{16}$ 语义路由空间弦 + 膨胀感受野 + `S10` 矩对齐损失）<br>• `V19_S2_wrong`（反事实错标签控制）<br>• `V19_S2_exemplar`（单参考图 $\phi(\Pi_\parallel(x_{\mathrm{ref}}))$ 零类别表模式） | 检验纯单步（`<30 ms/img`）能否在**完全不读 `sid`** 的前提下突破 **`38%–42%+` Top-1**，并验证单张参考图连续弦泛化 | ~25 min | `V19_S2` Top-1 **`> 37.5%`**（纯单步超越商业 `Seedream 4.5`！），$\Delta_{\mathrm{causal}} > +33\text{pp}$ |
| **Stage 5** | **跨域零样本验证 & 高清图库导出**<br>• 自然照片集 / 未见画派零样本评测（若 4090 数据就绪则跑 `R5/P2A` 子集，或跨参考池盲测）<br>• 导出最强多步与单步臂完整 750 张 PNG 打包回传 | 为论文提供无可辩驳的泛化证据与高清定性对比拼版素材 | ~12 min | 750 PNG 完整无损回传至 `docs/performance/images/` |

---

## 三、应急预案（Pre-Registered Fallback Branches）

科学实验必须预留清晰的容错分支，确保单轮实验必出决定性成果：
1. **若 `VGG-16` 像素梯度在 `S10` 上出现高频棋盘格纹（Checkerboard Artifacts）**：
   - *机理预防*：在计算 $g(z) = \nabla_z \mathcal{J}_{\mathrm{RGB}}(\mathcal{D}_\star(z))$ 时，同时在 $1.0\times$（$256^2$）与 $0.5\times$（$128^2$ 双线性降采样）两个尺度求梯度均值，并对潜梯度施加极轻的 $3\times 3$ 高斯平滑核 $\mathcal{G}_{\sigma=0.6}$（抑制反卷积步长周期伪影），再进入各向异性 RMS 归一化与可行步长线搜索！
2. **若潜色度完全替换在部分极简画作上引起色彩过饱和**：
   - *机理预防*：潜色度预对齐设置混合系数 $\alpha_C \in \{0.5, 0.75\}$（在 Stage 2 自动对比），既破除源画暗调枷锁，又保留平滑过渡。

---

## 四、实验完成后的论文与台账一体化升级清单

一旦 4090 跑完并回传：
1. **更新全仓权威数据台账**：将 `V19` 全部实测臂并入 `docs/performance/build_metrics.py` $\to$ `metrics.{csv,json}` 与 `MASTER_LEDGER.md`；
2. **制作王牌定性视觉对比大图（`fig5_qualitative_grid.pdf`）**：同源并排展示 `Source`、`StyleID`、`StyTr2`、`Seedream 4.5`、`SaMam`、**`PRISM-1`（单步 `~29ms`）**、**`PRISM-Pro`** 以及 **`PRISM-1 (Wrong-ID)`** 在五大画派（尤其突出浮世绘与文艺复兴的跨越！）上的高清生成图与局部笔触放大框；
3. **无缝升级 `docs/icme2027/`**：
   - 将第三节升级为更优雅的**正交商—纤维对偶 $(\Pi_\perp, \Pi_\parallel)$ + 各向异性三算子分解**（彻底消除“测试期需要 `sid`”的审稿死角，并让 `PRISM-Dec` 同时承担离线像素级轨迹蒸馏与在线极速渲染双重使命）；
   - 更新 `macros.tex`、`Table I–III` 与 `fig1–fig5`，重新渲染逐页 PNG 做最终视觉验收；
   - 执行清晰的分层 Git 提交。
