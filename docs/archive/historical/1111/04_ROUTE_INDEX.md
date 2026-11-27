# 全路线总索引

34条WEAVE路线 + 20条VAE/几何/Decoder路线。路线按研究假设合并同系列臂，不把每个seed、checkpoint或超参值冒充独立机制。
各章保留原始表；附录arm_ledger按来源表保留所有已纳入原表的行，解决“正文概括后遗漏具体配置”的问题。

| ID | 路线 | 侧 | 状态 | 主要测量/结果 | 详细页 |
|---|---|---|---|---|---|
| W01 | 论文模型重现、逐 epoch 选择与端点轴 | 3060 | 完成 / 历史锚点 | fresh epoch6 DINO-S 0.4867、CLIP-S 0.7074、LPIPS 0.2508、DINO-C 0.8280；接近论文 epoch10。 | [正文](weave/01_early_model_and_v4_v5.md) |
| W02 | Target-HF 增量、子带、空间与记忆支路 | 3060 | 多轮完成 / 多数未形成稳定主线 | subband 约 DINO-S 0.488624 / DINO-C 0.798123；spatial HF 0.490074 但 DINO-C 0.404308，内容崩塌；禁用 memory 下降。 | [正文](weave/01_early_model_and_v4_v5.md) |
| W03 | 12h 受控过拟合、监督/容量/轨迹探针 | 3060 | 完成部分 / 协议修正后重新判读 | 历史 750 锚点 DINO-S 0.49023、DINO-C 0.87962、CLIP-S 0.70463、LPIPS 0.26203；原晋级门槛 0.50523 只属于该口径。 | [正文](weave/01_early_model_and_v4_v5.md) |
| W04 | V4：DINO 内容匹配、局部拼贴与频带伪 GT | 3060 | 审计和代表臂完成 / 部分后继未执行 | D_P5 位移能量约 93.34% 在 LL；名义 25% 的 128px 覆盖实际约 23.40%；A_P5 目标风格可达保留率 53.8%，E7 87.2%。 | [正文](weave/01_early_model_and_v4_v5.md) |
| W05 | V4 S1/S2 与真实执行边界 | 3060 | S1 完成；S2 见历史记录；S3/S4a 当时未启动 | S1 cross600 DINO-S 0.37465、DINO-C 0.86207、CLIP-S 0.66408、LPIPS 0.25148、CSD ΔT +0.01565；四门槛只过 LPIPS。 | [正文](weave/01_early_model_and_v4_v5.md) |
| W06 | V5：逐图 latent Oracle 与真实风格编码器 B1 | 3060 | 完成 / 按决策树未进入 Direct/ODE 摊销分支 | O 系列 ΔT 大多在 ±0.01；最大 ΔM 来自大量去除源风格；B1 unconstrained ΔT +0.00868，强结构 +0.00189。 | [正文](weave/01_early_model_and_v4_v5.md) |
| W07 | V5 Route C：带噪声的条件 Rectified Flow | 3060 | 完整扫描及双种子最终评测完成 / 未过总门槛 | cross600 双 seed ΔT +0.02890、DINO-S 0.41624、DINO-C 0.80933、CLIP-S 0.68277、LPIPS 0.22004；same ΔS −0.14273。 | [正文](weave/01_early_model_and_v4_v5.md) |
| W08 | C1 内容规范化与真实自配对 Direct | 3060 | 完成 | D1 full 双 seed ΔT +0.02969；α=.70 后 ΔT +0.02139、DINO-C .8777、LPIPS .2507。 | [正文](weave/02_v6_direct_scorediff_ablation.md) |
| W09 | D1 与 F1 条件 Flow 对照 | 3060 | 完成 | D1 1.88M、训练约572s；F1 2.55M、约939s，ΔT 约+.0152；D1 该设置更优。 | [正文](weave/02_v6_direct_scorediff_ablation.md) |
| W10 | 条件 Score Difference | 3060 | 完成 / 保留参照 | 保守 lr=.048/.055 的 CSD Top-1 约4–5%；放宽到N24 lr=.100或N12 lr=.160，目标命中率大增但内容下降。 | [正文](weave/02_v6_direct_scorediff_ablation.md) |
| W11 | Style-ID、规范化、内容不变性减法 | 3060 | 完成 | N1 ΔT −.0174；N2 +.0235；N5 +.0170 / DINO-C .8916；辅助B1仅占风格损失 .048%。 | [正文](weave/02_v6_direct_scorediff_ablation.md) |
| W12 | 容量、步数、统计参考池与损失剂量 | 3060 | 完成 | L11 217,824参数，较1,881,792减少88.4%；Val ΔT+.0421、DINO-C .9216、LPIPS .3403，但Top-1 .030。 | [正文](weave/02_v6_direct_scorediff_ablation.md) |
| W13 | 多尺度注入与 Style Keys | 3060 | 完成 / 收益未稳定复现 | 单关各层差异很小、全关注入有害；style keys 在dim32约+.0022，dim48约+.0011，目标Top-1仍低。 | [正文](weave/02_v6_direct_scorediff_ablation.md) |
| W14 | 被取消的加法/乘法分支 | 3060 | 预实验否决 / 延后 / 未执行 | HF定向特征5类识别 .503，DWT矩 .555，距离矩阵相关 .921；未支持额外HF监督。 | [正文](weave/02_v6_direct_scorediff_ablation.md) |
| W15 | V7 Tweedie 单步/少步 | 3060 | 扫描完成 | 最高Top-1 .305 时 DINO-C .4153、LPIPS .7490；保守点保内容但Top-1明显下降。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W16 | V7 纯 MSE 条件 ScoreNet | 3060 | 训练与诊断完成 | 各σ MSE降低；Ukiyo wrong−correct仍−.00174，正确率27.5%。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W17 | V7 RDMD、Bounded、Hybrid、Stable | 3060 | 完成 / 未达统一验收 | Pure RDMD DINO-C .0804；Hybrid .1113；Stable .906–.965但Top-1 .8–2.3%。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W18 | V7 DMD 符号修正 | 3060 | 修复后重测完成 | pure Top-1 4.0% / DINO-C .8297；content 1.8% / .9575。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W19 | V8 去 Score 的概率流与 Content-OT | 3060 | 完成与口径修订 | B2_raw ΔT+.0578、Top-1 2.7%、DINO-C .8949；EMA更近No-op，更多步未形成明显收益。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W20 | V9 双域 Blog-Style 布朗桥 + DCAE | 3060 | 完成 | OT .5 @4000 Top-1 31.67%；到8000降至10%；noOT最佳23.33%。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W21 | V9 OT开销与容量定档 | 3060 | profiling与对照完成 | 原配置85%耗时在OT；7.45M达到32.6M的最佳Top-1 23.33%，2000步约2.8min；1.89M 15%。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W22 | V10 五域40臂 suite | 3060 | 38臂实现 / 2未实现；完整评测仅部分 | 38/40可执行；13矩阵EDM、36三域路径一致性未实现。六组代表完整对照Top-1 .060–.110、20方向meanΔT全负。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W23 | V10 04L/04M长训练与混合Batch | 3060 | 完成 | Top-1 .11→.13，ΔT−.0347→−.0307；Rococo仍0；方向采样20项各5%，条件响应明确。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W24 | V11 修正梯度后的SWD/在线桥 | 3060 | 四臂完成 | d1/d2 Top-1 .11/.10；e1 .045；e2 .13但DINO-C .6457，ΔT均负。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W25 | V11 EDM解析项与残差缩放 | 3060 | 纯推理对照完成 | Top-1 .055→.10、DINO-C .7626→.6974，ΔT−.0404→−.0351；SWD/MMD反而升。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W26 | V11 高斯pivot无条件生成 | 3060 | 完成 / 三个困难目标未建立 | style Top-1 ER/Roc/Uki均0，Imp .41、Min .67，总体.216。 | [正文](weave/03_v7_to_v11_distribution_routes.md) |
| W27 | R1判别器与独立梯度门控 | 3060 | 完成 | balanced Top-1 .875，最低recall .814；RMS .25探针CSD Top-1 .328 / DINO-C .818。 | [正文](weave/04_r1_oracle_distill_r2.md) |
| W28 | G0/G1 ranking、种子复现、Q2 embedding/trust | 3060 | 完成 / 未过Top-1主门槛 | G0 Top-1 .050，G1b .060，Q2a .065；trust降至.010。ΔT差可复现，Top-1增益CI触0。 | [正文](weave/04_r1_oracle_distill_r2.md) |
| W29 | 逐图Oracle一步/动态四步/冻结四步 | 3060 | 完成 | 原P0 Top-1 .067→.365；四步动态.242；固定四步与一步逐位相同。 | [正文](weave/04_r1_oracle_distill_r2.md) |
| W30 | P1 算子蒸馏：权重/容量对照 | 3060 | 完成 / 有明确增益但付内容代价 | Val B3/B4 Top-1 .190/.205 vs B0 .060；新RGB/BF16 cross600为.175/.2133，DINO-C .8325/.8199。 | [正文](weave/04_r1_oracle_distill_r2.md) |
| W31 | R2 P0/P1：统一评价与编辑幅度/频带 | 3060 | 完成 | Q2a .060→Oracle .343 CSD Top-1；DINO-S .3988→.4277；DINO-C .8873→.7303；R1与内容负梯度cos≈−.0295。 | [正文](weave/04_r1_oracle_distill_r2.md) |
| W32 | R2 P2：内容切向投影、LL教师、残差锚点 | 3060 | 完成 / 投影无独立收益 | 投影前后Top-1 .330/.330、DINO-C .7293/.7295；LL .245/.7883；source锚点 .040/.9195。 | [正文](weave/04_r1_oracle_distill_r2.md) |
| W33 | R2 P3/P4：新教师蒸馏与最终门槛 | 3060 | 四学生与拟合诊断完成 / 主门槛未过 | S1/S2/S4 DINO-S约.4065–.4082，低于B3 .4183；S3 .4194，差值CI含0；final_verdict=teacher_failure。 | [正文](weave/04_r1_oracle_distill_r2.md) |
| W34 | 历史复现、旧R5与实验考古 | 本地 / 3060 | 历史整理完成；本次不重跑 | 已有考古总表与独立数据集索引；本次纳入来源索引而不对每个已清理产物重新验真。 | [正文](weave/05_archaeology_and_unexecuted_routes.md) |
| V01 | A0：艺术数据 rec+KL 后训练 | 4090 | 完成 | LPIPS 0.0822→0.0762，PSNR 24.57→25.37，SSIM 0.7484→0.7739；CSD Keep 0.9322→0.9136。 | [正文](vae/01_art_vae_a0_a1_a2.md) |
| V02 | A1：类别交叉熵监督 | 4090 | 完成 / 未形成总体几何收益 | LPIPS 0.0766；浅 CNN 15.69%（SD0 17.52%）；留画家 14.29%（SD0 14.49%）；某线性统计未见画家探针 14.91%（SD0 12.42%）。 | [正文](vae/01_art_vae_a0_a1_a2.md) |
| V03 | A2：MoCo / SupCon | 4090 | 完成 / 未形成总体几何收益 | LPIPS 0.0763、PSNR 25.37、SSIM 0.7739；浅 CNN 17.52%，留画家 15.11%；总体几何 AUROC 仍约 54%。 | [正文](vae/01_art_vae_a0_a1_a2.md) |
| V04 | V2 SWD 排序 / Reference Bank | 4090 | 执行至 1800 步 / 路线淘汰 | 同题材 AUROC 约 51.1–51.4%；SWD 与 CSD 图像风格方向梯度余弦 0.0013–0.0028；latent RMS 4.65→3.792，类内/类间距离一起缩小约 42%，相对间隔 0.0716→0.0707。 | [正文](vae/02_geometry_teachers_and_failures.md) |
| V05 | CSD / ALADIN 教师合格性与双头审计 | 4090 | 完成 / CSD 有限合格 | CSD AUROC 78.02% / 74.27% / 60.33%，kNN@1 42.77%；SWD 57.39% / 56.37% / 51.11%；ALADIN 69.45% / 66.06% / 53.61%。CSD 双头 CKA 0.9811、余弦 0.7971、邻域重叠 84.92%。 | [正文](vae/02_geometry_teachers_and_failures.md) |
| V06 | Style-REPA 关系矩阵对齐 | 4090 | 冒烟通过 / 正式短程部分执行 | 日志最后可见 220/500，保存到 step100/200；220 步 L_rec=0.0706、KL=19.99、L_align=0.10229。未发现 step500 或完整四项验收结果。 | [正文](vae/02_geometry_teachers_and_failures.md) |
| V07 | 直接蒸馏风格读出首轮坍塌 | 4090 | 手动停止 @470/2740 | 教师余弦先至约 0.2633，随后后段均值约 0.049；SupCon 继续下降，重建与数值健康度没有同步爆炸。 | [正文](vae/02_geometry_teachers_and_failures.md) |
| V08 | 随机低维子空间与梯度分布 | 4090 | 完成 / 有限阴性 | 能量占比约 4.66%，接近维数比 768/16384=4.69%。 | [正文](vae/02_geometry_teachers_and_failures.md) |
| V09 | ABC 诊断：可读性、稳定方向与实际编辑 | 4090 | 完成 / 编辑未通过 | 线性约 2.39/4.25%，小 CNN 7.86/7.37%；content-matched AUROC 53.76/53.39；独立质心方向余弦约 0.626；实际方向 CSD ΔT 0.0034，随机 0.0159，图像 CSD 梯度 0.1706。 | [正文](vae/02_geometry_teachers_and_failures.md) |
| V10 | 线性 Random / PLS / Fisher 子空间与可逆 Flow | 4090 | 完成 500 步 / 困难判别未过 | 线性 content-matched AUROC 50.4–51.9%；Flow 教师余弦 0.4212/0.4280，跨画家 57.10/58.06%，content-matched 51.00% [48.41,53.57] / 52.25% [49.72,54.93]；逆变换相对误差约 1.8e−5。 | [正文](vae/02_geometry_teachers_and_failures.md) |
| V11 | S10 一次剪枝与 L1 容量恢复 | 4090 | 完成 / 早期知觉质量不足 | 一次剪枝 L1 0.5568；1000 步 L1 到 0.06586，但 LPIPS@512 约 0.390 / 诊断版本 0.3676，CSD Keep 约 0.608/0.6205。 | [正文](vae/03_s10_decoder_pruning_and_recovery.md) |
| V12 | LPIPS 分辨率修正与 clean 全量续训 | 4090 | 完成 / 平台早停 | 最终256图 LPIPS 0.09264（A0 0.07837）、PSNR26.82837（27.20875）、CSD Keep0.85346（0.91550）；两次续训 holdout CSD 约0.871。 | [正文](vae/03_s10_decoder_pruning_and_recovery.md) |
| V13 | S10 效率与输入梯度代理 | 4090 | 完成 / 推理段加速成立 | 49.490M→10.658M（−78.46%）；forward44.61→15.57ms（2.86×），forward+input-grad120.51→39.90ms（3.02×）；单探针梯度余弦0.1946。 | [正文](vae/03_s10_decoder_pruning_and_recovery.md) |
| V14 | A_joint：联合重建恢复 | 4090 | 完成400步 | 128图 A0 LPIPS0.06789，A_joint0.07805；B/C约0.0745/0.0742。后来同150图链路 C_edit0.07605优于S10 0.08419。 | [正文](vae/04_joint_style_vae_abc.md) |
| V15 | B_style：风格统计组织 | 4090 | 完成1500步 / 编辑门槛失败 | 完整latent探针13→15%；8维统计11.5→14.12%；跨内容替换Top1 A0 19.67%、A19.67%、B18.8%、C19.0%；判官自身61.5%。 | [正文](vae/04_joint_style_vae_abc.md) |
| V16 | C_edit：显式编辑目标 | 4090 | 记录到1120步 / best元数据upd1000 / 评测完成 | 训练edit_style约0.386→0.10，SupCon2.76→0.90；独立跨内容替换Top1仍19.0%；后来5个WEAVE生成器整套替换均降低CSD Top1。 | [正文](vae/04_joint_style_vae_abc.md) |
| V17 | 容量上限探针：完整latent插值 | 4090 | 完成 / 解释范围受限 | crossTop1 11.25%→12.25%→20.0%→39.5%→60.0%。 | [正文](vae/04_joint_style_vae_abc.md) |
| V18 | 通用 ArtEncoder Stage A | 4090 | 完成500步 / Gate A未过 / B、C跳过 | ref_style4.23647→4.22624（降0.01022）；AUROC0.4535→0.4566（+0.0031，低于要求+0.005）；LPIPS相对S10降0.0009；Gate=false。 | [正文](vae/05_art_encoder_and_local_readers.md) |
| V19 | 局部DeepSets风格读取器 | 4090 | 正式1500步完成 / Gate R未过 | AUROC0.4645、PD0.4651、5shot0.0353（本轮flat0.0912）、10shot0.0353、unseen0.1395、linearprobe0.0872；Gate=false。 | [正文](vae/05_art_encoder_and_local_readers.md) |
| V20 | CNN读取器容量/空间对照 | 4090 | 正式1500步完成 / 未突破 | 最终回传AUROC0.4770、PD0.4788、5shot0.0529、10shot0.0412、unseen0.1395、linearprobe0.0765；本轮flat0.0912。 | [正文](vae/05_art_encoder_and_local_readers.md) |

## 状态怎么读

完成≠成功；失败指特定预设门槛/已测目标。部分执行≠完整失败；跳过≠训练后阴性；实现/冒烟≠完整同预算评测。
“未见结果”表示本次证据不足，不断言产物永远不存在。REPA500、StyleCode+FiLM、新VAE重训WEAVE等均按此规则处理。
