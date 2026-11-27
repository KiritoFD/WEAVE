# Checkpoint 与运行资产清理清单

> 2026-11-27｜基于 main@0e314629 及 V35 专家简报。本文是**执行前清单，不是删除脚本**。Git 不含多数本地/远程权重，无法从仓库确认磁盘真实占用、当前活动任务与所有依赖。不得按 Vxx 目录批量删除。

## A. 热保留：最终部署不可缺少

| 资产 | 路径（仓库相对） | 依据 |
|---|---|---|
| P-Core Final EMA | runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt | 170 MB；SHA256 50BCBB4FBC3DAF527A0349D2E68B604914C216FAE3AC9746B2A49EFB3C36AD4C |
| E-Core Original EMA | runs/v32_nodino/ecore/ECore_NoDINO/ECore_NoDINO_final.pt | 42.5 MB；SHA256 9E226F6F8D1FC82F5AAF4D0F74954447008355F58B50080171FC89F4283535DC；含 raw_stats/priors |
| A0 Encoder | runs/a0/a0_encoder_only.pt | 130.4 MB，含 quant_conv；不可用 Decoder 替代 |
| S10B 主权重 | deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt | 40.7 MB；SHA256 AC0BF25007D28B2234AA6AFCFEAB3DD35242F5AEF69F9ED49194A9A2BA83A58D |
| ConvNeXt | Torch hub checkpoints/convnext_tiny-983f1562.pth | Stage3 截断；保存 torchvision 版本与原始 hash |
| K8 银行 | runs/v32_nodino/cache/k8_cnx417_bank.pt | 5.3 MB；sub_scene/sub_raw417 |
| A3 先验 | runs/v34_frontier/cache/v34_analytic_priors.pt | P_Y/P_C/u_y/mu_low_1d/std_low_1d/mu_c/std_c |
| Stiefel 基 | runs/v32_nodino/cache/cnx_stiefel_cache.pt | 必须保留 W_stiefel_cnx；当前 loader 仍读完整 1.03GB 缓存 |
| Y16 投影 | runs/v18/cache/ccf_pool_cache.pt | 必须保留 luma_proj.w/b；当前 loader 仍读完整约1.4GB 缓存 |

以上 hash 引自 [专家简报](docs/archive/historical/v35/EXPERT_OPTIMIZATION_DOSSIER.md)，不是本次独立磁盘校验。冻结配置：P-Core step3500 EMA、A3=(1,1)、ws_mul=1、wc_mul=1.2、dtau0=-.05、steps=6。必须保存完整配置与源码依赖。P 权重仅 EMA，不是可精确续训 checkpoint。

**E-Core 教师血缘同样重要。** V22 clean teacher pool、生成监督缓存、数据索引、V28 Adv1NFE 配方、V32 E 训练配置至少保留一份可复现冷档。不能因为目前推理不加载教师就删除其唯一副本。

## B. 论文与机制复现：冷保留，不永久删除

| 资产组 | 保留粒度 |
|---|---|
| V32 22M P-Core | 原基线 EMA、配置、K8/PCA 缓存与评测输出；用作 Lite、扩容及 Pareto 对照 |
| V34 Stage C | C0/C1/C2/C3 各最终权重、初始化/seed/配方；A0–A3 共享 C2 无需复制权重 |
| V35 P3/P4 | Q0/Q1/Q2/Q3、R0/R1 各最终权重与缓存；Q0 若与 C2 字节相同可去重；Q3 已知路径 runs/v35_final/p3/P3_Q3_stiefel_nopc/P3_Q3_stiefel_nopc_final.pt |
| V35 P5 | 4500/5500/7000 的 online/EMA 若仍存在，各一份与续训日志；不得把重启 optimizer 的轨迹称自然续训 |
| V35 P9e | Unified58M 最终权重、训练配置、共享主干对照数据 |
| Decoder D0/D1/D2 | 原 SD-VAE、WikiArt-FT、S10B 各一份原始权重与 PQC 契约；D0/D1 仅作消融，不用于默认出图 |
| 历史科学证据 | V15 scaling 两端、V22 目标锚对照、V28 一致教师/Adv 前后、S10 Taylor 剪枝与恢复等支持论文的 checkpoint |
| 固定评测 | test750 / holdout / train_calib 清单、原图身份、参考图、裁判权重、逐任务 JSON、生成 latent、关键固定图片；评测资产不可按训练缓存处理 |
| 活跃 One-shot | 3060 上正在训练的全量 WikiArt 版本、最新及上一个完整恢复点、optimizer/EMA/scaler/RNG/sampler/manifest；旧 O0–O3 作为对照冷存 |

路径未列明的组以 run config、manifest 与真实文件扫描为准；不要猜测文件名。V35 最终报告和归档 V34 报告是索引，不等同于磁盘清单。

## C. 可归档候选（不是立即删除名单）

- V6–V18 非论文代表臂、V24–V26 已淘汰局部 Value / UOT / GCL / 弱教师、V30 区域损失、V31 EditFlow / NeutralBridge、V33 Q64+ / Adapter、V34 连续条件 / S2 / bridge 失败臂：先保留最终模型、关键对照及完整日志，冷存其余中间 checkpoint。
- 同一 run 的过密 step 快照：只在保留最终、最优（按原选择协议）、关键消融点、最近两个可恢复点后归档其余。
- 重复 S10 standalone、同权重不同强度/CFG 名称的副本：仅以 SHA256 或逐张量身份验证去重，保留加载接口与恢复映射；同拓扑不等于同权重。
- 旧 ONNX / TRT engine、编译缓存、临时输出、重复 latent 缓存：可再生成时才可清理；engine 绑定 GPU/版本/shape/权重，保留现行部署的构建配置与质量验收结果。
- Git 已追踪 docs/baseline/ours/checkpoints 下的 7 个早期二进制基线不在本轮物理清理范围；保留基线复现，另议 Git/LFS 策略。

## D. 先打包再清理

建议新增 deploy_bundle_v35/，从上述热资产提取必需张量。打包、校验、独立目录运行通过之前，**不移动 ccf_pool_cache.pt / cnx_stiefel_cache.pt**。当前 loader 依赖完整文件；仅复制小张量不会自动令旧 loader 可用。

包内至少含 P/E EMA、A0、S10/PQC、ConvNeXt-S3、W_stiefel、luma_proj、bank、A3 priors、E raw_stats/priors、调度 JSON、版本与 SHA256 manifest。保持缩放 0.18215 恰好一次、PQC 恰好一次。记录来源 blob/checkpoint/hash、形状、dtype、参数与实际文件大小；不要为节省空间先改 dtype。

## E. 执行顺序（4090 / 3060 / 本地各自执行）

1. 只读 inventory：文件路径、大小、mtime、SHA256、硬链接/符号链接、owner run、当前进程、最后使用与依赖。Windows junction 不递归越界；不把同一盘别名算成两份数据。
2. 根据源码、配置、shell/tmux 任务、运行日志与 manifest 建依赖闭包；未能判定的文件标 UNKNOWN，保留。
3. 先复制到独立冷存目标，再校验 hash、总数量、总字节与可读性。生成 source→archive 映射及恢复命令。不同磁盘之间禁止用未经校验的 move 代替 copy+verify。
4. 从独立部署包运行 E-only、E→P、s=0、固定噪声及完整 test750；从冷档随机恢复一个消融 run 验证。
5. 人工审阅清单后才允许释放原副本。保留至少一个独立备份；不删原始 WikiArt、唯一裁判、唯一教师、活动训练目录，不使用递归通配符清空 runs。

只读扫描示例（Linux/WSL，从仓库根执行）：

~~~bash
find runs deliver/checkpoints -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name '*.ckpt' -o -name '*.engine' -o -name '*.onnx' \) -printf '%s\t%p\n' | sort -nr
~~~

PowerShell 示例：

~~~powershell
Get-ChildItem runs,deliver/checkpoints -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Extension -in '.pt','.pth','.safetensors','.ckpt','.engine','.onnx' } |
  Sort-Object Length -Descending | Select-Object FullName,Length,LastWriteTime
~~~

扫描结果不得直接接 Remove-Item/rm；hash 计算和归档前复查活动进程。本文未执行任何磁盘删除或 checkpoint 移动。
