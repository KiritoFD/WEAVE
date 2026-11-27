# CLEAVE Final：RTX 3060 12GB 精准 Profile 与无损加速执行方案

> 2026-11-27，基线为 V35 P8/P9。目标是同一最终模型更快，而非再训练、再改变 Pareto 质量前沿。当前 3060 实测仅有旧模型/One-shot，不能把 4090 数字当作 3060 基线。参见 [最终报告](v35/V35_FINAL_REPORT.md)、[源码与资产简报](v35/EXPERT_OPTIMIZATION_DOSSIER.md)、[历史 TRT 对照](vae/08_STAGED_PROFILE_AND_TRT.md)、[cleanup](../cleanup.md)。

## 1. 决策：先纠正测量，再消除冗余计算

4090 已测的 P 路径 B16 分解：A0+Struct 15.79ms（22.6%），CNX+Stiefel 1.00ms（1.4%），E+A3 1.05ms（1.5%），P 39.49ms（56.6%），S10 12.49ms（17.9%），合计 69.82ms/img。此占比仅是 4090 的起始假设，不是 3060 性能预测。3060 历史 S10 同拓扑 TRT 对 PyTorch 解码段约 2.54×，但当时测的是 C0 权重，最终 S10B 必须重新导出并校验。

最高优先级：**精确测量、E 路径真正旁路 P、消除 host sync、实际 CFG/NFE 计数、A0/S10 FP16 TensorRT、P-Core 布局与图捕获**。不把 4 分支直接砍成 2，不把模型重新蒸馏作为优化，不替换 SVD 为近似算法作为默认路径。

Amdahl 上限（只用于估算）：若 P 段占 .566，单独加速两倍，整体上限 1/(.434+.566/2)=1.39×；若同时 A0/Decoder 各加速两倍，按 4090 分解的理想上限约 1.94×。3060 要用自己的阶段占比重算。

## 2. P0：建立真正可信的 benchmark（1h）

### 2.1 源码审计：目前两个硬错误和一个口径缺口

- [common.py](../exp/v35_final/common.py#L521-L615) 的 ledger 用 n_steps*4 推算 P NFE，但 [integrate_v33_trajectory](../exp/v33_semantic_control/models.py#L512-L563) 首步 midpoint 调用 _vel 两次，其后 DPM-2M 分支各一次。名义 6 步是 **7 次批量模型调用 / 28 次单条件等价评估**，加 E=29；这是源码静态推导，运行时还需 hook 证实。不能继续写 24+1。若首步改 Euler，必须作为新的近似求解器独立验收。
- 同一测速函数 B16 循环 collect_parts=True 每段 synchronize；B1 不做此事，因而两者端到端口径不统一。分段用 CUDA events 记录，不在每段同步；完整调用只在计时窗口前后同步。
- 当前 benchmark 使用随机 GPU RGB，模型预加载、3 次 warmup，只测 s=.85，未证明实际生产路由的输入构造与 c64 一致（代码拼接 y64/lcn/y64/y64）。先复用真实 eval/deploy preprocessing，严格比对同一源图、tid、eps 的 z_s/c64/q32/ze/zo/RGB。检查整条 infer 是否有 no_grad/inference_mode，而非仅 relies on frozen weights；历史数字称 resident-model GPU RGB→RGB，不叫 cold boot。

### 2.2 固定环境与输入

3060 Windows WDDM 与 WSL 分开记录，优先利用已有 WSL TRT 工具链。GPU 训练任务与推理 profile 串行，尤其不能与全量 WikiArt One-shot 同时争抢显存。先记录 GPU 型号、驱动、CUDA、torch、cuDNN、TRT、WSL 版本、功耗/温度/时钟、其他进程、系统 RAM/交换。Ampere SM86 支持 FP16/BF16/TF32 Tensor Core，不支持原生 FP8 Tensor Core；不投入 FP8/NVFP4 路线。

~~~bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported())"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv -l 1
~~~

WSL 的 nvidia-smi 部分字段可能不支持，记录 N/A；Windows 额外记录专用/共享 GPU 内存（Task Manager/性能计数器），不能用 torch allocated 代替实际占用。

使用真实固定 150 源图、目标与固定 eps，从 test750 manifest 取确定样本顺序；禁止随机生成图作为唯一性能输入。512px 不变。B=1/2/4/8/12/16，显存不足标 OOM 并停止继续增大；P 的实际 UNet batch 是 4B。各档单独进程，或释放所有旧模型与图池后重新加载。

### 2.3 基准矩阵

| 模式 | 必测档位 | 计时内容 |
|---|---|---|
| E-only | l025/l050/l060/l090 | 实际 E 路由所需的 A0/结构/语义/风格条件、E、S10；无 P 加载/前向（分别报告完整常驻与 E 专用部署） |
| E→P | s060/s085/s100/s150 | 完整编码→E+A3→P→S10，固定 P8 配方 |
| hot cached | 同上 | 已缓存源 latent、Q32、结构；改变目标与强度时重新计算必要分支 |
| host RGB→RGB | E/P 代表点 | 从 pinned CPU RGB 到 CPU RGB，含 H2D/D2H，不含图像文件编解码 |
| process cold | E/P 代表点 | 新进程 import/load/首次出图；区分是否命中 OS/engine cache；不得混入稳态吞吐 |

稳态 warmup≥10 batch 且时钟稳定，至少 50 batch 或 10 秒计时，重复 3 轮；CUDA Event GPU 毫秒、同步 wall clock、p50/p95 batch latency、batch 总毫秒、ms/img、images/s。B16 ms/img 不是单张交互延迟。每段 event 计时另跑一次，不能代替端到端。冷启动至少 3 次独立进程。峰值同时报 allocated/reserved/驱动显存/Windows shared memory，测量前 reset_peak_memory_stats。

Profile 只抓代表点 B1/B8（若 B8 不适合则 B4），使用 torch.profiler CPU+CUDA、record_shapes、profile_memory、NVTX；5 warmup/5 active 即可，trace 不参与测速。若 WSL 支持，用 Nsight Systems 查 cudaMemcpy、cudaDeviceSynchronize、kernel launch gaps、GPU idle，Nsight Compute 只分析耗时最高的 3–5 个 kernel；不可用则不阻塞。输出 top20 op、阶段 time、kernel count、H2D/D2H、CPU launch gap、算子 FLOPs/带宽瓶颈分类。卷积实际 Tensor Core 使用应由 profile 验证，不凭 dtype 猜测。

## 3. P1：零语义改变的 PyTorch 优化（1.5h）

逐项隔离，任何变更必须经过 parity gate（第 7 节）。

1. **同步与调度**：integrator 中 grid 建在 GPU，循环反复 grid[i].item() 会强制同步；用 host 预计算相同 float32 网格/标量（确保数值与旧网格一致），预构造所有 tau/t/flags；消除 bool((tau_v<=0).all()) 的 device sync（对 Python scalar tau 使用静态分支；per-image tau 另测）。缓存固定条件 repeat 与 AdaGN 的 time-independent 投影（仅当代码确实可拆分）。用 profiler 证实同步减少。
2. **channels_last**：仅对 P/A0/S10/E 逐个比较 NCHW 与 NHWC，模型权重与 4D 输入一致；q/latent 几何维度不变。不要把所有 tensor 强行 contiguous；检查 layout thrash。P-Core 双 repeat_interleave 可换等价 nearest upsample，先逐 tensor 验证索引与边界完全一致。
3. **精度**：A0 BF16、P BF16 原口径为对照；测试 FP16 autocast + FP32 累积的 solver、CFG/rescale、A3、SVD 与标准化。ConvNeXt/E 单独比较 FP32/TF32/FP16，S10 保持原 FP16。不要全链路 half() 后做 SVD；色度统计、微小方差、norm 与 CFG rescale 必须稳定。任何 NaN/Inf 或 s=0 变化立即回退。
4. **CFG 精确剪枝**：现行公式 v = ws_j*v_sc + (wc-ws_j)*v_0c + wsp_eff*v_s0 + (1-wc-wsp_eff)*v_00，rescale_phi>0 时仍依赖 v_sc。只在某系数严格为 0 且不参与 rescale 时省略该分支；不假设 unconditional 与 content-only 跨步可复用，因为 z/t 每步变化。比较 concat 4B 与分块 2B+2B；分块是显存策略不必更快。保留四分支默认。
5. **编译与 CUDA Graph**：WSL 先试 torch.compile 对单个 P UNet、A0、S10，而非整条含 SVD/动态 Python 的 pipeline。分别测 default 与 reduce-overhead；记录 compile 首次开销、recompile/graph break、peak reserved。固定 B/shape 的 CUDA Graph 只在移除同步/动态分支后捕获 P UNet 或整段 solver；静态 buffer copy 输入/输出，防止不同图共用残留；不能重复固定 eps。B1/B4/B8 各自缓存，图池占用计入显存。若 compile/graph 无稳定收益，直接回退 eager。

候选只保留在代表点端到端 ≥5% 加速（或独立阶段 ≥15% 且 E2E ≥3%）且通过质量门控者。不要把微基准加速当 E2E 成功。

## 4. P2：TensorRT FP16（3h，最高确定性工程路线）

先 A0 与 S10，最后 P；分段导出不重写算法。复用已有 [TRT 工具目录](../tools/trt/)（历史脚本的 API/shape/权重必须重新适配），不要直接拿 C0 engine 当 S10B。

| Engine | 输入 | 输出 / 关键约束 |
|---|---|---|
| A0 | B×3×512×512 | posterior mean 对应的 4×64×64 latent，缩放 .18215 恰好一次；不导出 sampling/logvar 无用分支（是否可裁剪先验算） |
| S10B | B×4×64×64 scaled latent | / .18215、checkpoint 自带 PQC 恰好一次、RGB clamp/映射与 decode_s10 同口径 |
| P UNet | 4B latent、y16、Q32、tid/u/t、drop masks、gamma/layout | 四分支 velocity；不把 solver 迭代硬编码进首个 engine；保持独立条件与 null 分支 |
| E | 真实 E forward 输入 | 可选，仅 E-only latency 证明有价值时构建 |

导出流程：冻结 eval/inference_mode、真实张量样例、ONNX checker、ORT FP32 对 PyTorch FP32 中间张量；FP16 TRT 对原生产精度 baseline。对目标安装 TRT 实际 API 适配，不照搬旧 explicit-batch/STRONGLY_TYPED flags。固定 512px，先 B1 与实测最佳吞吐 B（例如4或8），P profile 的 batch 为4B；动态 batch 只在静态引擎稳定后再测。引擎构建记录 workspace（建议先1GB，必要时2GB）、tactic、TRT/CUDA/GPU版本、checkpoint hash、输入/输出 dtype/layout、shape profile。关闭不安全的 fallback；INT8/QDQ 不在本轮默认范围。

使用 CUDA tensor device pointers、同 stream enqueue、直接 GPU→GPU 传递，避免每一段 ORT numpy/CPU copy。构建/运行分开，engine cache key 必须含权重/shape/precision/版本。TRT 11.x 与 Windows 绑定差异以环境实测为准；已有 execute_v2/v3 兼容层仅作起点。

候选组合：A0 TRT、S10 TRT、A0+S10 TRT、最佳 PyTorch P vs TRT P；分别测 E/P 完整链路，不能相加不同运行口径的阶段耗时。P 导出不支持动态 mask 或 norm 时保留 PyTorch P，不因编译困难修改模型语义。

## 5. P3：求解器与 CFG 计算量优化（1.5h，可选近似）

P0–P2 都通过后才做。保持模型权重冻结、同噪声，先在 train_calib 上比较候选；test750 只用于冻结后的正式验证，不按测试结果反复调参。

| ID | 唯一改变 | 名义 P 步数 / 实际单条件等价 NFE（预期） |
|---|---|---|
| S0 | 原 midpoint startup + 6-step | 6 / 28 |
| S1 | Euler startup + 6-step | 6 / 24 |
| S2 | 原 startup + 4-step | 4 / 20 |
| S3 | 原 startup + 5-step | 5 / 24 |
| S4 | 精确零系数分支省略（有适用点才做） | 以实际 hook 为准 |

S1 改变积分初始精度，S2/S3 改变时间网格，皆非无损。4→3/2 steps、3/2-branch CFG、跨步预测复用、CFG 蒸馏与 SVD Newton-Schulz 不进入默认队列。若后续仍需要探索，必须独立验证质量及保持正交误差；当前 Q32 仅占 4090 约1.4%，没有必要优先冒险。

每个候选先测 s060/s085/s100/s150 的固定任务；最终候选全量 test750，校验 matched-C 前沿不退化。近似方案仅当速度 ≥15% E2E 且第7节非劣门控通过，作为 fast 档并保留原 exact-quality 路径。不得将低步数结果与原质量模型名称混用。

## 6. P4：部署包、双路径与缓存（1h）

按 [cleanup](../cleanup.md) 从巨型训练缓存提取 W_stiefel、luma_proj、K8、A3 先验，建立独立 bundle loader。验证空缓存目录、无网络下加载；记录实际文件体积而非预估。加载阶段单独 profile，不将去掉 2.4GB I/O 的收益计入稳态 throughput。

- E-only：不初始化 P；若同一服务需要热切换，可 lazy load P 并分别测 first-P latency 与常驻显存。保留 E 的原生强度与 A3 变体；不得因 P 的 s060 质量更高就把所有高内容请求强制送进 P。
- P 路径：缓存 source-only VAE/Q32/结构；改变目标时重算 router/E/A3，改变噪声时不得错误复用生成结果；cache key 含像素 hash、分辨率、权重/预处理版本、dtype、设备。
- 批量不同强度按 E/P、steps、CFG shape 分桶，避免 E 请求被 P 阻塞；交互 B1 与吞吐最佳 batch 是两个不同指标。
- E→P 交界 RMSE=.1935 已知。此次 profile 不修改强度调度；若需要平滑插值，作为独立视觉/质量实验，避免速度优化同时改变风格前沿。

## 7. 数值和质量验收（1.5h，不能省略）

所有候选使用同 source/target/noise/checkpoint/solver（近似候选明确例外）。先做 16 个固定样本的逐层 parity，覆盖五派、same、wrong、s=0、tau=0、强弱档。记录 max/mean abs、RMSE、cos、每通道统计、非有限数、最终 RGB 误差，不能只看平均 PSNR。对纯布局/缓存/同步改动预期相同或浮点舍入级误差，推荐 latent relative RMSE≤1e-5（不满足则逐层定位）；对低精度/TRT 先用固定样本校准容忍度（起始建议 latent relative RMSE≤1e-2、RGB MAE≤2/255，不能将该阈值视为最终质量证据）。任何 s=0 latent identity 或 tau=0 A3 invariant 失效均拒绝。

最终所有入选部署后端必须完整 test750，Cross600+Same150；固定 E l050/l060、P s060/s085/s100/s150。独立裁判生成后另进程测量，禁止训练或逐图选择。配对来源聚类 bootstrap 95% CI，实用非劣阈值：ΔC/S 下界≥−.005，ΔTop1/Same 下界≥−2pp，ΔLPIPS 上界≤+.005；若 CI 不足以证明非劣则保留原质量版并标不确定。另看逐类 ER/Uki、source-lock、wrong-target 以及固定视觉样例。matched-C 只在共同覆盖区插值，不外推。

## 8. 预算、工程交付、停机

| 顺序 | 事项 | 上限 |
|---|---|---:|
| P0 | baseline + profiler + NFE/路由审计 | 1h |
| P1 | PyTorch 无损路径 | 1.5h |
| P2 | TRT A0/S10/P 分段及集成 | 3h |
| P3 | 低 NFE 探针 | 1.5h |
| P4 | bundle/cache/lazy E/P | 1h |
| P5 | 全量质量门控、双路径测速 | 1.5h |
| 机动 | 版本/OOM/engine build | .5h |
| 合计 | 单张3060，串行 | 10h |

预算是计划，不是实测耗时。与 One-shot 长训错峰；时间不够先砍 P3 和 P TRT，再砍 compile 探索，不砍最终验收。建议峰值驱动专用显存≤10GB、无持续 WDDM shared memory paging，留出至少约1GB 系统余量（以真实可用显存为准）。OOM 记录而非静默缩 batch。失败后只恢复原配置，不触发无上限搜索。

建议新增 exp/deploy_v35/（后续实现，不在本 PR 声称已运行）：benchmark.py、export_engines.py、bundle.py、parity.py；生产算法复用现有模块。JSON schema 至少包括 git SHA、asset hashes、backend、precision、B、branch、s、seed、actual_model_calls、condition_equivalent_NFE、wall/event ms、p50/p95、allocated/reserved/driver/shared、温度/功耗、compile/build/load time、quality deltas/CI。输出 docs/v35/data/3060_profile_summary.json 和逐任务证据，原始 trace/engine 不入 Git。测试覆盖 E-only 不调用 P、CFG 分支、SVD 正交、PQC/scale、cache invalidation、actual NFE、zero identity。最终更新 Pareto 的实际 E/P 3060 吞吐，不拿不同硬件倍率替代。

**最终交付是两个真正可部署的速度档，而不是一个新的模型：11M E-only 极速编辑、55.67M E+P 高质量重绘。** 先追求全链路≥1.3×、再评估≥1.5×；这些是相对于新 3060 同协议 baseline 的工程目标，非承诺。若已达到目标且质量不退，停止优化，不启动 encoder 剪枝/蒸馏等新研究。
