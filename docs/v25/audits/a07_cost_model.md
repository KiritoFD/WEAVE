# A07：完整 RGB→RGB 性能归因模型（端到端成本模型）

- 任务：A07（docs/19_pull_together/agent2.md §六），Slot A，2026-11-20
- 工具：`tools/v25_cpu/a07_cost_model.py`（一键复算：`python -m tools.v25_cpu.a07_cost_model`，纯 CPU，`CUDA_VISIBLE_DEVICES=""`，`OMP/MKL/OPENBLAS_NUM_THREADS=2`，不跑 GPU、不 ssh、不写文件）
- 口径约定：**measured** = 有 JSON/台账出处的实测值；**derived** = 由两个以上实测锚点反解；**est** = 估算（必附依据链）。阶段链：`T_total = T_encode + T_feature + T_style + T_anchor + T_flow + T_decode + T_IO`。

## 1. 参数量（live 复算，CPU torch 逐参数求和）

| 引擎 | 参数 (M) | 组成 |
|---|---:|---|
| prism1 | 1.280 | `SpatiallyRoutedChordNet(width=96, chord_dim=128, spatial_router)` 单步 net（台账口径"1.5M"≈ net + StyleFiberEncoder 0.305M） |
| d6_flow | 21.947 | `SLFUNet(base_dim=144, u_dim=128)`（代码 docstring 旧口径 ~20.8M、ledger 记 22M，live=21.95M） |
| hybrid | 23.227 | SLFUNet + P1（双模型串行，权重独立） |
| mecha_a | 23.227 | 同 hybrid 教师口径 + 门控（A0/A1 零参数，A2/A3 tiny） |
| o3_ref | 22.598 | `SLFUNetV24OneShot` 22.227M（含 `RefAttnV24` 0.281M）+ `RefAdapter` 0.066M + `StyleFiberEncoder` 0.305M |
| student (pending) | 1.642 (B0) | P1 基座冻结 1.280M + `GammaResidualStudent` B0 残差 0.362M；B1 = 1.280+4.695 = 5.98M |
| 共享栈 | 10.660 + … | S10 decoder 10.66M（含 post_quant_conv）；SD0-VAE / DINOv2-S / ConvNeXt-T 特征栈 |
| 评测裁判 | ≈477 | CSD ViT-L 303.7 + DINOv2-S 21.7 + CLIP-B32 151.3 + LPIPS-alex 0.6 —— **仅 measure 阶段，不入部署 T_total** |

## 2. NFE vs 实际模型前向（核心表；image-equivalent 口径）

| 引擎 | NFE (vel evals) | CFG 放大 | 流前向 | fresh 总前向 | 协议批总前向 | 结构依据（代码行） |
|---|---:|---:|---:|---:|---:|---|
| prism1 | 1 | ×1 | 1 | 6 | 3 | `exp/v24/p0_m0_m2.py` L31-63（单步 stu 前向 + chroma-shift 锚，无 CFG 无迭代）；`src/v25/config.py` L161-167 |
| d6_flow | **16** | ×2 (cs_j) | 32 | 35 | 33 | `exp/v22/core.py` L28-32（D6/D6H steps=8）、L185-199（`rk2_integrate` 每步 2 次 vel_fn）、L136-151（cs_j `torch.cat([z,z],0)`=×2）；`src/v25/config.py` L168-174 |
| hybrid | 16 | ×4 (full) | 65 | 68 | **1** | `exp/v22/core.py` L153-180（full 分支 `z_cur.repeat(4,...)`=×4）；`exp/v25/r1_inference_ablation.py` L156-158（V24 基座缓存线性混合，协议评测仅 decode） |
| mecha_a | 16 | ×2 (cs_j) | 33 | 36 | **1** | `exp/v25/mechanism_a.py` L560-612（教师 latent 缓存 + `apply_gate` 张量算子；A0/A1 零 NN）；teacher_spec=`src/v25/config.py` L105-113 |
| o3_ref | **20** | ×2 (CFG `cat[zz,zz]`) | 40 | 47 | 42 | `exp/v24/o3_sg25_eval.py` L154-197 计时段（见 §4）；`src/v25/config.py` L198-205、L374（steps=10, wj=2.2） |
| student (pending) | 1 | ×1 | 1 | 6 | 3 | `exp/v25/mechanism_b.py` L165-169、L107-142（单步残差） |

- fresh 分解：prism1/student = enc1 + feat2(DINO ×2) + style1 + flow1 + dec1 = 6；d6/mecha = enc1 + feat1(DINO ×1) + flow + dec1（y16 = latent LCN 算子；**T4 锚 fresh = P1 前向 + canonicalizer，属一次性批量构建，不入每图前向**）；o3 = enc1 + feat4(DINO ×4) + style1 + flow40 + dec1 = 47。
- 协议批分解（test750 臂缓存齐备）：prism1/student = 3（style tiny + flow + dec）；d6 = 33（style = core8 bank 查表 einsum，无 NN）；hybrid/mecha = 1（教师 latent 缓存线性混合/门控，仅 decode）；o3 = 42（flow40 + style tiny + dec；ref 原型缓存）。
- **勘误**：agent2.md 任务文本" d6_flow(cs_j 20NFE)"与代码不符 —— d6 实际 **16 vel evals**（8 步 × RK2 ×2）；"20 NFE" 恰是 **o3_ref** 的 10 步 × 2。NFE 还都不是真实张量工作量：cs_j ×2 / full ×4 的 CFG batch 放大后，image-equivalent 前向为 32 / 65 / 40。

## 3. 实测锚点（全部带出处）

| key | 值 | 出处 |
|---|---|---|
| PRISM-1 @4090 | 17.0 ms/img（58.7 img/s） | docs/performance/MASTER_LEDGER.md（V19_S2）；协议批形态，latent/特征全缓存 |
| PRISM-1 @3060 | ~29 ms/img | AGENTS.md 环境备忘（跨卡换算记录值，非同协议实测） |
| V20 D6 家族 | 76.6–134.1 ms/img（D1=134.1） | MASTER_LEDGER.md（V20.D1_core_slf_hiS）/ AGENTS.md；3060 12GB，训练峰值 4.06GB |
| V21.1Shot | 135.4 ms/img | MASTER_LEDGER.md（vs StyleGallery 30,400 ms，224×） |
| O3 计时段 | **1251.6 / 1245.4 ms**（t40/t50）；峰值显存 **2111.8 MB** | runs/v24/o3_sg25_eval.json `sg25[*].e2e_ms / peak_vram_mb`；3060 batch=1，`torch.cuda.synchronize` 包住生成段（编码→流→S10 解码） |
| O3_label_free 计时段 | **1672.4 / 1529.5 ms**；2135.0 MB | runs/v25/r3_oneshot_minimal.json `label_free_blind.sg25[*]`；docs/v25/GPU3060_RESULTS.md 记 1.53–1.67 s/img |
| T_IO（协议 750 臂） | infer torch.save ~738MB = 1.26s；measure torch.load = 0.4s；eval.json = 0.023s；features.pt = 0.01s；PNG 750 张(8 线程 l1) = 3.23s（仅 write_png=True）；sha256 30MB = 0.015s（B03 hash 开启才有）；register = 39.23ms/次（含 git 探测；设 `WEAVE_GIT_SHA` 后 1.814ms） | docs/v25/audits/b04_io_report.md §2/§3/§4/§7/§8 |
| 单臂评测墙钟基线 | ~50 s/臂；CPU/I/O 占 9.8%（PNG 开）/ 3.4%（PNG 关） | b04_io_report.md §8（总控给定基线） |

a04_strength_curves.json（19 实测臂，families = prism1_lambda / d6_tau_min / hybrid_gamma / o3_tau0）**只有质量指标，无 NFE/延迟字段** —— 本文件首次把延迟锚点与 a04 引擎族对齐。

## 4. o3_ref 计时段 1.25 s/img 的阶段占比反推（est/derived，@3060 batch=1）

计时段组成（o3_sg25_eval.py L158-187，`t0`→`ts.append`）：

| 阶段 | 次数 | 区间 (ms) | 占比 | kind | 依据链 |
|---|---|---|---|---|---|
| **T_flow** | 20 vel evals ×2 batch = 40 img-eq | 1016–1123 | **81.2–90.1%** | derived | 计时段实测 1245.4–1251.6 减固定栈：`(1245.4−229)/1251.6 ~ (1251.6−129)/1245.4`；单价 24–28.1 ms/img-eq（batch1 启动开销主导） |
| T_decode（S10） | 1 | 30–50 | 2.4–4.0% | est | 10.66M decoder 512² batch1，无单测 |
| T_encode（SD0 VAE enc） | 1 | 30–60 | 2.4–4.8% | est | 83.65M 全量的 encoder 半壁（est 34–37M）@512²；L159 |
| T_feature（4× DINO-S + 1× CNX） | 5 | 68–115 | 5.4–9.2% | est | L161/162 whitened ×2 + L164/165 moments ×2（DINO-S 15–25ms/次）+ L163 CNX-s2（8–15ms）；其中 **ref 侧 2×DINO + 1×CNX 可缓存** |
| T_style（fiber+adapter） | 3 tiny | 1–4 | 0.1–0.3% | est | 0.305M+0.066M tiny MLP @795d |
| T_anchor（τ0 桥） | 0 | <1 | ~0% | est | `z=(1−τ0)·ε+τ0·z_src` 张量算子（L169） |
| 固定栈合计 | 7 NN 前向 | 129–229 | ~10–17% | derived | — |

**结论（回答点名问题）**：1.25 s/img 里 **flow 是大头（~83–90%），feature 提取（DINO/CNX/encoder）不是大头（合计 ~10–17%）**。计时段外还有未计时项（L189-197）：PIL PNG save + 裁判 `stack.extract`（3 图：DINO/CSD/CLIP）+ LPIPS + VGG gram ×2 —— 墙钟 > 计时段，故 1.53–1.67 s/img 的 label_free 读数与 1.25 s 不是同一计时口径下的纯净对比。label_free 与 O3 同架构（22.23M）但计时段差 +23%~+34%（反解 forward 单价 33–38.6 vs 25.8–28.1 ms）—— 差异未定因，列入 must-measure M7。

## 5. "20 NFE ≠ 端到端速度" 的量化差异来源（四层放大）

| 层 | 机制 | 量化 |
|---|---|---|
| (1) CFG batch 放大 | NFE 只数 velocity 评估；每次 eval 的 batch 维 ×2（cs_j `cat`）/ ×4（full `repeat`） | o3: 20 NFE → 40 img-eq；hybrid: 16 → 65（含 P1）；d6: 16 → 32 |
| (2) 非 flow 固定栈 | VAE enc 1 + DINO 4 + CNX 1 + S10 1 = 7 前向，与 NFE 无关 | ~129–229 ms/图（est） |
| (3) batch 利用率 | batch=1 启动开销主导 vs 协议批 compute-bound | o3 batch1 单价 24–28.1 ms/img-eq（derived）vs V20 D1 协议批 134.1/64 = **2.10 ms/img-eq** —— **~12× 批化摊销** |
| (4) 计时段外开销 | I/O + 裁判 + PNG | T_IO 2.3–6.6 ms/img（协议摊销）；SG-25 裁判/PNG 未计时 |

综合比值：**o3 vs prism1 @3060 = 1252/29 ≈ 43× 延迟**，而 NFE 比 = 20×、流前向比 = 40×、全栈前向比 = 47/6 ≈ 8× —— NFE 单独既低估流级放大（漏 ×2 CFG）又漏掉固定栈与批化利用率，**不解释端到端**。交叉验证：d6_flow 若按 NFE 直觉应是 PRISM-1 的 16×，实际协议批延迟族锚 76.6–134.1 ms/img vs PRISM-1 ~29 ms @3060 ≈ 2.6–4.6× —— 因为 32 img-eq 在批化下单价仅 ~2.1 ms，而 PRISM-1 的 17/29 ms 是单步整图价。

## 6. 三档口径（cold / warm / 同参考复用）

| 引擎 | cold 首次加载 | warm 热缓存稳态 | 同参考复用特征缓存 |
|---|---|---|---|
| prism1 | est：ckpt ~5.1MB + bundle 读盘 + CUDA ctx；b04 hash 吞吐 2.0GB/s 下权重可忽略，主项 torch/CUDA 初始化（无实测） | **measured**：17.0 ms/img @4090（58.7 img/s）；~29 ms @3060；+T_IO 2.3–6.6 ms/img | 同 warm（协议路径风格弦/锚/特征全缓存，无参考侧增量） |
| d6_flow | est：ckpt ~88MB + T4 锚 750 latent 一次性构建（P1 ×750 + canonicalizer）+ y16/d16 缓存（无实测） | derived：cs_j=32 img-eq ≈ V20 full 一半 → **est 45–80 ms/img**（32×1.2–2.1 + decode 5–10ms + 开销；单价由 D1 134.1/64=2.10 反推）；实测族锚 76.6–134.1（full 口径）；+T_IO | 同 warm（core 引擎协议路径本就消费 kNN bank/原型缓存，无 per-ref 前向） |
| hybrid | est：同 d6 + P1 ckpt + V24 基座缓存 ~738MB 读入 | est：**eval 侧 1 前向/图（仅 decode）**；fresh = 68 前向 → est 85–150 ms/img@3060（无实测臂） | 同 warm |
| mecha_a | est：同 d6 教师口径 + 门控网（A2/A3 tiny）；教师 latent 缓存再生一次性 | est：A0/A1 eval 0 流前向（缓存 dz + `apply_gate`）；A2/A3 + tiny 前向；fresh 教师口径 = 33 img-eq | 同 warm |
| o3_ref | est：权重读盘 OneShot ~89MB + S10 ~43MB + SD0-VAE ~335MB + DINO ~88MB + CNX ~110MB（fp32 估）+ CUDA ctx + 首图 warmup = **est 8–25s**；b04 仅给 hash 吞吐（读+hash ~0.3s），反序列化/初始化无实测 | **measured**：1251.6/1245.4 ms 计时段 @3060 batch=1；label_free 1672.4/1529.5；峰值显存 2111.8/2135.0 MB | est：缓存 ref 侧（L162/163/165：2×DINO + 1×CNX）**省 38–65 ms/img（~3–5%）→ 1180–1214 ms/img**；B02 `feature_cache="reference_v2"` 即此契约 |
| student (pending) | 同 prism1（B0/B1 ckpt 未产出） | est：与 prism1 同级 17–35 ms/img；**ckpt 未产出，禁止写论文** | 同 prism1 |

显存锚点：o3 batch=1 实测 2.1 GB；V20 训练峰值 4.06 GB @3060 12GB；**协议批推理显存（batch25/50）全引擎未测**（M9）。

## 7. T_IO 并入（B04 对接）

`T_IO/臂 = 1.69 s（PNG 关，必付）～ 4.92 s（PNG 开）`，单图摊销 **2.3 / 6.6 ms**；构成（b04 §8）：infer `torch.save` 738MB 1.26s（`src/infer/run.py` L75-77）+ measure `torch.load` 0.4s（`src/eval/measure.py` L62）+ eval.json 0.023s + features.pt 0.01s；条件项：PNG 3.23s（runner 默认 `--no-png`）、sha256 0.015s（B03 hash）、register 39.23ms/次（设 `WEAVE_GIT_SHA` 降 20×）。要点：**T_IO 与模型快慢无关，换更强 GPU 不缩短**；~4MB batch .pt 分片粒度合理，逐任务小文件是反模式（b04 §8.6）。

## 8. 必须实测的最小清单（给 MEASURE/G 队列；估算未替换前不得写入论文）

| ID | 内容 | 验证/替换哪个估算 |
|---|---|---|
| M1 | o3_ref 计时段内部分段计时（VAE enc / 4×DINO+CNX / 20 vel evals / S10 dec），`torch.cuda.synchronize` 包段 | 把 §4 全部 est/derived 区间换成实测 |
| M2 | d6_flow(cs_j, D6H) 协议批 750 任务延迟（ms/img）+ 峰值显存 | §6 warm 的 derived 45–80 ms/img —— 主表引擎唯一缺的直接锚点 |
| M3 | hybrid fresh 路径延迟（68 img-eq 口径） | §6 est 85–150 ms/img |
| M4 | PRISM-1 @3060 同协议批口径重测（现 ~29ms 是换算记录） | 表 5 的 43× 比值分母 |
| M5 | 全引擎 cold 首图延迟（读盘+反序列化+CUDA ctx+首图 warmup） | §6 cold est 8–25s |
| M6 | 同参考复用收益：o3 去 ref 侧 2×DINO+1×CNX 的实测差 | §6 refcache est 38–65 ms（B02 cache 落地依据） |
| M7 | label_free vs O3 计时段差 +0.28–0.42 s/img 定因（同架构不应差 23–34%） | §4 末注；R7 前必须解释 |
| M8 | 评测裁判墙钟（CSD/DINO/CLIP/LPIPS 750 图秒数，现含在 ~50s 臂墙钟内未分离） | 分离推理/评测口径 |
| M9 | prism1/d6/hybrid 协议批推理峰值显存 | 12GB 3060 部署边界（现仅 o3 batch=1 2.1GB、V20 训练 4.06GB） |

## 9. 风险与限制

- 所有 est 区间未含 GPU 实测背书，只用于归因排序与量级判断；写入论文前必须走 §8 清单。
- V20 D1 的 134.1 ms/img 是 full-branch 口径族锚（D1..D7 区间），d6_flow 默认是 cs_j —— derived 折半（45–80 ms/img）未经直接实测。
- "d6_flow(cs_j 20NFE)" 为上游任务文本笔误（实际 16；20 属 o3_ref），已在 §2 勘误，避免进入后续文档。
- b04 的 cold 口径是"首触"而非落盘冲洗；PNG 吞吐为噪声上界（真艺术图更可压缩）。
- 计时段（e2e_ms）不含 PNG+裁判；跨文档比较延迟时必须区分"计时段"与"墙钟"两个口径。
