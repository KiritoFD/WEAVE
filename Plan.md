# WEAVE 下一代实验计划（A / C / D；B 暂停）

> 维护规则：本文件的进度勾选必须与 `Exp.md` 记录一致；每个 run 结束立即更新，不事后补。
> 统一评估口径：600 跨风格为主表，150 同风格为 Identity 补充；参考图策略固定为 `first_image_of_target_style_test_dir`；指标 DINO-S / CLIP-S / DINO-C / LPIPS；权重离线可复现（见下）。
> **目标变更（2026-11-04）**：不再复现论文 0.4916，也不追旧 ckpt；唯一比较对象是本仓库基线（600：DINO-S 0.3977 / CLIP-S 0.6425 / DINO-C 0.7939 / LPIPS 0.3053；150：DINO-S 0.8263）。

## 0.1 SOTA 判定口径（写死，每张表都按它判）

- **有风格收益**：600 跨风格 DINO-S 相对基线 **≥ +0.02** 且 CLIP-S **≥ +0.01**。
- **约束**：DINO-C 不低于基线 −0.01；LPIPS 不高于基线 +0.01。
- **必须拆分**：按目标风格逐项报告（最弱风格不能被均值掩盖；本 split 5 风格为 Early_Renaissance / Impressionism / Minimalism / Rococo / Ukiyo_e，其中 Minimalism 历史最弱），同时报 150 同风格确认 identity 未牺牲。
- **超参充分性**：每条路线至少 3 档核心超参 × 2 seed；单档单次不构成结论。
- **每次评测落盘**：全量 750 图 → gate → registry，且 `results.json` 必须含 `protocol`（split 定义 / 参考图策略 / 指标定义）、`counts`、`backbone`、`per_style`（由 `tools/finalize_results.py` 写入）。
- **吞吐改动**：先跑 200 步对照，确认 loss 轨迹与改动前一致，再放量。
- **早停口径（2026-11-04 起）**：所有新 run 默认 `internal_probe_enabled=true` / `internal_early_stop_enabled=true`；启用早停必须同时构建 `target_latent_hf_subband_delta` 头（否则启动 raise），探针读数恒定直接 raise；`min_steps=2000` 之前不触发。cap32/cap128 现队列不中途改脚本（未武装，两臂可比），队列后另跑 `cap32_es/cap128_es` 一致武装对。registry 记 `early_stop_armed/fired/fired_epoch/rho_series/gate_mean_series/stop_reason`。

## 0. 环境与依赖（已就位）

| 依赖 | 状态 | 路径 |
|---|---|---|
| DINOv2-base | 就位，前向已验证 | `/mnt/i/Latent_Style/weights/dinov2-base` |
| CLIP ViT-L/14 | 已废弃，权重已删除 | — |
| CLIP ViT-B/32 | 当前唯一口径 | `openai/clip-vit-base-patch32` |
| VGG16 | 就位 | `/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth` |
| SD-VAE ft-ema | 就位 | `/mnt/i/Latent_Style/eval_cache/hf/modelscope/...` |
| 离线开关 | `HF_HOME=/mnt/i/Latent_Style/eval_cache/hf`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` |

硬件预算：单卡 RTX 3060 12GB；常驻占用随 Edge/LM Studio 变化——1b 阶段实测仅剩约 6GB，2026-11-04 凌晨实测常驻仅 1.2GB，因此显存上限按 **10GB 预算**定档（训练峰值 9.59GB）。
基线 ckpt：`exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt`（Route A 起点）。

## 1. 阶段与进度

- [x] P0 权重离线就位 + 前向验证
- [x] P1 吞吐爬坡：b1 峰值 6.16GB（0.40s/step）；显存释放后按 ~11GB 预算复测 b2/b4/b8 → **b2 仍 OOM**，训练固定 batch=1
- [x] P1b 输出级监督显存优化（1b 队列结束后）：VGG→dinov2-base、损失分支 384、patch 描述子分块、bf16 AMP → 逐档重测完成，定档 batch=14/microbatch=14（峰值 9.59GB，Route A 监督下 wall ≈8 img/s）
- [x] P2 DINOv2 patch-token 缓存：`content_5k` 5000 样本（5×1000，1.9GB）+ `refs_test` 150 样本（57MB），manifest 已写
- [ ] P3 路线 A 超参扫描——1b 五档（seed 0）已完成并出表（最高 0.3992，未达 +0.02）；λs 0.1/0.3/1.0 × 2 seed 队列已排（`runs/sweep_queue.sh`）
- [ ] P4 路线 C 参考图分支——ref/Style-ID 两臂 × 2 seed 已完成：ref 0.3943 vs 对照 0.3942（噪声级），**无跨风格收益**；nce 0.05/0.3 × 2 seed 队列已排
- [ ] P5 路线 D VAE 探针 + 多尺度纹理模块——探针 `latent_ok`；纹理模块 hidden=32 × 2 seed 已完成：0.3943 vs 对照 0.3942，**无增益**；hidden 16/64 × 2 seed 队列已排
- [ ] P6 基线同口径重评（L/14 CLIP + 统一 split）——进行中（`runs/nextgen_eval_full`）

## 2. 路线 A：输出级（可微）风格监督

**假设**：latent 空间的 flow 目标与感知风格弱相关；把监督搬到"解码后图像"上（VGG 风格/内容 + 自相似），能直接提升感知风格强度而不过度牺牲内容。

**机制**：`bridge.output_supervision_*`，VAE/TAESD 解码 → VGG16 特征 → SWD/Gram 风格损失 + 内容损失；仅在训练时。

**扫描矩阵**（每档训到收敛或明确平台期，每档 ≥2 seed）：

| 维度 | 档位 |
|---|---|
| λ_s | 0.05 / 0.1 / 0.3 |
| λ_c | 0.1 / 0.3 |
| 可微展开步数 | 2 / 4 |
| apply_every | 1 / 2 |

训练预算：从基线 ckpt 后训练；步数按平台期确定（≥1000 步起），batch 由 P1 决定，seed 0/1。

**已知坑（必须写入每个 run 的配置）**：resume 基线 ckpt 会连带恢复 optimizer/scheduler 状态，把 lr 锁死在 1e-5。所有后训练 run 必须设 `training.resume_optimizer=false`（必要时 `resume_training_state=false`），并在日志里核对 `lr=` 实际值。

**round1b 队列（lr=5e-5，2000 步，seed 0）**：`b_ls03_lc01_s2_e1` → `b_ls01_lc01_s2_e1` → `b_ls03_lc03_s2_e1` → `b_ls03_lc01_s4_e1` → `b_ls03_lc01_s2_e2`。
首档结果：600 DINO-S 0.3992（基线 0.3977），DINO-C 0.8071（+0.013），LPIPS 0.2655（−0.040）→ 风格增益不足 +0.03，但内容/LPIPS 明显改善。

**验收**：600 跨风格 DINO-S 相对同口径基线提升 ≥ +0.03 且 DINO-C 下降 ≤ 0.03（或 LPIPS 增幅 ≤ 0.02）→ 成立；仅风格升而内容大幅崩（DINO-C 降 >0.05）→ 不成立。

**依赖**：VGG16、TAESD/SD-VAE 解码器、显存 ≥ 5GB。

## 3. 路线 C：参考图 Style Encoder（DINOv2 patch tokens）

**假设**：现有近半参数花在 Style-ID 记忆表上，跨风格泛化弱；用 DINOv2 patch token 的参考分支 + InfoNCE（同风格不同内容）可替代或补强 Style-ID。

**配置**：`reference_encoder.py` 分支；DINO cache 离线预提取（P2）；对照 = 保留旧 Style-ID 路径 vs 替换/并联。

**验收**：600 跨风格 DINO-S 提升 ≥ +0.03 且 Identity 同风格 DINO-S 下降 ≤ 0.05；或同等风格指标下参数量/显存不增。

**依赖**：P2 DINO cache（dinov2-base，patch tokens 768 维）。

## 4. 路线 D：VAE 重建上限探针 + 多尺度纹理模块

**步骤**：真实画 → VAE 编码 → 解码 → 比较重建前后的纹理/风格指标（DINO-S、LPIPS、VGG 纹理统计）。
若重建上限本身已低于目标风格增益 → latent HH 不值得做，直接放弃该子路线。
否则开小型多尺度纹理模块（`texture_module.py`）做开/关对照。

**验收**：探针给出量化上限；纹理模块开启后 600 跨风格 DINO-S 提升 ≥ +0.02 且 LPIPS 增幅 ≤ 0.02。

## 5. 路线 B：内容锚定风格操作（方案已定、排队待启动）

**状态：已定方案、排队待启动。排在 A/C/D 之后，当前不启动任何 B 训练、不找教师权重。**

**核心思路**：不再用注意力搬运参考图像素，而是让网络学到"风格操作"，在内容图自身结构上构造风格化伪目标，再拿它当 FM 端点。
架构差异必须体现在代码里：旧的是 `iDWT(LL_c, HF_ref)`，新的是 `iDWT(T_LL(LL_c,s), T_HF(HF_c,s))` —— 参考图只提供风格操作，空间结构来自内容图。

### B1 内容锚定交叉注意力（首选）
- Q 来自内容：局部结构（边缘方向、区域平滑度、纹理尺度）；K 来自参考图：局部结构类型（"什么区域适合什么笔触"）；V 来自参考图：颜色、笔触尺度、方向分布、局部纹理变换参数。
- `A_ij = softmax(q_i k_j^T / τ)`，`s_i = Σ_j A_ij v_j`，再用小型可学习网络 `T_φ` 调制内容局部特征（不手写 AdaIN）。
- LL 不再完全锁死：允许颜色和平涂关系变化，只约束主要结构。
- 关键风险：仅靠 FM loss，`T_φ` 会退化成恒等映射。伪目标生成器必须先接受**独立的风格分布监督**；伪目标先自评（是否明显有风格、是否复制参考图结构），不达标直接淘汰，不进 FM。

### B2 严格对齐的自监督配对（作为 B1 预训练）
- `y = 真实绘画`，`x' = Destyle(y)`：只用随机颜色归一化、低频外观扰动、受控高频抑制，**保持几何**，不依赖外部生成模型。
- 训练 `T_φ(x', r_s) → y`；参考图混入裁剪/打乱位置的 patch 与同风格其他作品，防止模型直接从 x' 恢复 y 而忽略参考。
- 必须追加真实内容图上的无配对训练，不能只做重建。

### B3 内部 patch 判别器（最后手段）
- 条件于 style id 的多尺度 PatchGAN（真实绘画 patch vs 生成 patch）+ 固定图像 patch-SWD：`L_T = λ_style·L_patch + λ_content·L_structure + λ_pair·L_B2`。
- 只在 B1/B2 已能稳定保结构之后才上，受控强度，防黑点/噪纹/结构破坏。

### 训练框架解耦（工程上最重要）
1. 阶段 1：只训 `T_φ`，用 B2 配对监督，逐步引入真实照片上的内容约束与风格分布监督；此阶段独立评价 `z*` 质量。
2. 阶段 2：固定 `T_φ`，`z_1 = stopgrad(T_φ(z_c, z_s))`，`z_t = (1-t)z_c + t z_1`，`v = z_1 - z_c`，沿用现有 DWT 分头与 FM。初期冻结 `T_φ`，确认有效后再考虑 EMA。
3. 阶段 3：补真正的参考图条件，替换 `style_id → style_memory` 主路径。

### 必须通过的两个单元测试
1. 固定内容与 style_id，只换参考画 → 输出产生可测的风格变化（防忽略参考）。
2. 平移/打乱参考图空间位置 → 参考图物体轮廓不应被搬到输出里（防复制结构）。

### 实验顺序与判据
B0（当前 WEAVE 统一基线）→ B1a（内部注意力 + 伪端点；判据：伪目标自身优于原 SAT）→ B2（配对预训练；判据：重建好且参考条件不被忽略）→ B1+B2（主力候选；判据：风格升且 DINO-C/LPIPS 不崩）→ B3（判据：细节提升且无黑点/噪纹/结构复制）。

- 只看 cross-style 并按目标风格拆分评估（历史上部分风格明显偏弱，均值会掩盖问题）。
- 伪目标质量单独评估，不能只看 FM loss 下降。
- 数据：先用 5k 缓存快速验证，再上 18k 做等步数对照（18k 需先编码 latent）。

## 6. 防假结论闸门（硬性）

1. 每个 run 启动时写实际生效配置快照到 run 目录，并打印"新机制已激活"断言（风格损失非零、梯度范数 > 0、reference 分支参与 forward）。
2. 总 loss 下降且各分项损失非零，否则标"实现可疑"，不得进结果表。
3. 关闭新机制的对照组必须与已有基线**逐位一致**（同 seed 同步数）；不一致先查实现。
4. 每组 ≥2 seed；单 seed 标"待复现"。
5. 所有 run 登记 `/mnt/i/Latent_Style/runs/registry.jsonl`。
