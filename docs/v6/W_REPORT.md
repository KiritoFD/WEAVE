# WEAVE W 阶段报告 — 冻结潜空间下的单步拟合修复

> 协议：`tasks_750.json`（600 cross + 150 same），`paper_rgb_v1`，原始 RGB 零点，统一 TRT fp16 SD-VAE 评测。单卡串行 `sbflow/scripts/run_w_12h.sh`。
> 前提：**Encoder / latent 空间完全冻结**（A0 与 sd-vae-ft-ema 架构一致，encoder conv_in 差异仅 2e-4，A0 latent 可直接喂既有 Q2a/B4 生成器）。S10 Decoder 必须与 **A0 Encoder + A0 post_quant_conv** 配对（4090 正确配方，实测 DINO-C 0.9387；错配 sd-vae latent 会崩到 0.22）。

## 立项依据（R2 结论纠错）

R2 表 D 铁证：学生与 Oracle 教师位移 cos 0.638–0.683、RMS 比 0.52–0.65、相对拟合误差 0.53–0.60、R1 margin gap 1.29–1.77 —— **学生没把教师学进去**。Oracle 教师本身成功（Top-1 34.3%、DINO-S 0.4277）。因此病灶是**蒸馏损失**：潜空间逐像素 L2 无法传递高频、相位敏感的判别方向，且与旧 V6 DWT/C1 统计损失拔河。R2 原判『教师失败』已更正为『教师成功、学生拟合失败 → 研究蒸馏优化』。

## W 实验与实现

| 实验 | 实现 | 关键改动 |
|---|---|---|
| 实验一 W_FeatDistill | `w_distill.py` | R1 多尺度特征(64/32/16)匹配 + KL 分类边界 + 反吸引子 Margin；像素 L2 **仅 Haar LL**，高频放开 |
| 实验二 W_LossClean | `train_eval_v6.py` 开关 | 关旧 DWT `w_cross` 冲突项，C1 内容不变性仅约束 LL 低频结构 |
| 实验三 困难风格 | `w_distill.py` | 按风格 RMS（Min/Imp 0.15、ER/Roc/Ukiyo 0.32）+ 对 Imp/Min 决策边界的反吸引子排斥 |
| 实验四 W_S10_ImgCalib | `w_img_calib.py`+`w_vae.py` | 冻结 A0 Encoder + S10 Decoder，150 步 RGB 域 CSD margin + DINO-C 联合微调 |
| 实验五 845K+StyleKeys | 训练臂 W4 | dim32 + rank32 风格残差基，强特征监督下重测容量 |

- 损失量级经 `TERM_SCALE` 归一化 + `W_LOSS_SCALE=0.006` 对齐历史像素 MSE 量级（`L_distill≈0.09`），λ=20/50 的既有标定继续有效。
- 反吸引子用 `softplus(-margin/tau)` 有界化（原始 `-margin` 无下界会导致 logits 发散）。
- 14 项单元测试全部通过（`w_unit_tests.py`），含 A0/S10 配对与 `decode_s10` 缩放校验。
- 50 步真实 microbatch 稳定性标定：218K 臂 loss 2.06→1.92 单调、显存 5.4GB、~1.2s/step；**845K+StyleKeys 对照臂 loss 3.59→5.55 上升（不稳定）**，按计划『低优先级臂在稳定性/预算压力下砍掉』主动禁用，记为负结果，不进入正式训练矩阵。
- **S10 Decoder 配对纠错**：初版用 sd-vae-ft-ema latent 喂 S10 导致 DINO-C 崩到 0.22；经 ssh 4090 找到完整 VAE 结果后确认正确配方 = A0 Encoder + A0 post_quant_conv + S10，实测 DINO-C 0.9387（标准 0.9657 / A0 0.9666）。A0 Encoder 权重已拉回并固化于 `w_vae.py`。

## 统一协议结果（Cross600）

| 模型 | Top-1 | Margin | CSD ΔT | DINO-S | gain | DINO-C | LPIPS | MinOff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2a Direct (R2) | 6.0% | -0.2596 | 0.0799 | 0.3988 | 0.0355 | 0.8873 | 0.3759 | 25.6% |
| Q2a+Oracle (R2) | 34.3% | -0.0546 | 0.0688 | 0.4277 | 0.0644 | 0.7303 | 0.5240 | 27.3% |
| B3 λ20 (R2 anchor) | 17.8% | -0.1462 | 0.1025 | 0.4183 | 0.0550 | 0.8325 | 0.4412 | 26.9% |
| B4 λ50 (R2) | 20.8% | -0.1282 | 0.1023 | 0.4214 | 0.0581 | 0.8199 | 0.4539 | 27.1% |
| W1 FeatDistill | 10.7% | -0.1813 | 0.0819 | 0.4149 | 0.0517 | 0.8371 | 0.4254 | 26.7% |
| W2 LossClean | 10.2% | -0.1839 | 0.0782 | 0.4142 | 0.0509 | 0.8336 | 0.4271 | 27.1% |
| W3 LossClean λ50 | 13.5% | -0.1620 | 0.0829 | 0.4205 | 0.0572 | 0.8149 | 0.4445 | 27.3% |
| W4 845K+SK | — | — | — | — | — | — | — | — |
| W5 S10 ImgCalib | 9.2% | -0.2088 | 0.1089 | 0.4154 | 0.0521 | 0.8804 | 0.3957 | 24.4% |
| V1 A0-pool LossClean | 10.3% | -0.1839 | 0.0782 | 0.4145 | 0.0512 | 0.8332 | 0.4277 | 27.3% |
| V2 A0-pool PixelMSE | 18.2% | -0.1461 | 0.1026 | 0.4184 | 0.0552 | 0.8325 | 0.4413 | 26.7% |

## 换 VAE 重编码训练对照（用户追加）

- 用户指出：A0 与 sd-vae 的 latent 差异不应靠『差异小』论证绕过，应直接**用 A0 Encoder 重新编码 15,545 张训练池**（`reencode_a0.py`，产出 `orig_mean_fp16_a0/`，schema 与原池完全一致，仅 `latents`/`vae` 字段不同），再在同一配方下训练对照臂，看换 VAE 是否带来真实差异。
- 训练侧新增 `--latent-dir` 开关（默认仍为原 sd-vae 池，R2/W 所有既有臂完全不受影响）。
- 对照臂：`V1_A0_LossClean`（与 W2 完全同配方，只换 latent 池）、`V2_A0_PixelMSE`（A0 池 + 原始像素 MSE 蒸馏，隔离 VAE 本身的影响）。
- 串行入口 `run_w_vaeswap_12h.sh`：A0 重编码 → 两臂训练 → 750 评测 → 配对比较，独立脚本以避开正在运行的 W 主流水线（单卡排他）。
- **结果**：V1 Top-1 10.3%、V2 18.2%，均未超过 B3/B4；换 A0 latent 池本身没有解决风格迁移问题。

## P0：内容条件源风格泄漏与 Style-ID 因果审计（已完成）

采用 40 张分层源图（每风格 8 张），先审计 latent 表示，再固定同一内容条件、强制切换 5 个 Style-ID 并解码评测。

### 潜空间源风格泄漏（R1 判官）

| Latent | Source-style Top-1 | Source-style Margin |
|---|---:|---:|
| raw posterior latent | 82.5% | +3.4672 |
| c1_gentle canonicalized | 75.0% | +2.4528 |

### 固定内容、改变 Style-ID 的因果响应

| Metric | Value |
|---|---:|
| Cross600 Top-1 | 8.1% |
| Cross600 Margin | -0.2442 |
| Cross600 DINO-S | 0.3816 |
| Cross600 DINO-C | 0.8705 |
| Same150 Top-1 | 100.0% |
| Same150 DINO-C | 0.8909 |
| **输出仍被判为源风格的比例** | **82.5%** |

分目标 Top-1：ER 6.2%、Imp 12.5%、Min 6.2%、Rococo 15.6%、Ukiyo 0.0%。

**结论**：即使内容条件固定，输出仍有 82.5% 被判为源风格；Style-ID 的因果作用很弱。内容条件中的源风格泄漏是真实存在的，后续应优先改造内容条件，而不是继续增强蒸馏损失。

## P1：真实 latent 密度先验 + 引导编辑（已完成）

> 动机：R2 表 D 表明学生没有拟合 Oracle 教师，因此进一步测试“真实数据 score 先验 + R1 引导”能否改善单步模型的风格—内容权衡。详细报告见 [W_P1_REPORT.md](docs/v6/W_P1_REPORT.md)。

Score prior 训练：3.27M 参数，1000 步，训练/验证 latent 数 14,545/1,000，最终 loss 0.5655，显存 7.27GB。

| 方法 | Top-1 | Margin | CSD ΔT | DINO-S | DINO-C | LPIPS | Min 误判 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B3 λ20（anchor） | 17.8% | -0.1462 | 0.1025 | 0.4183 | 0.8325 | 0.4412 | 26.9% |
| R1 only（4 步） | 31.7% | -0.0632 | 0.0722 | 0.4361 | 0.7516 | 0.5172 | 24.2% |
| Score only（4 步） | 31.8% | -0.1398 | -0.0097 | 0.2845 | 0.5086 | 0.6245 | 58.8% |
| R1 + Score（4 步） | 34.3% | -0.0932 | 0.0143 | 0.3559 | 0.5936 | 0.5750 | 47.9% |

- **无配置通过 P4.3 联合门槛**。Score only 虽提高 Top-1，但严重破坏内容与同风格一致性；R1+Score 达到 Oracle 级 Top-1，但内容/画风相似度明显下降。
- 结论：真实数据 score 先验没有解决 R1 的内容损伤问题；该路线不再作为主线推进。

## P2：空间自适应风格编辑门控（已完成）

> 动机：全局频带裁剪和全局 latent 编辑都会同时改动内容与风格区域。P2 改为在推理阶段对 R1 梯度施加空间/频带/内容保护门控，比较 uniform、HF、Top-K、content-tangential 与 target-style energy gating。

- 起点：B4 λ50。
- 评测：`tasks_750.json`，与既有协议一致。
- 脚本：`sbflow/scripts/run_w_gating_eval.sh`。
- 状态：**已完成**。

### Cross600 结果

| 方法 | Top-1 | Margin | CSD ΔT | DINO-S | DINO-C | LPIPS | Min 误判 | Same150 Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B4 λ50（起点） | 20.8% | -0.1282 | 0.1023 | 0.4214 | 0.8199 | 0.4539 | 27.1% | — |
| Q2a+Oracle | 34.3% | -0.0546 | 0.0688 | 0.4277 | 0.7303 | 0.5240 | 27.3% | — |
| P2 Gated Full | 35.8% | -0.0551 | 0.0746 | 0.4353 | 0.7445 | 0.5243 | 24.0% | 96.0% |
| P2 Gated HF | 34.8% | -0.0555 | 0.0822 | 0.4334 | 0.7033 | 0.5499 | 25.6% | 94.7% |
| P2 Gated TopK | 35.0% | -0.0718 | 0.0778 | 0.4372 | 0.7569 | 0.5062 | 26.0% | 95.3% |
| P2 Gated Content | 35.8% | -0.0557 | 0.0750 | 0.4353 | 0.7457 | 0.5229 | 24.2% | 96.7% |
| P2 Gated StyleEnergy | 35.7% | -0.0577 | 0.0729 | 0.4339 | 0.7421 | 0.5208 | 24.2% | 95.3% |

### 相对 B3 的配对聚类 bootstrap

| 方法 | ΔTop-1 [95% CI] | ΔDINO-S [95% CI] | ΔDINO-C | ΔLPIPS | P4.3 门槛 |
|---|---:|---:|---:|---:|---|
| P2 Gated Full | +18.0pp [+15.0, +20.9] | +0.0171 [+0.0133, +0.0207] | -0.0880 | +0.0831 | PARTIAL |
| P2 Gated HF | +17.0pp [+13.9, +20.0] | +0.0151 [+0.0110, +0.0191] | -0.1291 | +0.1087 | PARTIAL |
| P2 Gated TopK | +17.2pp [+14.0, +20.2] | +0.0189 [+0.0156, +0.0220] | -0.0755 | +0.0650 | PARTIAL |
| P2 Gated Content | +18.0pp [+14.9, +21.0] | +0.0170 [+0.0133, +0.0207] | -0.0867 | +0.0818 | PARTIAL |
| P2 Gated StyleEnergy | +17.8pp [+14.7, +20.8] | +0.0156 [+0.0116, +0.0194] | -0.0903 | +0.0796 | PARTIAL |

### 结论

- 所有门控模式都显著优于 B3 的 Top-1，但 **DINO-S 提升均略低于 +0.020 的预设主门槛**，因此按原始标准判定为 PARTIAL，不算正式通过。
- **P2 Gated TopK 是当前最好的平衡点**：DINO-S 0.4372、DINO-C 0.7569、LPIPS 0.5062，均优于其他门控模式的内容保持表现。
- 与 Oracle 相比，P2 Gated Full/Content 的 Top-1、DINO-S、DINO-C、Min 误判均更好，说明空间自适应门控已经优于单步 Oracle 教师；但严格预设门槛仍未通过。
- 下一步若继续该方向，应围绕 TopK 门控做步长/步数/保留比例的小规模扫描，而不是直接进入大规模训练。

### TopK 扫描（P2b，已完成）

在相同起点 B4、相同评测协议下继续扫描 TopK 保留比例与步数。`topk25_s8` 使用 8 步、每步 RMS 0.04，保持总编辑预算与 4 步 × 0.08 一致。

| 方法 | Top-1 | DINO-S | DINO-C | LPIPS | Min 误判 | Same150 Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| P2 TopK10 | 31.2% | 0.4327 | 0.7580 | 0.4998 | 26.7% | 94.7% |
| P2 TopK25 | 35.0% | 0.4372 | 0.7569 | 0.5062 | 26.0% | 95.3% |
| P2 TopK50 | 33.5% | 0.4386 | 0.7529 | 0.5130 | 25.6% | 96.0% |
| P2 TopK25 S8 | 33.0% | 0.4389 | 0.7552 | 0.5095 | 26.2% | 95.3% |

相对 B3 的配对比较：

| 方法 | ΔTop-1 [95% CI] | ΔDINO-S [95% CI] | ΔDINO-C | ΔLPIPS | P4.3 门槛 |
|---|---:|---:|---:|---:|---|
| P2 TopK10 | +13.3pp [+10.5, +16.1] | +0.0144 [+0.0114, +0.0174] | -0.0744 | +0.0586 | PARTIAL |
| P2 TopK50 | +15.7pp [+12.6, +18.6] | +0.0203 [+0.0169, +0.0238] | -0.0796 | +0.0718 | PARTIAL |
| P2 TopK25 S8 | +15.2pp [+12.2, +18.1] | +0.0207 [+0.0171, +0.0241] | -0.0772 | +0.0683 | PARTIAL |

**P2 结论已按 batch-mask 修复结果更正**：旧 TopK 实现错误地让 batch 共享掩码。修复后完整 750 复测显示，`TopK50` 与 `TopK25 S8` 的 DINO-S 增益分别为 +0.0187 和 +0.0185，不再达到 +0.020 主门槛。P2 空间门控方向仍有稳定风格提升，但严格 P4.3 联合门槛仍为 PARTIAL/未通过。详见 `docs/v6/TOPK_FIX_RETEST.md`。

## P3：真实画作驱动的条件流匹配（试点已完成）

> 目标：避免旧 V8 桥流的失败模式——不使用无配对 latent L2，也不使用伪 GT 合成端点；改为用 **真实画作内容匹配对** 训练条件 Flow Matching。

### 数据与训练

- 使用 DINO 检索表，从 15,545 张真实画作中构建 **20 个 directed style route × 100 对 = 2,000 对**真实内容匹配样本。
- `z0` 与 `z1` 都是真实画作 latent：`z0` 为源风格原图，`z1` 为目标风格中内容最接近的真实画作。
- 配对缓存：`pseudo_gt_24h/pairs/real_content_match`，20 个 shard，校验通过。
- 模型：现有 sbflow 条件 Flow Matching 框架，982,618 参数。
- 训练：batch 96，300 步试点；loss 从约 2.264 降至 2.182，训练稳定，峰值显存约 2.2GB。
- checkpoint：`runs/p3/real_pair_pilot/epoch_0015.pt`。

### 750 图评估（root eval 协议：CLIP/LPIPS）

| 指标 | Style transfer | Identity reconstruction |
|---|---:|---:|
| CLIP style | 0.6936 | 0.8337 |
| CLIP style Δ vs old IDT | 0.0537 | 0.1938 |
| CLIP text-style | 0.2173 | 0.2684 |
| Content LPIPS | 0.3448 | 0.2829 |

All-pairs overview：CLIP style 0.7216，CLIP style Δ 0.0817，content LPIPS 0.3324。

### 结论

- P3 试点证明了真实画作内容匹配对可以稳定训练条件 Flow Matching，且不会出现旧 V8 桥流的崩坏。
- 该评估器输出的是 CLIP/LPIPS 指标，不能与 B3/B4/W 系列的 CSD/DINO 指标直接排名。
- 从自身指标看，P3 试点已有可辨识的风格迁移能力；下一步可做更长训练，并/或将 P3 模型接入统一 CSD/DINO 评测接口进行同口径比较。

### 2000 步长训练结果

| 模型 | Style-transfer CLIP style | Style-transfer LPIPS | Identity CLIP style | Identity LPIPS |
|---|---:|---:|---:|---:|
| P3 pilot（300 步） | 0.6936 | 0.3448 | 0.8337 | 0.2829 |
| P3 long（2000 步） | 0.7062 | 0.4048 | 0.8387 | 0.3044 |

- 长训练提升了风格相似度（style-transfer CLIP style 0.6936 → 0.7062，identity CLIP style 0.8337 → 0.8387）。
- 但内容/重建 LPIPS 同步变差（style-transfer LPIPS 0.3448 → 0.4048，identity LPIPS 0.2829 → 0.3044）。
- 结论：真实画作 Flow Matching 路线可行，但当前 2000 步模型在风格增强与内容保持之间仍偏向风格；后续若继续该路线，应增加内容保持约束或减少步数/增强正则，而不是简单继续加训练步数。

### P3 同口径 CSD/DINO 评测（已完成）

将 P3 长训练模型生成的 750 张图像接入统一 `r2_eval.py` 协议后，结果如下：

| 指标 | P3 RealPair FM | B3 λ20 | B4 λ50 | P2 TopK25 S8 |
|---|---:|---:|---:|---:|
| Cross600 Top-1 | 17.0% | 17.8% | 20.8% | 33.0% |
| DINO-S | 0.4056 | 0.4183 | 0.4214 | 0.4389 |
| DINO-C | 0.7468 | 0.8325 | 0.8199 | 0.7552 |
| LPIPS | 0.5714 | 0.4412 | 0.4539 | 0.5095 |
| Min 误判 | 27.7% | 26.9% | 27.1% | 26.2% |
| Same150 Top-1 | 94.0% | — | — | 95.3% |

相对 B3 的配对检验：ΔTop-1 -0.8pp [-3.5, +1.8]；ΔDINO-S -0.0126 [-0.0198, -0.0054]；ΔDINO-C -0.0857；ΔLPIPS +0.1302。判定：**NO**。

**P3 最终结论**：真实画作 Flow Matching 训练稳定，但在统一 CSD/DINO 协议下未超过 B3/B4，也明显落后于 P2 TopK 门控。该路线暂不继续扩大训练。

## P4.3 成功门槛判定（相对 B3 anchor）

- 主门槛 DINO-S ≥ +0.020 且 CI 下界 > 0：**无模型通过**。
- 各臂配对判定：{'Q2a Direct (R2)': 'NO', 'Q2a + R1 Oracle (R2)': 'PARTIAL', 'B4 λ=50 (R2)': 'PARTIAL', 'ScoreDiff N24 (R2)': 'NO', 'W1 FeatDistill 218K': 'NO', 'W2 LossClean 218K': 'NO', 'W3 LossClean 218K λ50': 'NO', 'W5 S10 ImgCalib': 'NO'}

- 按计划：即便无 +0.020 的独立显著收益，也按完整结果交付，不通过修改测试协议或选择性展示制造提升。

## 复现

```bash
bash sbflow/scripts/run_w_12h.sh            # W 主流水线（可续跑）
bash sbflow/scripts/run_w_12h.sh --dry-run  # 仅预检（编译 + 14 单测 + A0/S10 校验）
bash sbflow/scripts/run_w_vaeswap_12h.sh    # A0 重编码 + 换 VAE 训练对照
bash sbflow/scripts/run_w_p1_12h.sh         # score prior + R1/Score/联合引导评测
bash sbflow/scripts/run_w_gating_eval.sh    # P2 空间自适应门控评测
python3 -m tools.v6_next.w_unit_tests       # 14 项单元测试
python3 -m tools.v6_next.w_next_p0_audit    # 源风格泄漏 / Style-ID 因果审计
```

- 结果目录 `runs/v6_w/`；A0 Encoder 在 `runs/a0/a0_encoder_only.pt`（SHA256 前缀 `70715532435ad6c2`，从 4090 `/root/Workspace/WEAVE_VAE/runs_vae/A0/` 导出）。
- 已知限制：Val200 ⊂ Test750 且 Test750 反复参与研发；未构造作者隔离盲集，所有显著性为研发集内配对，不宣称盲测。
