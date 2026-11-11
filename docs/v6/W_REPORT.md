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

## P2：空间自适应风格编辑门控（进行中）

> 动机：全局频带裁剪和全局 latent 编辑都会同时改动内容与风格区域。P2 改为在推理阶段对 R1 梯度施加空间/频带/内容保护门控，比较 uniform、HF、Top-K、content-tangential 与 target-style energy gating。

- 起点：B4 λ50。
- 评测：`tasks_750.json`，与既有协议一致。
- 脚本：`sbflow/scripts/run_w_gating_eval.sh`。
- 状态：已启动，结果待完成。

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
