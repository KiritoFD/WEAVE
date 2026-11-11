# 总览：从 Local Reader 失败到 S10B 定稿（2026-11-11 ~ 11-12）

> 本文是整段工作的**决策链总览**。各环节的原始细节见本目录分篇；
> `docs/v6/` 保留历史原件。重点结论：**S10 不需要重训就能配合原版 SD0 Encoder**。

---

## 0. 决策链

| # | 阶段 | 研究问题 | 结果 | 决策 |
|---|---|---|---|---|
| ① | Gate R | 极小局部读取器能否从原生 latent 稳定读画风？ | **失败**：hard AUROC 0.4645/0.4770（目标 ≥0.55），5-shot 0.0353/0.0529（flat latent 0.0912） | 停止 Encoder 训练路线，不再微调固定统计量 |
| ② | SD0-S10 V2 | 冻结官方 SD0 Encoder 训练 10.66M S10 | C0/C1 两 Gate 通过，重建明显优于旧 S10，但 CSD Keep 仍低于官方 | 保留多个 Pareto 候选 |
| ③ | 速度口径 | 新 Decoder 推理速度能否与论文比？ | 三套口径互不可比；3060 同硬件 TRT b8 实测 C0 = 27.08 ms/img（2.71×） | 只报同口径占比与相对加速 |
| ④ | 旧 S10 适配 SD0 | 旧 S10（A0 配套）能否零成本用于 SD0？ | **B 直接兼容已接近 A**，且下游反超；闭式校准 C 显著更差 | **直接使用旧 S10，不重训、不校准** |
| ⑤ | S10B 定稿 | 去掉 A0 运行时依赖 | 4 步验收全过，字节级证明 S10B = 旧 S10 | 发布自包含包 |

**贯穿全链的教训**：多次「看似需要重训/改结构」的结论，最终都被更仔细的接口与协议核查推翻——
`post_quant_conv` 单次应用、`in_train` 数据隔离、`paper_rgb_v1` 评测口径、未缩放 latent 契约。工程正确性先于算法投入。

---

## 1. ① Gate R：Local Style Reader（失败归档）

- 20,848 参数 Deep Sets 读取器 + 34,624 参数 CNN 对照，SupCon + 内容匹配硬负样本
- 全量 1500 步：Local AUROC 0.4645、CNN 0.4770、5-shot 0.0353/0.0529、未见流派 0.1395
- **不过度推断**：easy-neg AUROC 0.5375 → 0.5976/0.6098 → latent **有**风格信息；hard AUROC<0.5 源于内容匹配的异风格在 latent 上比同风格跨画家更近
- 配对 Bootstrap 修正为「按源图聚类 + 每源图 pos−hard gap」；CNN 略优 → 收益来自可训练非线性而非局部集合归纳偏置
- 📄 归档 `docs/v6/LOCAL_STYLE_READER_REPORT.md`、`docs/v6/ART_VAE_ENCODER_REPORT.md`

## 2. ② SD0-S10 V2：48h 训练链

**边界**：官方 `sd-vae-ft-ema` Encoder/quant_conv 冻结；未缩放 4×64×64 latent；`post_quant_conv` 随 Decoder 训练且恰好一次；Teacher 官方 49.49M eval+detach；S10 拓扑不变。

| 阶段 | 结果 | Gate |
|---|---|---|
| P0 数据 | 78,661 / 2,048 / 256 重新编码，全扫零异常、128 张重编码 cos 0.99995、Encoder 确定性 0.0 | — |
| P1 W/C | W LPIPS 0.09060 vs C 0.12819（各 800 步） | chose **W** |
| C0（3000 步，211.6 min） | LPIPS 0.08588、HF corr 0.47810、PSNR 25.42 | **passed** |
| C1（2500 步，192.2 min） | LPIPS 0.08504、resp_cos 0.6495→0.6710、PSNR 25.49 | **passed** |

C1 真实训练 **0 次 OOM / 0 次二次 backward**；此前担心的隔离响应分支经 `test_c1_graph.py` 纯 CPU 验证 + 实跑确认。

**四层验收**（同协议 SD0 holdout，LPIPS）：official 0.07836 / old_s10 0.09340 / c0 0.08588 / c1 **0.08504**；CSD Keep 0.961 / 0.876 / 0.853 / 0.848。

**链异常与自动降级**：step-9 final eval 因 teacher 在 bs=8 带 autograd 的 LPIPS 输入梯度 OOM（21.57GB）退出；`guard_p5.sh` 写负面报告、仍拉起 P5、权重零丢失；随后 input-grad 段 8→2 图修复并全量重跑通过。

📄 [01_S10_TRAINING_FULL.md](01_S10_TRAINING_FULL.md) §4 ｜ [04_SD0_S10_V2_EXECUTION.md](04_SD0_S10_V2_EXECUTION.md) ｜ [05_SD0_S10_V2_FINAL.md](05_SD0_S10_V2_FINAL.md)

## 3. ③ 推理速度口径

| 来源 | 硬件/后端 | 数字 | 测什么 |
|---|---|---|---|
| 榜单 `2.86× / 15.57ms` | 4090 / PyTorch / bs1 | A0 44.61 → S10 15.57 ms | **纯 Decoder 段** |
| 论文 supplement | **3060 12GB** / ONNX FP16 encoder + TRT FP16 decoder / bs8 | 97.3 ms/img（encoder+WEAVE 18.7 + TRT decoder 78.6） | **整条生成流水** |
| 本轮 P5 Infra | 4090 / PyTorch / bs1 | official 57.83 / S10 类 18.1 ms | 纯 Decoder 段 |
| 3060 同硬件复测 | 3060 / TRT b8 | 官方 73.34 → 新 C0 **27.08 ms/img（2.71×）** | TRT Decoder 段 |

**绝对毫秒不可比**；只有同硬件/同后端的分段占比与相对加速可比。论文的 32 方法榜**没有任何外部 baseline 的推理时间列**。
📄 [07_INFERENCE_SPEED.md](07_INFERENCE_SPEED.md) ｜ [08_STAGED_PROFILE_AND_TRT.md](08_STAGED_PROFILE_AND_TRT.md)

## 4. ④ 关键决策：旧 S10 直接可用

方案「保持 S10 主体 10.658M 不变，只考察输入端 4→4 PQC 映射」的 A/B/C/D 对照：

| 档 | 定义 | P0 重建 DINO-C | P2 Oracle1 Affinity | P2 B4 Affinity |
|---|---|---:|---:|---:|
| A（参考，榜单第 4 名） | A0 + A0PQC + 旧S10 | 0.9695 | 0.5587 | — |
| **B（零训练）** | **SD0 + A0PQC + 旧S10** | **0.9678** | **0.5616** | **0.5511** |
| C（闭式校准） | SD0 + 最小二乘 PQC + 旧S10 | — | 0.5608 | 0.5503 |
| D（官方基线） | SD0 + SD0PQC + 官方 D0 | — | 0.5548 | 0.5483 |

- 接口审计：未缩放 latent、恰好一次 PQC、`cos(A0,SD0)=0.992142`、RMS 比 1.001719
- **历史「SD0→旧S10 DINO-C≈0.22」是接口/缩放错误，正确接口下 0.968**
- A 的 `0.5587 / 0.4360 / 0.6814` 与 `docs/baseline` 榜单第 4 名逐位吻合 → 同协议可比
- **C 显著劣于 B**（Affinity −0.00077、DINO-S −0.00211、DINO-C −0.00233、LPIPS +0.00126，配对 Bootstrap CI 全排除 0）→ 印证「中间特征最优 ≠ RGB 质量最优」
- B 显著优于官方 D0（Affinity +0.0068、LPIPS −0.0114）

**按方案决策表：B 已接近 A → 直接使用旧 S10，不做 1×1/3×3 适配、不重训 Decoder。**
📄 [09_S10B_DECISION.md](09_S10B_DECISION.md) ｜ `docs/v6/S10_SD0_ADAPT_REPORT.md`

## 5. ⑤ S10B 定稿

4 步验收：
1. B 实际 PQC 与 checkpoint 内 PQC 哈希一致（max\|diff\| 5e-6，仅展示舍入）
2. PQC + S10 主体导出为 `deliver/checkpoints/s10b_standalone/`
3. 独立加载与 B 输出**逐元素一致**（随机 latent 与真实缓存 SD0 latent 上 max\|diff\| 均 **0.0**）
4. 契约固定：未缩放 latent、单次 PQC、官方 SD0 Encoder 双 sha256、原 ckpt/主体/PQC 全部 sha256、结构与版本

**字节级证明**：`state_dict` sha256 `f8edda08…d5f` 两边相同、140 张量 `torch.equal`、PQC weight/bias 相同 → **S10B 就是旧 S10**。B 没改模型任何参数，只换了 Encoder。

📄 [10_S10B_FINAL.md](10_S10B_FINAL.md) ｜ [S10_READY_TO_USE.md](S10_READY_TO_USE.md)

---

## 6. 交付状态

| 项 | 位置 |
|---|---|
| 三个新 Decoder（recon/response/overall） | 4090 `runs/sd0_s10/`，本地 `deliver/checkpoints/runs/sd0_s10/` |
| 两个独立 Diffusers 包装器 | 4090 `export_c0_best/` / `export_c1_best/` |
| **S10B 自包含包** | 本地 `deliver/checkpoints/s10b_standalone/` |
| 32 方法榜同协议打分 | `deliver/q2a_oracle_scores_baseline_protocol.json` |
| A/B/C/D 适配结论 | `deliver/s10_sd0_adapt_FINAL.json` |
| 3000 张 Q2a+Oracle 出图 | 本地 `deliver/q2a_oracle_images/`（git 忽略） |

**未完成（如实）**：本轮 SD0-S10 的 TensorRT 正式全量验收（3060 已测 C0 TRT b8）；可选 P4 感知校准；「修复版 TopK」仓库无明确权重未纳入；「无锐化伪影/大扰动无伪影」需人工视觉网格判定。

---

## 7. 一句话

**停止为兼容 SD0 重训 Decoder。** 旧 S10 拿未缩放原版 SD0 latent 直接跑，零成本、零额外推理开销，且是当前最好的 Ours 配置（Affinity 0.5616 > 榜单第 4 名 0.5587 > 本轮新 C0/C1 0.5575/0.5576）。SD0-S10 V2 的 C0/C1 保留为「更好重建 + 响应一致性」的研究产物，但不是下游必要性路线。
