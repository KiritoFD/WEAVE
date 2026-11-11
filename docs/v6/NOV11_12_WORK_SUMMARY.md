# 2026-11-11 ~ 11-12 工作总览：从 Local Reader 失败到 S10B 定稿

> 本次工作横跨三条相继推进的路线，最终收敛为一个**零重训、零 A0 运行时依赖**的交付结论。
> 所有文档、代码与结果 JSON 均随本次 commit 入库；大权重/图片包受 `.gitignore` 约束不入库，
> 但已在本地 `deliver/` 与 4090/3060 上留存。

---

## 0. 路线与决策时间线

| 阶段 | 研究问题 | 结论 | 决策 |
|---|---|---|---|
| ① Gate R | 极小局部读取器能否从原生 latent 稳定读画风？ | **失败**（hard AUROC 0.4645/0.4770 < 0.55；5-shot 低于 flat latent） | 不再微调固定统计量，停止 Encoder 训练路线 |
| ② SD0-S10 V2 | 冻结官方 SD0 Encoder，训练 10.66M S10 Decoder | 训练成功，两 Gate 通过；C0/C1 明显优于旧 S10 的重建，但 CSD Keep 仍低于官方 | 保留多个 Pareto 候选，不指定单一冠军 |
| ③ 旧 S10 适配 SD0 | 旧 S10（A0 配套）能否在原版 SD0 Encoder 下零成本可用？ | **B 直接兼容已接近 A**；闭式校准 C 反而更差 | **直接使用旧 S10 + SD0 Encoder，不重训、不校准** |
| ④ S10B 定稿 | 去掉 A0 运行时依赖 | 4 步验收全过，字节级证明 S10B = 旧 S10 | 发布自包含包 |

---

## 1. ① Gate R（Local Style Reader）— 失败归档

- 方案：冻结 A0 Encoder，训练 20,848 参数 Deep Sets 局部读取器 + 34,624 参数 CNN 对照
- 结果：Cross-Artist Content-Matched AUROC 0.4645 / 0.4770（目标 ≥0.55）；5-shot 0.0353 / 0.0529（flat latent 0.0912）
- **关键不过度推断**：easy-neg AUROC 从 0.5375 提到 0.5976/0.6098 → latent **有**风格信息；失败源于内容匹配负样本下风格与内容耦合
- 配对 Bootstrap 用「按源图聚类保留重复簇」修正；CNN 对照略优于 Local Reader → 收益来自可训练非线性，非局部归纳偏置
- 📄 [LOCAL_STYLE_READER_REPORT.md](LOCAL_STYLE_READER_REPORT.md) ｜ 前序 [ART_VAE_ENCODER_REPORT.md](ART_VAE_ENCODER_REPORT.md)
- 数据：`deliver/local_reader_*.json`、`deliver/art_vae_*.json`

---

## 2. ② SD0-S10 V2 — 48h 训练链

**固定边界**：官方 `sd-vae-ft-ema` Encoder/quant_conv 冻结；未缩放 latent；`post_quant_conv` 随 Decoder 训练且恰好一次；Teacher 官方 49.49M eval+detach；不改 S10 拓扑。

| 阶段 | 结果 | Gate |
|---|---|---|
| P0 数据 | 78,661 / 2,048 / 256，全扫零异常行，128 张重编码 cos 0.99995 | — |
| P1 W/C | W LPIPS 0.09060 vs C 0.12819 | chose **W** |
| C0（3000 步） | LPIPS **0.08588**、HF corr 0.47810、PSNR 25.42 | **passed** |
| C1（2500 步） | LPIPS **0.08504**、resp_cos 0.6495→0.6710、PSNR 25.49 | **passed** |

**P5 四层验收**（同协议 SD0 holdout）：official 0.07836 / old_s10 0.09340 / c0 0.08588 / c1 **0.08504**（LPIPS）；CSD Keep 0.9609 / 0.8759 / 0.8531 / 0.8479。

**下游 750 任务**：4 Decoder × 5 arm = 20 组全出图 + 打分，75 对配对 Bootstrap。

- 📄 执行记录 [SD0_S10_V2_EXECUTION.md](SD0_S10_V2_EXECUTION.md) ｜ 最终 [SD0_S10_V2_FINAL.md](SD0_S10_V2_FINAL.md)
- 📄 排行榜定位 [Q2A_ORACLE_DECODER_RANKING.md](Q2A_ORACLE_DECODER_RANKING.md)（错误口径）→ **[Q2A_ORACLE_BASELINE_RANKING.md](Q2A_ORACLE_BASELINE_RANKING.md)（`docs/baseline` 32 方法同协议，以此为准）**
- 代码：`tools/vae_suite/sd0_*.py`、`train_sd0_s10.py`、`eval_*.py`、`run_*.sh`、`guard_p5.sh`、`gen_q2a_oracle_images.py`、`score_baseline_protocol.py`、`export_sd0_s10.py`、`audit_gates.py`、`test_*.py`
- 数据：`deliver/sd0_s10_*.json`、`deliver/q2a_oracle_*.json`
- 出图：`deliver/q2a_oracle_images.tar.gz`（1.18 GB，本地留存，git 忽略）

---

## 3. 推理速度口径

**三套口径互不可比**：
- 榜单 15.57 ms / 2.86×：4090、bs1、PyTorch、**纯 Decoder 段**
- 论文 97.3 ms/img：3060 12GB、ONNX/TRT FP16、bs8、**整条生成流水**（encoder 18.7 + WEAVE + TRT decoder 78.6）
- 本轮 P5 Infra：4090、bs1、PyTorch 纯 Decoder 段（official 57.83 / S10 类 18.1 ms）

📄 [INFERENCE_SPEED_BASELINE.md](INFERENCE_SPEED_BASELINE.md)
📄 3060 同硬件分段 profile [STAGED_PROFILE_3060.md](STAGED_PROFILE_3060.md)、[S10_TRT_B8_SPEED.md](S10_TRT_B8_SPEED.md)
代码：`tools/profile_staged.py`、`tools/profile_staged_3060.py`

---

## 4. ③ 旧 S10 适配原版 SD0 Encoder — 决定性结论

方案：保持 S10 主体 10.658M 参数不变，只考察输入端 4→4 PQC 映射。

| 档 | 定义 | P0 重建 DINO-C | P2 Oracle1 Affinity |
|---|---|---:|---:|
| A | A0 + A0PQC + 旧S10（参考） | 0.9695 | 0.5587 |
| **B** | **SD0 + A0PQC + 旧S10（零训练）** | **0.9678** | **0.5616** |
| C | SD0 + 最小二乘校准 PQC + 旧S10 | — | 0.5608 |
| D | SD0 + SD0PQC + 官方 D0 | — | 0.5548 |

- **接口审计**：未缩放 latent、恰好一次 PQC、`cos(A0,SD0)=0.992142`、RMS 比 1.001719
- **历史「SD0→旧S10 DINO-C≈0.22」未复现** → 当时是缩放/PQC 接口错误，不是 latent 不可弥合
- **A 的 0.5587/0.4360/0.6814 与 `docs/baseline` 榜单第 4 名逐位吻合** → 打分链路完全可比
- **C 显著劣于 B**（Affinity −0.00077、DINO-S −0.00211、DINO-C −0.00233、LPIPS +0.00126，CI 全排除 0）；B4 分支同结论
- **按方案决策表：B 已接近 A → 直接使用旧 S10，无需训练、无需 1×1/3×3 适配、无需重训 Decoder**
- 📄 [S10_SD0_ADAPT_REPORT.md](S10_SD0_ADAPT_REPORT.md)
- 代码：`tools/vae_suite/s10_sd0_adapt.py`、`s10_sd0_p2*.py`
- 数据：`deliver/s10_sd0_*.json`

---

## 5. ④ S10B 定稿 — 零 A0 运行时依赖

用户要求的 4 步全部通过：

1. **PQC 哈希比对**：B 实际 PQC 与 checkpoint 内 PQC max\|diff\| 5e-6（仅 5 位小数展示舍入）→ 原装
2. **导出**：`deliver/checkpoints/s10b_standalone/{decoder.pt,config.json,decoder.py}`
3. **独立加载逐元素一致**：随机 latent 与真实缓存 SD0 latent 上 max\|diff\| 均为 **0.0**
4. **契约固定**：未缩放 latent、单次 PQC、官方 SD0 Encoder 双 sha256、原 ckpt/主体/PQC 全部 sha256、结构与版本

**核心答案：S10B 就是我们一直用的旧 S10，字节级完全相同**
```
state_dict sha256 : f8edda08…d5f  ==  f8edda08…d5f
140 张量全部 torch.equal : True
pqc weight / bias : True / True
```
B 实验没改模型任何一个参数，只把喂它的 Encoder 从 A0 换成官方 SD0；C 的校准被否；主体 10.658M 参数自始至终未动。独立包只是**包装**（把 PQC、契约、哈希、零依赖加载器打成自包含交付物）。

📄 [S10B_FINAL.md](S10B_FINAL.md) ｜ `deliver/s10b_final_manifest.json`

---

## 6. 最终交付状态

| 项 | 状态 |
|---|---|
| 三个新 Decoder checkpoint（recon/response/overall） | 4090 `runs/sd0_s10/`，本地 `deliver/checkpoints/runs/sd0_s10/` |
| 两个独立 Diffusers 包装器 | 4090 `export_c0_best/` / `export_c1_best/` |
| **S10B 自包含包** | 本地 `deliver/checkpoints/s10b_standalone/` |
| 32 方法榜同协议打分 | `deliver/q2a_oracle_scores_baseline_protocol.json` |
| A/B/C/D 适配结论 | `deliver/s10_sd0_adapt_FINAL.json` |
| 3000 张 Q2a+Oracle 出图 | 本地 `deliver/q2a_oracle_images/`（git 忽略） |

**仍未完成（如实）**：TensorRT 在本轮 SD0-S10 上的正式全量验收（3060 已测 C0 TRT b8 = 27.08 ms/img、2.71×）；可选 P4 感知校准；「修复版 TopK」仓库无明确权重未纳入；「无锐化伪影/大扰动无伪影」需人工视觉网格。

---

## 7. 提交范围说明

本次 commit **只**纳入：本轮 11 篇 `docs/v6/` 文档、`tools/vae_suite/` 中本轮新增的 sd0/s10 系列、`tools/profile_staged*.py`、`tools/local_reader/` 与 `tools/art_vae/` 前序代码、以及 `deliver/` 里的小结果 JSON。

**未纳入**：`docs/v7/*`、`tools/v6/score_diff.py`、`tools/v6_next/r2_eval.py` 等他人的已修改文件；1.18 GB 图片包、320 MB VAE 权重、40 MB S10B 权重（均被 `.gitignore` 的 `*.pt/*.tar.gz/*.png/*.safetensors` 排除，但保留在本地/远端）。
