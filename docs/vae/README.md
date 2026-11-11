# docs/vae — VAE / S10 Decoder 专门文档

> 本目录集中收录 **S10 Decoder 全流程训练、SD0 适配、评测与推理速度**的全部文档。
> `docs/v6/` 下的原件保留作为历史归档；**本目录为权威入口**。
> 由 commit `bd04475` 之后的专门提交建立（2026-11-12）。

---

## 导航

| 编号 | 文档 | 内容 |
|---|---|---|
| **01** | **[01_S10_TRAINING_FULL.md](01_S10_TRAINING_FULL.md)** | ⭐ **S10 全流程训练过程**（权威）：剪枝方法、接口契约、干净数据池、原始 S10 五阶段轨迹、三个 bug、CSD 平台、SD0-S10 V2 二次训练、最终结论 |
| 02 | [02_ORIGINAL_S10_REPORT.md](02_ORIGINAL_S10_REPORT.md) | 原始 S10 12h 计划最终报告（指标/CSD 平台/三 bug/L0 方向诊断/风格可分性） |
| 03 | [03_S10_EXECUTION_LOG.md](03_S10_EXECUTION_LOG.md) | 12h 计划执行日志（P0/P1/P2 逐步记录与数据泄漏发现） |
| 04 | [04_SD0_S10_V2_EXECUTION.md](04_SD0_S10_V2_EXECUTION.md) | SD0-S10 V2 执行方案与 P0 基线（Cold/Warm 修正、离线依赖、组件清单） |
| 05 | [05_SD0_S10_V2_FINAL.md](05_SD0_S10_V2_FINAL.md) | SD0-S10 V2 最终结果（两 Gate、四层验收、链异常修复、交付物） |
| 06 | [06_Q2A_ORACLE_BASELINE_RANKING.md](06_Q2A_ORACLE_BASELINE_RANKING.md) | Q2a+Oracle1 × 新 Decoder 在 `docs/baseline` 32 方法榜的定位（`paper_rgb_v1` 同协议） |
| 07 | [07_INFERENCE_SPEED.md](07_INFERENCE_SPEED.md) | 推理速度三套口径辨析（4090 Decoder 段 / 论文 3060 端到端 / 本轮 Infra） |
| 08 | [08_STAGED_PROFILE_AND_TRT.md](08_STAGED_PROFILE_AND_TRT.md) | 3060 同硬件分段 profile 与 TRT b8 测速（encoder/WEAVE/decoder 占比） |
| 09 | **[09_S10B_DECISION.md](09_S10B_DECISION.md)** | ⭐ **关键决策**：旧 S10 无需重训即可用于原版 SD0 Encoder；A/B/C/D 对照；S10B 字节级证明 |
| 10 | [10_S10B_FINAL.md](10_S10B_FINAL.md) | S10B 自包含交付（4 步验收、哈希、独立加载逐元素一致、契约） |
| 11 | [11_DECODER_ABC_DIAGNOSIS.md](11_DECODER_ABC_DIAGNOSIS.md) | 前置诊断：画风在 Decoder 哪一步消失（S10 加 Loss 之前） |
| 12 | [12_STYLE_VAE_ABC_REPORT.md](12_STYLE_VAE_ABC_REPORT.md) | 风格友好型 VAE 联合重训 A/B/C（失败，促成纯重建 S10 路线） |
| 13 | [13_VAE_A0_A1_A2_REPORT.md](13_VAE_A0_A1_A2_REPORT.md) | Art-VAE B0/A0/A1/A2 全景评测 |
| 14 | [14_PRUNING_RECOVERY_POSTMORTEM.md](14_PRUNING_RECOVERY_POSTMORTEM.md) | 剪枝与恢复训练复盘（V11/V12/V13 + 原报告表 1–10） |
| — | **[00_OVERVIEW.md](00_OVERVIEW.md)** | 完整工作总览（Gate R → SD0-S10 V2 → 适配 → S10B 的决策链） |
| — | [S10_READY_TO_USE.md](S10_READY_TO_USE.md) | 一页速查：怎么直接把 S10 拿来用 |

---

## 一页结论

**不要再为兼容原版 SD0 latent 重训 10.66M Decoder。**

旧 S10（`s10_p5_lowlr/decoder.pt`，A0 配套训练）保持权重完全不变，只把喂它的 Encoder 换成官方 `sd-vae-ft-ema`、输入用**未缩放 posterior-mean latent**、沿用 checkpoint 自带的 `post_quant_conv` 且**恰好应用一次**，即可直接使用：

| 档 | Encoder | PQC | P0 重建 DINO-C | P2 Oracle1 Affinity |
|---|---|---|---:|---:|
| A（历史榜单第 4 名） | A0 | A0 原版 | 0.9695 | 0.5587 |
| **B（直接用，零训练）** | **SD0** | **A0 原版（原封不动）** | **0.9678** | **0.5616** |
| C（闭式校准 PQC） | SD0 | 最小二乘 | — | 0.5608（显著更差） |
| D（官方 49.49M） | SD0 | SD0 原版 | — | 0.5548 |

历史「SD0→旧S10 DINO-C≈0.22」是接口/缩放错误；正确接口下是 0.968。`cos(A0,SD0)=0.992142`、RMS 比 1.001719。

`S10B` **字节级就是旧 S10**（`state_dict` sha256 相同、140 张量 `torch.equal`、PQC 相同）；独立包只把 PQC、契约、哈希、零依赖加载器打成自包含交付物，**彻底去掉 A0 运行时依赖**。

---

## 训练全流程速览（详见 01）

```
A0 Decoder 49.490M
  └─ Taylor 重要性剪枝 → 10.658M（widths [64,128,192,256]），零步 L1 0.5568
       └─ P1  L1 容量恢复      13k      1000步  → L1 0.0659 / LPIPS 0.390 / CSD 0.608
            └─ P0  L1+LPIPS@512 13k     600步   → L1 0.0720 / LPIPS 0.1099 / CSD 0.8484
                 └─ P2  全量 78,661    1400步   → L1 0.0686 / LPIPS 0.0862 / CSD 0.8710
                      └─ P3  低 LR 续训  400步   → L1 0.0683 / LPIPS 0.0851 / CSD 0.8713
                           └─ s10_p5_lowlr：测试集 LPIPS 0.0926 / CSD 0.8535 / forward 15.57ms (2.86×)
```

**Loss 自始至终只有 `L1 + 0.5·LPIPS@512`**：无教师蒸馏、无 CSD/HF Loss。CSD 从 0.608 自动回升到 0.871 并进入平台。

---

## 三组不可混淆的口径

1. **训练侧**：原始 S10 用 A0 Encoder / A0 缓存；SD0-S10 V2 用官方 SD0 Encoder 重新编码的缓存。两者**不能靠 RMS 缩放互转**。
2. **评测侧**：`docs/baseline` 32 方法榜用 `paper_rgb_v1`（DINO-S = 与目标风格 30 张参考的最大余弦，CLIP-S = 目标风格原型余弦，Style Affinity = 两者均值）。任何「生成图 vs 源图余弦」都不是该口径，不能与榜比。
3. **速度侧**：4090 的 15.57 ms 是纯 Decoder 段 PyTorch bs1；论文的 97.3 ms/img 是 3060 ONNX/TRT bs8 整条生成流水。**绝对毫秒不可比**，只有同硬件同后端的分段占比与相对加速可比。

---

## 复现代码索引

```
tools/vae_suite/
  decoder_prune.py / decoder_build_s10.py   结构化剪枝 + Taylor 重要性
  decoder_cache_full.py                    干净 A0 缓存（已修 in_train 泄漏）
  decoder_train.py                         原始 S10 训练（L1 + LPIPS@512）
  sd0_cache.py                             官方 SD0 重新编码缓存
  build_cold_sd0.py                        Cold-SD0 剪枝
  sd0_losses.py                            Haar HF / 响应蒸馏 / off-manifold
  train_sd0_s10.py                         C0/C1 统一训练循环
  eval_sd0_s10.py                          四层验收第 1/2/4 层
  eval_generator_swap_full.py              第三层五生成器替换
  score_baseline_protocol.py               paper_rgb_v1 打分
  gen_q2a_oracle_images.py                 750 任务出图
  s10_sd0_adapt.py / s10_sd0_p2*.py        A/B/C/D 适配对照
  export_s10b.py                           S10B 导出 + 哈希 + 逐元素验证
  run_sd0_s10.sh / run_p5_final.sh / guard_p5.sh   串行总控与自动降级
tools/profile_staged.py / profile_staged_3060.py   分段推理 profile
```

结果 JSON 在 `deliver/`；大权重与 3000 张出图受 `.gitignore` 约束不入库，本地/4090/3060 留存。
