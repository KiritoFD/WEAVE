# SD0-S10 V2 最终结果（2026-11-12）

**状态**：训练链 P0→P1→C0→C1 全部完成，两个 Gate 均通过；链内 step-9 final eval 因 OOM 异常退出，guard 按降级路径交付；随后手动修复并**全量重跑通过**所有四层评测；三个交付 checkpoint 与两个独立 Diffusers 包装器齐备。

## 1. 训练与 Gate

| 阶段 | 步数/耗时 | LPIPS@512 | CSD Keep | HF corr L0 | PSNR | Gate |
|---|---|---:|---:|---:|---:|---|
| P1_W (Warm-A0) | 800 / 56.4min | 0.09060 | 0.85726 | 0.45455 | 25.25 | — |
| P1_C (Cold-SD0) | 800 / 56.6min | 0.12819 | 0.79743 | 0.39597 | 24.40 | — |
| **P1 Gate** | — | — | — | — | — | **chose W**（Cold 等成本下仍落后 +0.038） |
| **C0** | 3000 / 211.6min | **0.08588** | 0.85313 | **0.47810** | 25.42 | **passed**（LPIPS −0.00476、HF corr +0.01751、hf_l1 −0.00111） |
| **C1** | 2500 / 192.2min | **0.08504** | 0.84787 | **0.48772** | **25.49** | **passed**（resp 0.30186→0.29550、resp_cos 0.64953→0.67099） |

C1 真实训练中 **0 次 OOM、0 次二次 backward、0 KeyError**，此前标记的「隔离响应分支」风险实证排除。

## 2. 四层验收（同协议 SD0 holdout，不套用 A0 历史值）

| 变体 | LPIPS | PSNR | CSD | HF corr | HF 能量比 |
|---|---:|---:|---:|---:|---:|
| official SD0 | 0.07836 | 24.72 | 0.9609 | 0.4247 | 0.9856 |
| old_s10 (A0 训练) | 0.09340 | 24.89 | 0.8759 | 0.4254 | 0.6942 |
| **c0** | 0.08588 | **25.42** | 0.8531 | 0.4781 | 0.5900 |
| **c1** | **0.08504** | **25.49** | 0.8479 | **0.4877** | 0.5879 |

新 C0/C1 相对旧 S10：LPIPS 降 0.0075/0.0084、PSNR +0.53/+0.60、HF 相关 +0.053/+0.062；但 CSD Keep 仍低于官方（0.848–0.853 vs 0.961），HF 能量比仍偏低（0.588–0.590 vs 0.986）——**高频幅值衰减未被完全修复**，λ_e 保持 0 是正确的保守选择。

**第 2 层（微扰/Jacobian，c0 样本）**：small band cos 0.6342 / amp 0.682 / HF corr 0.2284；medium band cos 0.7252 / amp 0.7495 / HF corr 0.2938；两类输入梯度 `loss_grad_cos=0.2415`、`jtv_cos_median=0.9794`。注意 J^T v 用的是同一低频方向 v，余弦高与「LPIPS loss 梯度余弦 0.24」是**不同量**，不构成矛盾；但 J^T v 0.98 明显高于方案 0.5–0.6 的预期区间，需在解读时说明是低频方向所致，不能据此宣称梯度代理接近同构。

**第 3 层（五生成器替换，750 任务 / 600 cross + 150 same）**：4 变体 × 5 arm = **20 组全部完成**，75 对按源图聚类配对 Bootstrap。Oracle1 的 CSD Top-1：sdvae 46.4% / old_s10 50.1% / c0 48.9% / c1 48.3%；ScoreDiff_N12 的 CSD margin 在所有变体上都是负值（−0.086 ~ −0.093），这是历史已知现象，新 Decoder 未修复。完整逐组结果见 `generator_swap_results.json`。

**第 4 层（Infra）**：`eval_p5_final.json` 内含 PyTorch forward / forward+input-grad 的 ms 与峰值显存；**TensorRT 实测仍未做**。

## 3. 交付物

| 文件 | 来源 | 说明 |
|---|---|---|
| `sd0_s10_best_recon.pt` | c0_best | 最佳真实重建 Pareto 点 |
| `sd0_s10_best_response.pt` | c1_best | 重建约束下最佳响应一致性 |
| `sd0_s10_best_overall.pt` | c1_best | 两 Gate 均通过，默认推荐 |
| `export_c0_best/` / `export_c1_best/` | 导出 | 独立 `decoder.py` + `decoder.pt` + `config.json`，standalone 与训练包装器 **max\|diff\|=0** |
| `P5_FINAL_REPORT.json` | 汇总 | 三层评测 + 四 Gate + `NEGATIVE_REPORT.json` + 3 artifacts |

每个 checkpoint 含 `state_dict`(140 keys) / `post_quant_conv` / `opt` / `sched` / `meta`（Encoder 标识+双 sha256、结构、未缩放 latent 约定、`needs_post_quant_conv`）。

## 4. 链异常与修复记录

链在 step-9 `final eval` 因 **teacher(49.49M) 在 bs=8 带 autograd 做 LPIPS 输入梯度 OOM（21.57GB）** 退出，无 COMPLETE 标记。`guard_p5.sh` 按降级路径：写 `NEGATIVE_REPORT.json`（`CHAIN_INCOMPLETE_C1_OR_LATER`）、`.chain_status=2`、仍拉起保守 P5，`P5_FINAL_REPORT.json` 记 `status=complete_after_layer3_rerun`、`chain_status=2`。**权重零丢失**。

P5 首轮两个失败步骤已定位并修复，**全量重跑通过**：

| 失败 | 根因 | 修复 |
|---|---|---|
| layer124 OOM | `perturb_metrics` 用 8 张 512² 图对 teacher 建 autograd 图 | input-gradient 段改 **2 张**、每次 `autograd.grad` 后显式 `del` + `empty_cache`；全量 0 OOM |
| layer3 `ModuleNotFoundError: config_schema` | `tools.v6.score_diff` 连锁导入 `utils.inference` → 不存在的顶层 `config_schema` | 改用轻量 `tools.v6.score_edit`（同算法同签名，仅依赖 wavelet + models_v6） |
| layer3 `CLIPModel.encode_image` 不存在 | HF transformers API 误用 | 改 `get_image_features(pixel_values=x)` |
| layer3 shape mismatch `[5,128] vs [5,32]` | 把 ScoreDiff 的 ConditionalScoreNet 权重误加载进 `C3UNet(base_dim=16)` | gens 只构造 Oracle1/B4/B3 三个 C3UNet；ScoreDiff arm 走独立 `sdnet` + `edit_with_score_difference` |

修复后小规模 smoke（含 ScoreDiff arm）与全量 750 任务均通过，0 errors。

## 5. 仍未完成（如实）

1. **TensorRT 实测**（Infra 只测 PyTorch forward / forward+input-grad）
2. **P4 感知特征校准**（可选阶段，未编排；CSD Keep 仍明显低于官方，若后续要追平需人工决定）
3. 「无锐化伪影」「大扰动无新增伪影」两项**需人工视觉网格判定**，`audit_gates.py` 如实标 `None`，未臆断
4. 链内 step-9 的 OOM 使 `FINAL_REPORT.json`（总控版本）未生成；已有等价的 `P5_FINAL_REPORT.json` 覆盖全部内容
5. `jtv_cos_median≈0.98` 需在最终发布说明里注明是低频方向指标，不能当作「梯度代理合格」的证据

## 6. 结论

**最低成功标准达成**：保持 S10 结构（10.66M，推理加速预期不变），真实重建与高频保真明显优于旧 S10（LPIPS −0.0084、PSNR +0.60、HF 相关 +0.062），两阶段 Gate 均通过，C1 响应蒸馏带来额外小幅改善且未损伤重建。**主要成功标准部分达成**：响应误差下降、resp_cos 上升、干净重建保持，但独立 L2 的 J^T v 数值需谨慎解读；CSD Keep 与 HF 能量比仍与官方有差距，不能宣称「完全修复 CSD shortfall」。**进一步成功（修复 ScoreDiff 的 Decoder 替换损失）未达成**：ScoreDiff_N12 的 CSD margin 在所有变体上仍为负。按方案保留多个 Pareto 候选，不人为指定单一冠军。
