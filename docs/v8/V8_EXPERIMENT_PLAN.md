# WEAVE V8 实验计划（按 docs/bridge.md 实现）

## 已实现范围

- 独立 `tools/v8/` 模块，**不导入任何 ScoreNet / score_diff / Real-Fake Score / DMD**。
- D5 scale 一致性：读取 `pseudo_gt_24h/orig_mean_fp16/*.pt`，校验所有缓存共用 `stabilityai/sd-vae-ft-ema`、`scaling_factor=0.18215`；训练/配对/Bridge/ODE 使用同一已缩放 latent。
- Random 与 Content-OT 端点配对：C1 pooled LL + Spatial Self-Similarity 特征、FP32 log-domain Sinkhorn、真实目标端点采样与覆盖率记录，不做 barycentric average。
- Brownian/CFM 路径：解析 `z_tau` 与 `v_tau`，`sigma_mid=0` 退化为 CFM；端点精确、有限差分测试通过。
- 2M 参数 `VelocityUNet`：连续时间 SinEmbedding + Source/Target pair embedding + AdaLN U-Net；输出速度，不输出残差，不使用 clamp。
- Euler / Heun solvers，NFE≤16，s==t 由接口严格返回输入。
- 统一训练器：B0–B3 共享 optimizer updates、有效 batch、seed、LR schedule、源/目标采样序列；记录 loss、target RMS、分时间桶 MSE、梯度范数、EMA。
- Evaluator V2 风格评测：paper_rgb_v1、references_v1、Target Top-1、Margin、CSD ΔT、DINO-C、LPIPS、五类 per-target、混淆矩阵；VAE No-op 作为 ΔT_edit 基线。
- 配对分层聚类 Bootstrap 汇总脚本。

## 验收流程

1. P0：本地与 3060 数学单测、数据划分、OT 配对审计、100 更新 benchmark。
2. P1：B0/B1/B2/B3 同预算训练 800 updates，sigma_mid 短筛先取 0.10（文档候选 0.10/0.25 中较低且稳定者）。
3. 独立 Train-split 200 任务评估 4/8/16 NFE，选择 Pareto 点；禁止在 Test600 上选参数。
4. 若 Go/No-Go 通过，再在冻结 Test600+Same150 评测 4/8/16 NFE 并写 `V8_RESULTS.md`。
5. 仅当第一阶段有效才实现 B4/B5；当前 `online_dsbm.py` 为显式 gate，不伪造实现。

## 当前工程裁决

- V8 完全去 Score 的训练与推理链路已实现。
- B4/B5 严格 α-DSBM 暂不实现，等待 B0–B3 第一阶段 Go/No-Go。
- 旧 V6/V7 Score 代码保留为历史基线，不进入 V8 依赖链。
