# WEAVE V7 DMD 实现修复与重实验报告

> 日期：2026-11-08  
> 触发原因：用户指出提交 `d5100c0` 中 Pure/Hybrid DMD 的核心梯度方向可能反了，并且残差截断存在训练/推理不一致与无效 Identity Loss。  
> 本轮严格按 P0→P1→P2→P3 顺序执行，不改网络结构，不在同一 GPU 上并行跑两个训练任务。

---

## 一、P0：一维高斯解析单元测试（确认旧方向错误）

测试设置：
\\[
p=N(0,1),\\quad q=N(m,1),\\quad m=2,\\quad \\sigma=0.8
\\]

解析 epsilon prediction：
\\[
\\epsilon_{\\rm real}=\\sigma y_\\sigma/(1+\\sigma^2),\\quad
\\epsilon_{\\rm fake}=\\sigma(y_\\sigma-m)/(1+\\sigma^2)
\\]

因此：
\\[
d=(\\epsilon_{\\rm real}-\\epsilon_{\\rm fake})/\\sigma=m/(1+\\sigma^2)>0
\\]

KL 下降的正确更新为 `target = y - alpha*d`；梯度下降会使 \(m\) 减小。实测：

| 实现 | \(dL/dm\) | 一步 GD 后 \(m\) | 方向 |
|---|---:|---:|---|
| **修正后** `target = y - alpha*d` | +0.6248 | **1.9375**（下降，向 p 移动） | **正确** |
| **旧代码** `target = y + alpha*(eps_real-eps_fake)` | -0.4727 | **2.0473**（上升，远离 p） | **错误** |

**结论**：用户判断正确。此前 Pure/Hybrid 内容崩塌不能直接归因于 RDMD 不稳定或旧 VAE latent 质量差，因为核心优化方向本身错误。  
产物：`tools/v7/tests/test_dmd_gradient_sign.py`、`runs/v7/p0_dmd_gradient_sign.json`。

---

## 二、P1：实现修复清单

| 修复项 | 旧实现 | 本轮修复 |
|---|---|---|
| DMD 梯度符号 | `target = y + alpha*(eps_real-eps_fake)` | **`target = y - alpha*(eps_real-eps_fake)/sigma`**，loss 改为 `0.5*MSE(y,target)` |
| 残差限制 | Straight-Through `clamp`，反向梯度穿透饱和区 | **可微 `c*tanh(R/c)`**，由 `SingleStepTransportNet.forward` 统一实现 |
| 训练/推理一致性 | 训练 clamp、评测有时使用未截断残差 | **训练和评测都直接调用同一个 `transport(z,s,t)`** |
| Identity Loss | `s==t` 已由 mask 严格恒等，MSE 恒为 0，无梯度 | 删除无效 loss |
| 梯度监控 | 只看 loss 值 | 记录 `|g_dmd|`、`|g_content|`、`|g_total|`、残差 L2 |

P1 单元测试 `tools/v7/tests/test_transport_consistency.py`：
- 确定性两次前向最大差 0；
- 最大残差 0.34996，严格不超过 cap=0.35；
- `s==t` 残差最大值 0；
- `out_conv` 梯度范数 0.7963，tanh 残差可微；
- 产物：`runs/v7/p1_transport_consistency.json`。

---

## 三、P2：固定 S1 的两个同预算配置重训

训练配置完全一致，只改变 `lambda_content`：

| 配置 | 步数 | batch | `dmd_alpha` | `sigma_dmd` | residual cap | `lambda_content` | 真实 Score |
|---|---:|---:|---:|---:|---:|---:|---|
| `p2_fix_pure_dmd` | 800 | 40 | 0.10 | 0.80 | 0.35 | **0.0** | S1 冻结 |
| `p2_fix_dmd_content` | 800 | 40 | 0.10 | 0.80 | 0.35 | **0.20** | S1 冻结 |

训练中梯度和残差均稳定，无 NaN/爆炸：

| 配置 | final fake | final dmd | `|g_dmd|` | `|g_content|` | `|g_total|` | residual L2 | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure_dmd | 0.3306 | 0.0002 | 0.002 | 0.000 | 0.002 | 38.75 | 5797 MiB |
| dmd_content | 0.2890 | 0.0003 | 0.003 | 0.002 | 0.004 | 33.64 | 5797 MiB |

> 修复符号后，训练本身稳定；但 DMD generator 梯度范数很小（0.002–0.003），说明在旧 VAE latent / S1 Score 下，Score Difference 给 1-NFE 传输网络的有效分布匹配信号很弱。这是比“内容崩塌”更可信的瓶颈描述。

---

## 四、P3：冻结 750 任务评测

评测使用与训练完全一致的 `SingleStepTransportNet.forward`：
- `paper_rgb_v1` 原始 RGB 源图基准；
- 600 跨风格 + 150 同风格；
- `references_v1` 统一参考池；
- NFE = 1；
- 不额外施加 clamp / alpha 缩放。

| 模型 | NFE | CSD ΔT ↑ | Target Top-1 ↑ | Target Margin ↑ | DINO-C ↑ | LPIPS ↓ | 5类改善 | Ukiyo-e Top-1 | Ukiyo-e ΔT | same150 ΔS |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| **Calib_Balanced_s16（历史参考）** | 96 | **+0.0773** | **27.7%** | -0.0792 | 0.6803 | 0.5498 | 5/5 | 6.7% | — | +0.0262 |
| **V7 Fix pure_dmd** | **1** | -0.0177 | 4.0% | -0.2642 | 0.8297 | 0.3259 | 1/5 | 0.0% | -0.0306 | +0.0262 |
| **V7 Fix dmd_content** | **1** | +0.0145 | 1.8% | -0.4053 | **0.9575** | **0.1660** | 3/5 | 1.7% | +0.0226 | +0.0262 |

### 关键观察
1. **内容崩塌确实被消除了**：修正符号 + tanh 残差后，`dmd_content` 的 DINO-C 达 0.9575、LPIPS 仅 0.1660，`pure_dmd` 也保持 DINO-C 0.8297。此前 DINO-C≈0.08 的崩塌不再出现。
2. **但风格迁移仍极弱**：Top-1 仅 1.8%–4.0%，ΔT 不足 +0.02，Ukiyo-e Top-1 0–1.7%，远低于 Calib_Balanced_s16。
3. **梯度范数很小**：修正方向没有带来有效目标风格建立，说明当前 frozen S1 Score 与旧 VAE latent 之间的 `(eps_real - eps_fake)/sigma` 对生成器的分布匹配信号弱，而不是单纯“方向反了就能解决全部问题”。

---

## 五、修订后的科学裁决

必须把上一轮结论更新为：

- **旧实现确实存在严重 bug**：DMD 符号反了、Straight-Through clamp 不一致、Identity Loss 无效；
- 因此此前“RDMD 不稳定 / 旧 latent 不适合分布传输”的判断**不成立或至少证据不足**；
- 但修复后，在旧 VAE latent + 800 步短预算 + S1 Score 下，**1-NFE 传输网络仍没有达到风格迁移验收阈值**；
- 当前更准确的瓶颈表述是：**修正方向后训练稳定、内容不崩塌，但 Score Difference 对 generator 的有效梯度范数过小，尚未把输出分布推向目标画风**。

这与用户提出的最终架构判断并不冲突：

\\[
\\boxed{\\text{训练：Real Score + Fake Score + TransportNet}}
\\]

\\[
\\boxed{\\text{推理：TransportNet（1 NFE）}}
\\]

架构仍合理；但实验表明，要让该架构真正工作，后续至少需要：
1. 在 **新 VAE latent** 上重新训练 S1 Score 与 Fake Score；
2. 增大 / 重新校准 `dmd_alpha`（当前 `|g_dmd|≈0.002` 过小）；
3. 监控 real/fake Score difference 的每类范数，尤其 Ukiyo-e；
4. 可能需要更长时间或更接近 RDMD 理论的两阶段训练，而不是继续只调内容约束。

---

## 六、产物清单

- `tools/v7/tests/test_dmd_gradient_sign.py`：P0 解析符号测试；
- `tools/v7/tests/test_transport_consistency.py`：tanh 残差、确定性、s==t 恒等、可微性测试；
- `tools/v7/models_v7.py`：`c*tanh(R/c)` 平滑残差统一输出；
- `tools/v7/p2_train_rdmd_fix.py`：修正符号后的纯 DMD / DMD+内容约束训练器；
- `tools/v7/p3_eval_v7_fix.py`：与训练一致输出的 750 全量评测；
- `runs/v7/p0_dmd_gradient_sign.json` / `p1_transport_consistency.json`；
- `runs/v7/p2_fix_*_train_summary.json` / `p3_v7_fix_eval750_results.json`；
- `docs/v7/V7_DMD_FIX_EVALUATION_REPORT.md` / `V7_DMD_FIX_ROUTE_REPORT.md`。
