# V6 自研胜出模型与评估结果（Ours）

本目录归档在统一评测集（`tasks_750.json` 600 跨风格任务，`paper_rgb_v1` 协议）上经过严密验证胜出的核心模型 Checkpoint 与逐任务评测 JSON，已纳入 Git 版本控制。

---

## 一、胜出模型清单

| 模型名 | 参数量 | 文件路径 | Target Top-1 ↑ | Target Margin ↑ | DINO-C ↑ | LPIPS ↓ | 定位与核心价值 |
|---|---:|---|---:|---:|---:|---:|---|
| **Q2a + R1 Oracle (1 step)** | 218K+816K | `checkpoints/Q2a_embed_style.pt`<br>`checkpoints/r1_r1.pt` | **37.2%** | **-0.0536** | **0.6966** | **0.5438** | **全榜第 5 名**，推理上限与蒸馏教师（超越 AdaIN、AesPA、WEAVE 原版） |
| **P1_B4_distill50 (218K)** | 218K | `checkpoints/P1_B4_distill50.pt` | **24.2%** | **-0.1201** | **0.7960** | **0.4744** | **单步 Direct Top-1 最高纪录**，超越 WEAVE 原版（16.7%） |
| **P1_B3_distill20 (218K)** | 218K | `checkpoints/P1_B3_distill20.pt` | **20.8%** | **-0.1373** | **0.8098** | **0.4620** | **保真档（DINO-C ≥ 0.80）第 1 名**，风格与内容平衡最佳点 |
| **V6 ScoreDiff N=24** | 3.27M | `checkpoints/score_diff_net.pt` | **35.5%** | **-0.0352** | **0.5725** | **0.5455** | 强风格多步扩散参考基线 |
| **V6 ScoreDiff N=12 strong** | 3.27M | `checkpoints/score_diff_net.pt` | **30.8%** | **-0.0588** | **0.6242** | **0.4919** | 12 步平衡强风格参考 |
| **Q2a Direct (218K)** | 218K | `checkpoints/Q2a_embed_style.pt` | **7.0%** | **-0.2479** | **0.8703** | **0.4001** | 嵌入均值对齐单步基准 |
| **P1_B0_ctrl218 (218K)** | 218K | `checkpoints/P1_B0_ctrl218.pt` | **7.5%** | **-0.2635** | **0.8702** | **0.4051** | 续训无蒸馏严格对照 |
| **L11 Direct (218K)** | 218K | `checkpoints/L11_k1024_dim16.pt` | **3.5%** | **-0.2953** | **0.8909** | **0.3619** | 原始轻量 218K 基线 |

---

## 二、评测结果文件对齐（`evals/`）

每个 JSON 文件包含 600 跨风格任务 + 150 同风格任务的逐任务打分、5×5 混淆矩阵和分目标明细：

- `eval_750_oracle_s1.json`: Q2a + R1 Oracle 1-step 评测结果。
- `eval_750_P1_B4_distill50.json`: B4 蒸馏 (λ=50) 评测结果。
- `eval_750_P1_B3_distill20.json`: B3 蒸馏 (λ=20) 评测结果。
- `eval_750_P1_B0_ctrl218.json`: B0 续训对照评测结果。
- `eval_750_Q2a_embed_style.json`: Q2a Direct 单步评测结果。
- `eval_750_L11_k1024_dim16.json`: L11 Direct 基线评测结果。
- `eval_750_scorediff_N24.json`: ScoreDiff 24 步评测结果。
- `eval_750_scorediff_N12.json`: ScoreDiff 12 步评测结果。

---

## 三、模型加载与推理说明

### 1. 单步 Direct 模型（B3, B4, Q2a, L11）
```python
import torch
from tools.v6.models_v6 import C3UNet
from tools.v6.canonicalize import ContentCanonicalizer

ckpt = torch.load("docs/baseline/ours/checkpoints/P1_B3_distill20.pt", map_location="cpu")
model = C3UNet(base_dim=16, num_styles=5, mode="direct")
model.load_state_dict(ckpt["model"])
model.eval()

# 推理 (输入 latent c 和 target style_id)
# z_out = model(c, 0.0, c, target_style_id)
```

### 2. Q2a + R1 Oracle（双模型协作推理）
```python
import torch
import torch.nn.functional as F
from tools.v6.models_v6 import C3UNet
from tools.v6_next.probe_style import LatentStyleNet

# 1. 加载 Direct 生成器
ck_g = torch.load("docs/baseline/ours/checkpoints/Q2a_embed_style.pt", map_location="cpu")
gen = C3UNet(base_dim=16, num_styles=5, mode="direct").eval()
gen.load_state_dict(ck_g["model"])

# 2. 加载冻结 R1 Critic
ck_c = torch.load("docs/baseline/ours/checkpoints/r1_r1.pt", map_location="cpu")
critic = LatentStyleNet(width=64).eval()
critic.load_state_dict(ck_c["model"])

# 3. 前向与单步梯度引导
# z0 = gen(c, 0.0, c, target_id)
# z_req = z0.detach().clone().requires_grad_(True)
# q = critic(z_req.float()).float()
# m = q.gather(1, target_id[:, None]).squeeze(1) - torch.logsumexp(q.masked_fill(oh, -1e9), dim=1)
# g = torch.autograd.grad(m.sum(), z_req)[0]
# z_oracle = z0 + 0.25 * (g / (g.norm(dim=1, keepdim=True) / D**0.5 + 1e-12))
```
