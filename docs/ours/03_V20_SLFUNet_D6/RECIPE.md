# V20 SLFUNet Core (slf_core_v2.pt)

> **定位**：V20/V25 主干基座 (D5/D6/D6H/Hybrid 核心权重)｜**状态**：`active`
> **图像目录**：`images_750/` (750 PNGs, 源 `M0_v20_d6`)｜**配置**：[`config.json`](config.json)

## 1. Checkpoint 清单

| 本地文件 | 原训练路径 | 大小 | SHA256 (前 16 位) |
|---|---|---:|---|
| `slf_core_v2.pt` | `runs/v20/slf_flow/slf_core_v2.pt` | 83.76 MB | `85737c5331895032...` |

## 2. 架构与训练配方

- **架构**：SLFUNet(base_dim=144, u_dim=128), 21.95M, Coarse-16 瓶颈语义注入 (DINO-16 + y16_lcn)
- **训练配置**：
```json
{
  "script": "exp/v20/run_v20.py",
  "conditioning": "Core-kNN8 原型 + 5 类 onehot + layout_scale + drop_style/drop_content",
  "objective": "Rectified Flow Matching MSE + y16_lcn structure loss"
}
```

## 3. 推理配置

```json
{
  "anchor": "T4 (V19 Anisotropic S10-VGG)",
  "branches": "full (32 img-eq) or cs_j (16 img-eq, 20 NFE notation)",
  "solver": "D6_SOLVER (tau in [0.40,0.74]) or D6H (tau in [0.48,0.78])",
  "gate": "original sigmoid((dsd-mid)*550)",
  "decoder": "S10"
}
```

## 4. `test750` (`cross600`, `S10` 解码) 实测指标

| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Affinity ↑ | Top-1 ↑ | Same-150 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `M0_v20_d6` | 0.6688 | 0.5107 | 0.5374 | 0.6145 | 55.67% | 96.7% |
| `B_d6_t066` | 0.7007 | 0.5028 | 0.5213 | 0.6071 | 55.00% | 96.7% |
| `P02_d6h_csj` | 0.6782 | 0.5121 | 0.5350 | 0.6126 | 55.00% | 96.7% |
| `P02_d6h_full` | 0.6820 | 0.5073 | 0.5315 | 0.6112 | 55.17% | 96.7% |

## 5. 说明

V25 Round 1/2/R1b 全程基准权重; T4 锚下 tau_min=0.72 饱和于 DINO-C=0.7055, full CFG 对 Impressionism->Minimalism 去杂乱路线不可删。
