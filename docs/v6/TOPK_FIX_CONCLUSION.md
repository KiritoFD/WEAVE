# TopK batch-mask 修复复测结论

## 背景

`w_gating.py::topk_spatial_mask()` 旧实现错误地对 batch 维度求均值：

```python
energy = g.detach().pow(2).mean(dim=(0, 1))
```

这会让整个 batch 共享一张 TopK 掩码，导致评测结果依赖 batch 组成和任务排序。修复后改为逐样本独立选 TopK：

```python
energy = g.detach().square().sum(dim=1)  # [B,H,W]
idx = energy.flatten(1).topk(k, dim=1).indices
mask.scatter_(1, idx, 1.0)
```

`apply_gated_edit()` 其余部分（包括 `rms_normalize(g)`）保持不变。

## 修复审计

- TopK25 每图精确选中 1024 个空间位置。
- 旧/新掩码 IoU 均值 0.25，说明旧 batch 共享掩码与逐图掩码明显不同。
- 旧版 batch=1 与修复版 batch=1 latent 基本一致：RMS 差均值 0.000275。
- 旧版 batch=8 与修复版 batch=8 latent 明显不同：RMS 差均值 0.1593。
- 修复版 batch=1 与 batch=8 latent 最大 RMS 差 0.00541，属于 GPU batch 数值差异。
- 修复版任务顺序反转 latent 差为 0.0。

## 完整 750 任务复测结论

修复后 TopK 的真实收益仍成立，但**不再达到 +0.020 的 DINO-S 主门槛**：

| 配置 | 旧 DINO-S | 修复后 DINO-S | 相对 B3 修复后 ΔDINO-S |
|---|---:|---:|---:|
| TopK25 4步 | 0.4372 | 0.4361 | +0.0178 ~ +0.0179 |
| TopK50 4步 | 0.4386 | 0.4370 | +0.0187 |
| TopK25 8步 | 0.4389 | 0.4367 | +0.0185 |

因此此前报告中“TopK50 / TopK25 S8 达到 DINO-S +0.020 主门槛”的结论应更正：该结果受旧 batch 共享掩码影响，不是可靠的逐图空间自适应编辑收益。

## 结论

1. P2 TopK 方向仍有稳定、显著的风格提升，但严格门槛下只能判为 PARTIAL/未通过。
2. 修复前不要继续叠加 Ratio Gate、Proximal、Push–Pull。
3. `full`、`hf`、`content`、`style_energy` 未使用该函数，不受此 bug 直接影响。
4. 详细数值见 `docs/v6/TOPK_FIX_RETEST.md` 与 `runs/v6_w/topk_fix_final_summary.json`。
