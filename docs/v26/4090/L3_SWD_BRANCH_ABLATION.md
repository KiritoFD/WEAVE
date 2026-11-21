# L3 — 动态 SWD × 全局/局部参考分支消融 (V26-L)

> 生成时间: 2026-11-21T13:01:26.407521+00:00 ｜ init: L1 best=F2_global_teacher (Top-1 16.00%) ｜ 梯度尺度校准: {'swd': 0.01231, 'bary': 0.03203}

| 臂 | 风格损失 | 参考注入 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Top-1 ↑ |
|---|---|---|---:|---:|---:|---:|
| S0_vgg_global | VGG均值 | Global | 0.68126 | 0.48612 | 0.40262 | 16.17% |
| S1_swd_global | SWD | Global | 0.68824 | 0.48216 | 0.38804 | 14.00% |
| S4_bary_global | UOT Bary-SWD | Global | 0.68431 | 0.48223 | 0.39372 | 14.17% |
| S2_vgg_refattn | VGG均值 | Global+RefAttn | 0.68151 | 0.48713 | 0.40499 | 16.00% |
| S3_swd_refattn | SWD | Global+RefAttn | 0.68983 | 0.48241 | 0.38771 | 14.67% |

## 判读 (docs/1120/08 §L3 + 局部性理论 §5.2)

- S1>S0: 分布级风格监督有效;  S4>S0 且困难画派起飞: 摊销式局部性 (Bary-SWD) 有效;
- S2>S0: 局部注入有效;  S3>S1: SWD 下局部有额外价值;
- 全部无收益: 回退最简 Global-only. 不能只看 DINO-S — 必须同看 DINO-C/LPIPS/困难风格/灰化.
