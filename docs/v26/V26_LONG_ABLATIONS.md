# V26-L 消融总表

## L0 采样因果 (1200 upd ×3)

| 臂 | DINO-C | DINO-S | LPIPS | Top-1 |
|---|---:|---:|---:|---:|
| A0_legacy_nbr | None | None | None | 0.00% |
| A1_strict_cross | None | None | None | 0.00% |
| A2_core_cross | None | None | None | 0.00% |

## L1 任意参考化

| 臂 | DINO-C | DINO-S | LPIPS | Top-1 |
|---|---:|---:|---:|---:|
| F1_global_self | 0.7089 | 0.47521 | 0.36236 | 10.83% |
| F2_global_teacher | 0.67704 | 0.48625 | 0.41503 | 16.00% |

## L2 Decoder 归因

| 解码器 | DINO-C | DINO-S | LPIPS | Top-1 |
|---|---:|---:|---:|---:|

## L3 SWD × 分支 2×2

| 臂 | DINO-C | DINO-S | LPIPS | Top-1 |
|---|---:|---:|---:|---:|
| S0_vgg_global | 0.68126 | 0.48612 | 0.40262 | 16.17% |
| S1_swd_global | 0.68824 | 0.48216 | 0.38804 | 14.00% |
| S2_vgg_refattn | 0.68151 | 0.48713 | 0.40499 | 16.00% |
| S3_swd_refattn | 0.68983 | 0.48241 | 0.38771 | 14.67% |

## L4 Rollout 上界

| 臂 | DINO-C | DINO-S | LPIPS | Top-1 | GPU s |
|---|---:|---:|---:|---:|---:|
| T0_preview_1step | 0.64464 | 0.48647 | 0.544 | 16.67% | 773.0 |
| T1_rollout_trunc | 0.74163 | 0.44061 | 0.34315 | 9.17% | 1034.0 |
| T2_rollout_full | 0.73877 | 0.43816 | 0.34434 | 8.00% | 1413.7 |
