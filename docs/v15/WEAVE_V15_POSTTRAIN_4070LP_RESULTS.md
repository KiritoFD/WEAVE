# V15 Post-Training on RTX 4070 Laptop — Results

Date: 2026-11-16. Hardware: RTX 4070 Laptop 8GB (peak observed ≤45MB train / ~3.7GB eval, well under 7GB cap).
Plan: `docs/v15/WEAVE_V15_POSTTRAIN_4070LP.md`. Decoder: **S10** throughout (`src.utils.load_s10_decoder`/`decode_s10`).

## 0. Baseline reproduction — VERIFIED

`W_dim96_conv_base` (dim96/conv/adaln/4experts, 4000 updates, ckpt from 3060) re-evaluated locally:
`python -m exp.v15.eval_4070lp --arms W_dim96_conv_base --levels 0,0.4,0.7 --cfg 1 --batch 8 --decode-batch 1 --steps 20`
then `src.eval.measure` on all three levels.

**Pixel-level check vs 3060 raw payloads** (`t0_0.4/batch_00000.pt`, pulled over ssh):
- `z_final` max |Δ| = 0.0059 (fp16 quantization scale)
- `rgb_u8` max |Δ| = 1–2/255, mean |Δ| ≈ 0.03–0.04/255

→ generated images are identical modulo GPU bf16 nondeterminism. The residual quick-metric offset
(dino_s −0.002~0.003 vs GROUP_A §2.4; dino_c/top1 match to 4 decimals) is a **referee-feature
version difference** (local DINOv2 weights vs 3060's), not a protocol error. All post-train
comparisons below use the same local referee, so they are internally consistent.

## 1. Post-training arms (all init from `runs/v15/W_dim96_conv_base/final.pt`)

| arm | mode | updates | micro×accum (eff) | lr | wall | peak VRAM |
|---|---|---|---|---|---:|---:|
| PT_A_balance | balance 5-style FM | 1200 | 8×16 (128) | 4e-5 | 477s | 45MB |
| PT_B_band | balance + Haar H2/H1 velocity weight (0.75/0.25) | 1200 | 8×16 (128) | 4e-5 | 673s | 45MB |
| PT_C0_balance32 | balance, small-batch control | 1000 | 2×16 (32) | 2e-5 | 473s | 45MB |
| PT_C_guided | R1-critic guided cross-style edit (k=2, τ0=0.7, λs=.10 λc=.20 λd=.02) | 1000 | 2×16 (32) | 2e-5 | 727s | 58MB |

Training curves: `runs/v15/PT_*/train_curves.jsonl`. Script: `exp/v15/posttrain_4070lp.py`
(drop-in for the missing ZIP script; balance = equal per-style sampling, guided = unrolled
k-step SDEdit + frozen-R1 margin/content/Δ-to-frozen losses).

## 2. Authoritative results (`src.eval.measure`, cross600 / same150)

### t0 = 0.4

| arm | dino_s | dino_c | lpips | CSD Top-1 | CSD margin | same dino_s | same Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| W_dim96_conv_base | .5212 | .4421 | .4960 | .1533 | −.1769 | .6066 | .4333 |
| PT_A_balance | .5222 | .4478 | .4903 | .1450 | −.1754 | .6115 | .4400 |
| PT_B_band | .5231 | .4491 | .4890 | .1500 | −.1774 | .6130 | .4400 |
| PT_C0_balance32 | .5218 | .4464 | .4916 | .1483 | −.1752 | .6104 | .4333 |
| PT_C_guided | **.2917** | **.1317** | **.8657** | .2000 | −.1698 | .2926 | .1933 |

### t0 = 0.7

| arm | dino_s | dino_c | lpips | CSD Top-1 | CSD margin | same dino_s | same Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| W_dim96_conv_base | .4590 | .7506 | .2700 | .1083 | −.2362 | .6378 | .5667 |
| PT_A_balance | .4587 | .7535 | .2665 | .1067 | −.2374 | .6382 | .5733 |
| PT_B_band | .4596 | .7540 | .2654 | .1017 | −.2386 | .6391 | .5667 |
| PT_C0_balance32 | .4586 | .7526 | .2673 | .1067 | −.2368 | .6379 | .5733 |
| PT_C_guided | **.3361** | **.2621** | **.7661** | .1117 | −.1440 | .3742 | .2400 |
| PT_C_guided cfg2 | .2923 | .1790 | .8582 | .1700 | −.1614 | .3122 | .2000 |
| PT_C_guided shift+2 | .3358 | .2652 | .7693 | .1883 | −.1276 | .3709 | .3267 |
| BASE_H2_old (V11b ckpt_100, H2) | .4587 | .7504 | .2724 | .1133 | −.2362 | .6376 | .5533 |
| BASE_H2_wrong | .4589 | .7503 | .2722 | .1067 | −.2382 | .6384 | .5600 |
| BASE_H2_c256 (RGB-retrained, distinct5_512 256px, 2000 upd) | .4588 | .7502 | .2724 | .1133 | −.2365 | .6379 | .5533 |
| BASE_H2_c256_wrong | .4587 | .7501 | .2721 | .1050 | −.2381 | .6388 | .5600 |
| H2S_strong (H2_c256, τ0=.3 K=8 α=1.0 edge.3 rms3) | .4647 | .7409 | .3041 | .1233 | −.2545 | — | — |
| H2S_both (same, band=both) | .4649 | .7402 | .3171 | .1250 | −.2514 | — | — |
| H2S_strong_wrong | .4664 | .7405 | .3026 | .1200 | −.2633 | — | — |

## 2.1 Paired source-cluster bootstrap vs baseline (cross600, Δ vs W_dim96_conv_base)

Format: `metric:Δ[CI95]` (2000 resamples over 150 source clusters).

| arm | t0 | dino_s | dino_c | lpips | Top-1 | margin |
|---|---|---|---|---|---|---|
| PT_A | .4 | +.0010[−.0001,+.0021] | +.0058[+.0043,+.0072] | −.0057[−.0066,−.0048] | −.0083[−.0167,0] | +.0015 |
| PT_A | .7 | −.0003 | +.0028[+.0022,+.0035] | −.0035[−.0040,−.0031] | −.0017 | −.0012 |
| PT_B | .4 | +.0020[+.0008,+.0030] | +.0071[+.0054,+.0086] | −.0070[−.0082,−.0059] | −.0033 | −.0005 |
| PT_B | .7 | +.0007 | +.0034[+.0026,+.0042] | −.0046[−.0052,−.0040] | −.0067[−.0133,−.0017] | −.0024 |
| PT_C0 | .4 | +.0006 | +.0043[+.0032,+.0055] | −.0044[−.0053,−.0036] | −.0050 | +.0016 |
| PT_C0 | .7 | −.0004 | +.0020[+.0015,+.0025] | −.0027[−.0031,−.0023] | −.0017 | −.0006 |
| PT_C | .4 | **−.2295** | **−.3103** | **+.3697** | +.0467 | +.0070 |
| PT_C | .7 | **−.1228** | **−.4885** | **+.4961** | +.0033 | +.0922 |
| BASE_H2_old | .7 | −.0003 | −.0002 | +.0025 | +.0050 | −.0000 |
| BASE_H2_c256 | .7 | −.0001 | −.0004 | +.0024 | +.0050 | −.0003 |

Reading: A/B/C0 share one signature — tiny but CI-significant **dino_c ↑ / lpips ↓**
(i.e. slightly more conservative editing) with Top-1 flat-to-slightly-down. No arm
moves the style axis; PT_C's "Top-1/margin gains" accompany catastrophic content
collapse and are not style transfer.

## 3. Verdicts

- **PT_A / PT_B / PT_C0: near-neutral, one small real effect.** Paired bootstrap shows a
  consistent, CI-significant **dino_c +.002~.007 / lpips −.003~.007** (slightly more
  content-faithful editing) across all three recipes, while dino_s and Top-1 stay flat.
  Three different recipes (eff.batch 128 vs 32, lr 4e-5 vs 2e-5, band weighting) converge
  to the same point. **No recipe moves the style axis** — consistent with GROUP_A's
  conclusion that the bottleneck is conditioning injection, not optimization.
- **PT_C_guided: FAILED.** R1-margin-guided finetune collapsed the model (dino_c .75→.26 at
  t0.7): the critic margin is gameable without preserving the flow geometry; Δ-to-frozen
  (λ=.02) was far too weak a leash. cfg2 worsens it; wrong-style shift paradoxically *raises*
  Top-1 (.188 vs .112) — signature of style-agnostic collapse, not transfer.
- **BASE_H2_old / wrong: no-op.** V11b ckpt_100 H2-band edit changes rgb by mean 0.8/255;
  all metrics within ±0.005 of baseline.
- **BASE_H2_c256 @ default strength: no-op.** RGB-retrained flow (2000 updates on
  distinct5_512, 1073s, peak 34MB) produces statistically identical edits to the old
  ckpt — the edit_bands operating point is the bottleneck, not the flow weights.
- **H2S strength sweep (exploratory, beyond plan defaults):** τ0=.3 K=8 α=1.0
  edge-decay=.3 rms-clip=3.0 finally produces measurable edits — but the trade is
  the wrong direction: dino_s +.006 while dino_c −.010, lpips +.034, margin −.018.
  band=both is worse (lpips +.047). **Wrong-style control is indistinguishable from
  correct style** (xS .4664 vs .4647, xT1 .1200 vs .1233) → the residual signal is
  generic high-frequency painterliness, NOT target-style transfer. RGB-H2 route
  judged negative at every tested operating point.

## 4. Protocol notes / discrepancies

- Concurrency: running >3 eval/measure processes on the 8GB card caused CUDA OOM twice
  (PT_C shift2 eval, 3 measures) — all were rerun serially and completed. Keep ≤2 GPU jobs.
- `h2_apply_4070lp.py` initial bugs fixed: per-style sub-batch indexing and (B,3,H,W)
  channel order of `rgb_u8` payloads.
- Guided-mode loss components at end of training: style −10.7, content 0.59, delta 0.50 —
  the model moved far from frozen (delta 0.5 in latent MSE) which is exactly what the eval shows.
