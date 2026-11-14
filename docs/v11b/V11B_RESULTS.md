# V11-B · Wavelet Stroke Flow — Results (4070 Laptop)

Protocol: full 750-task eval (`eval_protocol/tasks_750.json`), cross600 readings,
paired bootstrap CIs vs LD4 baseline (v9 dev+test merged). No dev200 — per user
decision all decisions use the unified 750 eval.

## Pipeline

- LD4 WEAVE outputs (S10-decoded PNGs) → 2-level Haar DWT → keep LL2 fixed,
  regenerate detail band(s) H1/H2 with a style-conditioned conditional flow
  (1.41M params, width=56, trained 8000 steps on 4488 real paintings,
  Flow Matching on wavelet coefficients) → iDWT → PNG.
- Inference: `exp/v11b/run_v11b_750.py`; eval: `tools/eval_images_manifest.py`
  (resumable, `eval_rows.jsonl` + `features_partial.pt`).

## P0 — acceptance

- DWT/iDWT round-trip error 2.4e-7 on real 512px images.
- Per-style band sigma in `docs/v11b/P0_PROFILE.json`
  (Minimalism lowest energy ~0.014/0.038; Ukiyo_e highest ~0.047/0.111).
- Peak memory micro=8 ≈ 0.21GB; training micro=64 ≈ 4.4GB, GPU ~79%.

## P1 — S10 HF loss audit (`docs/v11b/P1_S10_HF_AUDIT.json`)

200 held-out real paintings, SD0-VAE encode → S10 decode → band statistics:

| band | energy retention (mean) | direction cosine (mean) |
|---|---:|---:|
| H1 | 0.688 | 0.561 |
| H2 | 0.786 | 0.753 |

The VAE→S10 round-trip already destroys ~21–31% of detail-band energy and
rotates band directions substantially — consistent with "WEAVE changes colour
and texture but lacks organised brushwork". The wavelet flow targets exactly
this lost band.

## P2 — training / G1 gate

`runs/v11b/flow_v1_w56`, 8000 steps, ~0.15 s/step, ~98 img/s.

G1 (style-conditioned velocity MSE on holdout, step 8000):
correct .0007142 vs wrong .0007335 vs uncond .0007222 (H1);
.0049516 vs .0050008 vs .0049602 (H2) — separation exists but is small
(~2% relative). **G1 formally marginal**: the model learned mostly generic
HF restoration with weak style conditioning. Per plan this is a warning,
not an automatic stop — downstream 750 eval decides.

## P3/P5 — 750-task results (cross600 means)

| config | dino_s | dino_c | lpips | top1 | margin | G | affinity |
|---|---:|---:|---:|---:|---:|---:|---:|
| LD4 (baseline) | .4401 | .7718 | .4872 | .3517 | -.0531 | -.0155 | .5615 |
| H1 t0.6 K4 a0.5 | .4397 | .7715 | .4877 | .3483 | -.0530 | -.0155 | .5613 |
| H2 t0.6 K4 a0.5 | .4413 | .7728 | .4888 | .3467 | -.0535 | -.0157 | .5623 |
| both t0.6 K4 a0.5 | .4408 | .7726 | .4897 | .3550 | -.0532 | -.0158 | .5620 |
| **H2 t0.35 K8 a0.75 cfg1.5** | **.4482** | **.7770** | .4927 | .3333 | -.0568 | -.0162 | .5666 |
| both t0.35 K8 a0.5 cfg1.5 | .4430 | .7738 | .4907 | .3500 | -.0538 | -.0161 | .5634 |
| H1 t0.35 K8 a0.75 cfg1.5 | .4402 | .7712 | .4870 | .3433 | -.0529 | -.0154 | .5616 |

same150 (all configs): top1 .98–.987, G ≈ .207–.209 — saturated, no signal.

### Paired bootstrap vs LD4 (cross600, n=600)

| config | Δdino_s | Δdino_c | Δlpips | Δtop1 | ΔG |
|---|---|---|---|---|---|
| H2 t035 | **+.0081** [+.0068,+.0094] | **+.0053** [+.0042,+.0065] | +.0056 [+.0046,+.0066] | −.0183 [−.038,+.003] | −.0007 n.s. |
| both t035 | +.0029 [+.0021,+.0037] | +.0021 [+.0014,+.0028] | +.0035 [+.0022,+.0048] | −.0017 n.s. | −.0006 sig. |
| both t06 | +.0007 [+.0003,+.0011] | +.0008 [+.0004,+.0012] | +.0025 [+.0017,+.0033] | +.0033 n.s. | −.0003 sig. |

## G2 gate assessment (pre-registered)

Targets: dino_s ≥ .4451, dino_c ≥ .7658, lpips ≤ .4952, top1 no clear drop.

- **H2 t0.35 K8 a0.75 cfg1.5**: dino_s .4482 ✓, dino_c .7770 ✓,
  lpips .4927 ✓ (all point estimates inside gates), top1 −1.8pp with CI
  [−3.8,+0.3]pp — borderline, not clearly significant.
  → **G2: marginal pass on 3/4 hard gates; top1 degradation is the open risk.**
- both t035 fails dino_s gate (.4430 < .4451).
- Conservative t0.6 configs are all within noise of LD4.

## P4 — ablations

- Band: H2 carries most of the gain (H1 alone ≈ null; both dilutes it).
- Strength: tau0 0.6→0.35, K 4→8, alpha→0.75, cfg 1.0→1.5 is the direction
  that helps; further pushes untested.
- Style-shuffle ((sid+2)%5, same H2 t035 config), paired vs correct-style,
  cross600 n=600:

| metric | correct style | shuffled style | Δ (shuffle−correct) |
|---|---:|---:|---:|
| dino_s | .4482 | .4495 | +.0013 [+.0003,+.0022] |
| dino_c | .7770 | .7768 | −.0002 n.s. |
| lpips | .4927 | .4913 | −.0014 sig. |
| target_top1 | .3333 | .3083 | **−.0250** [−.043,−.007] |
| target_margin | −.0568 | −.0633 | −.0065 sig. |
| G | −.0162 | −.0162 | +.0000 n.s. |

  Reading: correct style conditioning buys +2.5pp Top-1 / +.0065 margin —
  real but small. The dino_s/dino_c/lpips movement is almost entirely
  style-agnostic (shuffle does not hurt it). The flow's gain decomposes
  into a large generic-detail component and a small style-specific one.

## Interpretation

The flow's HF regeneration acts mostly as a *generic* detail refiner:
dino_s/dino_c gains are real and paired-significant, but target_top1 and G
do not improve — consistent with G1's weak style separation. This mirrors
the V10.2 lesson: recovery alignment ≠ style specificity. The gain is
"more painting-like detail", not "more target-style detail".

Artifacts: `runs/v11b/out_*` (750 PNGs each), `runs/v11b/eval_*`
(eval.json + eval_rows.jsonl + features.pt), code `exp/v11b/`.
