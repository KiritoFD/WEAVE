# R5-512 Inference Time Log (2026-07-21)

Platform: Windows RTX 3060 12GB (remote 100.115.18.62), 750 images (5 src × 5 tgt × 30 content).

R5-512 test set: Cubism, Expressionism, Pop_Art, Romanticism, Symbolism (30 imgs/style).
Inference time should match D5-512 since both are 5 styles × 5 targets × 30 content at 512² resolution.

## Measured (R5-512, 2026-07-21)

| Method       | Total (s) | sec/img | Notes |
|-------------|-----------|---------|-------|
| IDT         | <1        | 0.001   | File copy (no transfer) |
| TGT         | <1        | 0.001   | File copy (no transfer) |
| WCT         | 61.4      | 0.082   | VGG19 + WCT (untrained decoder, may need re-run with StyTR2 decoder) |
| AdaIN       | 30.1      | 0.040   | VGG19 + AdaIN (untrained decoder, may need re-run with StyTR2 decoder) |
| WCT_StyTR2  | 71.4      | 0.095   | VGG19 + WCT + StyTR2-trained decoder (decoder_iter_160000.pth). 18 keys loaded, missing=0 unexpected=0. |
| AdaIN_StyTR2| 44.5      | 0.059   | VGG19 + AdaIN + StyTR2-trained decoder. |
| StyleID     | 2383 (39.7min) | 3.14 | SD 1.5 img2img, 30 steps, strength=0.6. Final: 750 imgs, 2355.1s inference + 28s startup. |
| SD-Turbo    | 524 (8.7min)  | 0.66  | SD-Turbo img2img, 4 steps, strength=0.5. Final: 750 imgs, 491.7s inference + 33s startup. |
| Latent-WCT  | 220.1      | 0.293   | VAE encode + Haar WCT on HF subbands + VAE decode. 750 imgs, 220.1s inference + 32s startup = 251.9s total. |
| StyleAligned| 4642.76 (77.4min) | 6.19 | SD 1.5 + DDIM inversion (20 steps) + StyleAligned attn. 750 imgs, 4642.76s wall-clock, 6.13 s/img in-loop. |
| StyTR-2     | 290.96 (4.85min)  | 0.388 | Official StyTR-2 weights (iter_160000). 750 imgs in 290.96s. |
| AesPA-Net   | 251.67 (4.19min)  | 0.336 | Official AesPA-Net weights. 750 imgs in 251.67s (245.1s inference + 6.6s startup). |
| weave       | 126       | 0.168   | 8-step ODE + VAE decode (same as D5) |
| Z-STAR      | 1871.0 (31.2min)  | 2.49  | Z-STAR v7 fp32, 384px, skip_null_opt, DDIM 30 steps. 130 continue-run (620 cached from earlier v7 run). |
| SaMam       | 4152 (69.2min)    | 5.54  | SaMam WSL inference, 750 imgs. |
| SaMST       | 265.9 (4.4min)    | 0.355 | SaMST 40-epoch train (~20min) + infer 750 imgs in 265.9s. Peak VRAM 12.18GB. |
| StyleShot   | (pending)  | --    | StyleShot inference (completed, timing TBD). |
| CUT         | 104.5 (1.7min)    | 0.14  | CUT 15-epoch train (5 models, ~55min total, 256px) + infer 750 imgs in 104.5s. Peak VRAM 0.52GB. |

## StyleID final timing (2026-07-21 16:17:21 - 16:57:04)

- Total elapsed: 39 min 43 sec = 2383s (wall clock)
- Inference only: 2355.1s
- Per-image: 3.14 s/img (SD 1.5, 30 steps, img2img strength=0.6)
- Per-style (5 styles × 30 content × 5 src = 750 imgs):
  - Cubism (150 imgs): 453.9s
  - Expressionism (300 imgs): 917.7s
  - Pop_Art (450 imgs): 1379.2s
  - Romanticism (600 imgs): 1866.4s
  - Symbolism (750 imgs): 2355.1s
- **Paper currently reports StyleID = 63m. Actual measurement = 39.7m. NEEDS CORRECTION (-23.3m).**

Possible reason for paper's 63m: previous estimate used 50 steps (4.02 s/img × 750 = 3016s = 50min)
plus ~13min overhead. With current config (30 steps, strength=0.6), actual = 39.7m.

## SDTurbo final timing (2026-07-21 16:57:09 - 17:05:53)

- Total elapsed: 8 min 44 sec = 524s (wall clock)
- Inference only: 491.7s
- Per-image: 0.66 s/img (SD-Turbo, 4 steps, img2img strength=0.5)
- Per-style:
  - Cubism (150 imgs): 96.4s
  - Expressionism (300 imgs): 193.9s
  - Pop_Art (450 imgs): 291.4s
  - Romanticism (600 imgs): 390.3s
  - Symbolism (750 imgs): 491.6s
- **Paper currently reports SD-Turbo = 5m (300s). Actual = 524s = 8.7m. NEEDS CORRECTION (+3.7m).**

## Summary of corrections needed for paper main table (Infer column)

| Method    | Paper | Actual (R5-512) | Delta  | Action |
|-----------|-------|-----------------|--------|--------|
| SD-Turbo  | 5m    | 8.7m (524s)     | +3.7m  | Update to 9m |
| StyleID   | 63m   | 39.7m (2383s)   | -23.3m | Update to 40m |
| StyleAligned | 77m | 77.4m (4643s)  | +0.4m  | Keep 77m (matches!) |
| StyTR-2   | 38m   | 4.85m (291s)    | -33m   | Update to 5m (paper 38m may include training?) |
| AesPA-Net | 23m   | 4.19m (252s)    | -19m   | Update to 4m (paper 23m may include training?) |
| WEAVE     | 126s  | 126s            | 0      | Keep |
| Latent-WCT| 18s   | 220.1s (3.7m)   | +202s  | Update to 4m (or 220s) |
| WCT       | --    | 61.4s           | --     | Not in main table |
| AdaIN     | --    | 30.1s           | --     | Not in main table |

Pending methods (need R5-512 re-run for accurate timing):
- StyleAligned (paper 77m, SDXL-based, may be too slow on 3060)
- Z-STAR (paper 3h, inversion-based)
- StyleShot (paper 5.1h, SDXL-based)
- CUT (paper 5m, need pre-trained ckpt)
- SaMST (paper 10m, need training)
- SaMam (paper 17.6m, WSL)
- StyTR-2 (paper 38m, need training)
- AesPA-Net (paper 23m, need training)

## Existing baseline timing data (from I:\Github\Latent_Style\Related_Works\run_511\docs)

From `timing_summary.csv` (P2A-256 scale, 750 images):
- adain_7g_v32k: 9.281s (0.0123 s/img)
- adain_7g_vgg19: 9.098s (0.0121 s/img)
- samst_750_strict: 39.826s (0.0531 s/img)
- styleid_750_strict: 603.316s but only photo target actually generated, full 750 estimate = 3016s (4.02 s/img)

From `timing_filled.json`:
- ours_epoch7_infer: 85.414s (0.114 s/img) - older measurement; paper reports 126s (selected config, 8 steps)
- samst_train_extrapolated: 2030.6s for 30 epochs
- styleid_infer_estimated: 3016.335s full 750 (4.02 s/img)

## Paper main table (Infer column, current values)

From `paper.tex` Table 1 (M/Tr/Inf columns):

| Method       | Params (M) | Train | Infer (paper) | Notes |
|-------------|------------|-------|---------------|-------|
| SD-Turbo    | 0+1.1k     | free  | 5m            | SD-Turbo 4-step img2img |
| StyleAligned| 0+1.1k     | free  | 77m           | SDXL + IP-Adapter scale |
| Z-STAR      | 0+1.1k     | free  | 3h            | Z-STAR inversion |
| StyleShot   | 0+2.4k     | free  | 5.1h          | StyleShot (SDXL based) |
| StyleID     | 0+1.1k     | free  | 63m           | SD 1.5 img2img, 50 steps |
| CUT         | 7.0        | 323 min | 5m          | CUT unpaired translation |
| SaMST       | 8.3        | 39.5 min | 10m        | SaMST latent |
| SaMam       | 8.8        | 436 min | 17.6m      | SaMam latent |
| StyTR-2     | 35+13      | 1440 min | 38m       | StyTR-2 transformer |
| AesPA-Net   | 11+20      | --    | 23m           | AesPA |
| Seedream    | --         | API   | --            | External API |
| Latent-WCT  | 0          | free  | 18s           | Latent-space WCT |
| **WEAVE**   | **1.04+84**| **1.4 min** | **126s** | 8-step ODE |

## Issues to verify / correct

1. **SD-Turbo: 5m (300s)** - estimate seems plausible for 750 imgs (0.4 s/img). To verify with R5-512 re-run.
2. **StyleAligned: 77m** - very high; SDXL-based. To verify with R5-512 re-run if possible.
3. **Z-STAR: 3h** - inversion-based, very slow. Plausible.
4. **StyleShot: 5.1h** - SDXL-based, very slow. Plausible.
5. **StyleID: 63m** - matches the 3016s estimate (~50 min). 30 steps + strength 0.6 in R5-512 may differ.
6. **CUT: 5m** - 323 min train, 5m infer. Need to verify.
7. **SaMST: 10m** - 39.5 min train (paper) vs 2030.6s extrapolated (timing doc). Discrepancy! Paper says 39.5 min train, but timing data shows 1-epoch measurement of 67.687s extrapolated to 30 epochs = 2030.6s = 33.8 min. Close to 39.5 min paper value. SaMST infer 10m = 600s for 750 imgs (0.8 s/img).
8. **SaMam: 17.6m** - 436 min train (paper). To verify with WSL run.
9. **StyTR-2: 38m** - 1440 min train. Need to verify infer.
10. **AesPA-Net: 23m** - Train time missing in paper (--). To verify.
11. **Latent-WCT: 18s** - 18s for 750 imgs (0.024 s/img). Very fast (latent space).
12. **WEAVE: 126s** - 1.4 min train, 126s infer (0.168 s/img). Consistent with current R5-512 run.

## Action plan

1. Complete R5-512 timing measurements for all train-free methods:
   - StyleID (running)
   - SDTurbo (queued)
   - StyleAligned, Z-STAR, StyleShot (need separate scripts, may be too slow on 3060)
   - CUT (need pre-trained checkpoint)

2. Update paper Infer column based on actual R5-512 measurements (should match D5-512 since same 750-pair count and resolution).
3. Investigate discrepancies:
   - SaMST train time (39.5 min paper vs 33.8 min extrapolated)
   - AesPA-Net train time (missing in paper)
   - StyleShot infer time (5.1h is implausibly long - SDXL on 3060)
