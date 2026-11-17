# WEAVE V15 dim96 Conv post-training on RTX 4070 Laptop

This is an independently runnable, all-arm plan. No early quality gates.

## 0. Install scripts and preflight (Windows PowerShell)

Extract the accompanying ZIP and copy four Python files into `G:\WEAVE\exp\v15\`.
The module filenames must match the commands below:

- `v15_posttrain_4070lp.py` -> `posttrain_4070lp.py`
- `v15_eval_4070lp.py` -> `eval_4070lp.py`
- `v15_h2_train_4070lp.py` -> `h2_train_4070lp.py`
- `v15_h2_apply_4070lp.py` -> `h2_apply_4070lp.py`

```powershell
cd G:\WEAVE
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
Test-Path runs/v15/W_dim96_conv_base/final.pt
Test-Path weights_local/sd_pool_orig/00_Early_Renaissance.pt
Test-Path runs/v6_next/probes/r1_r1.pt
Test-Path runs/v9_local_4070/cache/n2c_z.pt
Test-Path runs/v11b/flow_v1_w56/ckpt_100.pt
Test-Path docs/v11b/P0_PROFILE.json
```

The original V15 checkpoint was generated on the 3060. If it is not on the
4070 Laptop, copy the exact checkpoint; no script can recreate its weights.

## 1. Independently finetune the same V15 checkpoint

All experiments use the original `W_dim96_conv_base/final.pt` as SOURCE.
The optimizer and schedule are new, with a low learning rate; this is
post-training, NOT an exact continuation of the original 4k-step run.

```powershell
# A: balanced five-style flow-matching continuation
python -m exp.v15.posttrain_4070lp --mode balance --source runs/v15/W_dim96_conv_base/final.pt --pool weights_local/sd_pool_orig --updates 1200 --micro 8 --accum 16 --lr 4e-5 --out runs/v15/PT_A_balance

# B: balanced + weighted latent H2/H1 velocity error
python -m exp.v15.posttrain_4070lp --mode band --source runs/v15/W_dim96_conv_base/final.pt --pool weights_local/sd_pool_orig --updates 1200 --micro 8 --accum 16 --lr 4e-5 --w-h2 0.75 --w-h1 0.25 --out runs/v15/PT_B_band

# C0: balanced control MATCHED to C's training budget
python -m exp.v15.posttrain_4070lp --mode balance --source runs/v15/W_dim96_conv_base/final.pt --pool weights_local/sd_pool_orig --updates 1000 --micro 2 --accum 16 --lr 2e-5 --out runs/v15/PT_C0_balance32

# C: critic-guided cross-style edit finetune (R1 is frozen)
python -m exp.v15.posttrain_4070lp --mode guided --source runs/v15/W_dim96_conv_base/final.pt --pool weights_local/sd_pool_orig --critic runs/v6_next/probes/r1_r1.pt --updates 1000 --micro 2 --accum 16 --lr 2e-5 --k-steps 2 --tau0 0.7 --lambda-style 0.10 --lambda-content 0.20 --lambda-delta 0.02 --out runs/v15/PT_C_guided
```

If 4070 Laptop GPU memory is insufficient, retry A/B with `--micro 4
--accum 32`, and C/C0 with `--micro 1 --accum 32`. Effective batch stays
unchanged for the paired comparisons. Record the actually measured memory,
training seconds, and gradient norm rather than assuming hardware limits.

## 2. Sample the full 750-task protocol and run the independent referee

The supplied eval script uses the SAME task ordering, 20-step Euler schedule,
per-task paired eps and S10 decoder as the original V15 eval.
Unlike the original evaluator, batch size and S10 decode batch can be set low.

```powershell
python -m exp.v15.eval_4070lp --arms W_dim96_conv_base,PT_A_balance,PT_B_band,PT_C0_balance32,PT_C_guided --levels 0.4,0.7 --cfg 1 --batch 4 --decode-batch 1 --steps 20

# Separate classifier-free guidance control; no retraining required
python -m exp.v15.eval_4070lp --arms W_dim96_conv_base,PT_C_guided --levels 0.4,0.7 --cfg 2 --batch 4 --decode-batch 1 --steps 20

# Correct vs wrong target: identical input and noise, different style label
python -m exp.v15.eval_4070lp --arms W_dim96_conv_base,PT_C_guided --levels 0.7 --cfg 1 --style-shift 2 --batch 4 --decode-batch 1 --steps 20

# Full, independent DINO / CSD / CLIP / LPIPS metric on each raw path
python -m src.eval.measure --raw runs/v15/eval/raw/PT_A_balance_ptcfg1/t0_0.7 --out runs/v15/eval/measured/PT_A_balance/t0_0.7 --device cuda --no-png
```

Repeat the final `src.eval.measure` call for every arm and level. Do not
substitute V15 DINO-NCM Top-1 for historical CSD Target Top-1.

## 3. RGB H2: old checkpoint on V15, with paired wrong-style control

Unlike the latent-band post-training above, this operates in **RGB 512px
after S10 decoding**, and edits ONLY RGB H2. It does not require retraining
V15 and is an important immediately actionable control.

```powershell
python -m exp.v15.h2_apply_4070lp --raw runs/v15/eval/raw/W_dim96_conv_base_ptcfg1/t0_0.7 --ckpt runs/v11b/flow_v1_w56/ckpt_100.pt --out runs/v15/eval/raw/BASE_H2_old/t0_0.7

python -m exp.v15.h2_apply_4070lp --raw runs/v15/eval/raw/W_dim96_conv_base_ptcfg1/t0_0.7 --ckpt runs/v11b/flow_v1_w56/ckpt_100.pt --out runs/v15/eval/raw/BASE_H2_wrong/t0_0.7 --wrong-style

python -m src.eval.measure --raw runs/v15/eval/raw/BASE_H2_old/t0_0.7 --out runs/v15/eval/measured/BASE_H2_old/t0_0.7 --device cuda --no-png
python -m src.eval.measure --raw runs/v15/eval/raw/BASE_H2_wrong/t0_0.7 --out runs/v15/eval/measured/BASE_H2_wrong/t0_0.7 --device cuda --no-png
```

Correct/wrong style share the same task/image/initial RGB/epsilon and H2 sigma;
ONLY the Style-ID input changes, avoiding the old V11b control confound.

## 4. Actual RGB-H2 post-training, requiring REAL RGB data

This stage needs style-classified RGB paintings, not only VAE latent files.
The original V11b image dataset is `datasets/distinct5_512/train` (5k).
A larger 15k dataset can be used if its real RGB files are available under
style-named folders; no guarantee is made that these assets are on the 4070.

```powershell
python -m exp.v15.h2_train_4070lp --source runs/v11b/flow_v1_w56/ckpt_100.pt --image-roots datasets/distinct5_512/train --crop 256 --width 56 --micro 2 --accum 16 --updates 2000 --lr 5e-5 --out runs/v15/H2_c256

python -m exp.v15.h2_apply_4070lp --raw runs/v15/eval/raw/W_dim96_conv_base_ptcfg1/t0_0.7 --ckpt runs/v15/H2_c256/final.pt --out runs/v15/eval/raw/BASE_H2_c256/t0_0.7
python -m src.eval.measure --raw runs/v15/eval/raw/BASE_H2_c256/t0_0.7 --out runs/v15/eval/measured/BASE_H2_c256/t0_0.7 --device cuda --no-png
```

To use multiple REAL RGB roots, e.g. the original 5k and the extra WikiArt
images, pass both after `--image-roots`; preserve painter-level holdout and
check content hashes for duplicates between roots.

## 5. Interpretation and caveats

- A tests class-prior imbalance; B separately tests latent wavelet
  frequency weighting. B does NOT edit RGB H2.
- C0 is required: C uses different effective batch and LR from A/B.
  R1's own classifier success does NOT establish independent style gains.
- D is an RGB H2 correction after S10. The old V11b result improved
  DINO-S and DINO-C but also worsened LPIPS and nominal CSD Top-1.
- Always report cross600 and same150 separately, CSD margin/top1,
  DINO-S, DINO-C, LPIPS, per-style confusion and paired source bootstrap.
- Baseline V15 has no independent source-content conditioning: its editing
  at tau0=0.7 is an SDEdit-style inference procedure, not a supervised
  cross-style map.
- All scripts were syntax-checked. The V15 post-train training logic was
  separately smoke-tested with synthetic CPU stubs, not actual WEAVE weights
  on a 4070 Laptop GPU. Hardware/runtime compatibility remains to be tested.
