@echo off
setlocal
REM Pull remote ablation / missing 750-image boards into submission_pack.
REM Requires OpenSSH scp to administrator@100.115.18.62:2222

set HOST=administrator@100.115.18.62
set PORT=2222
if not defined WEAVE_REMOTE_ROOT set "WEAVE_REMOTE_ROOT=I:/Github/Latent_Style/WEAVE"
set REMOTE=%WEAVE_REMOTE_ROOT%
set LOCAL=%~dp0..\results\cross_protocol\submission_pack\boards

if not exist "%LOCAL%\ablation" mkdir "%LOCAL%\ablation"

echo === Pull ablation_v2 boards (metrics + images) ===
for %%N in (
  a01_wo_endpoint_adain
  a02_wo_cross_attn
  a03_wo_flow
  b01_wll_0
  b02_wll_20
  b07_whh_0
  b08_whh_4
  d01_adain_0
  d02_adain_05
  d03_adain_20
  d06_steps_1
  d07_steps_32
) do (
  echo --- %%N ---
  if not exist "%LOCAL%\ablation\%%N" mkdir "%LOCAL%\ablation\%%N"
  REM eval layout
  scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_v2/%%N/eval/metrics.csv" "%LOCAL%\ablation\%%N\metrics.csv" 2>nul
  scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_v2/%%N/eval/dino_summary.json" "%LOCAL%\ablation\%%N\dino_summary.json" 2>nul
  scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_v2/%%N/eval/summary.json" "%LOCAL%\ablation\%%N\summary.json" 2>nul
  REM flat layout
  scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_v2/%%N/metrics.csv" "%LOCAL%\ablation\%%N\metrics.csv" 2>nul
  scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_v2/%%N/dino_summary.json" "%LOCAL%\ablation\%%N\dino_summary.json" 2>nul
  scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_v2/%%N/summary.json" "%LOCAL%\ablation\%%N\summary.json" 2>nul
  REM images (directory)
  scp -P %PORT% -o LogLevel=ERROR -r "%HOST%:%REMOTE%/exp/ablation_v2/%%N/eval/images" "%LOCAL%\ablation\%%N\images" 2>nul
  scp -P %PORT% -o LogLevel=ERROR -r "%HOST%:%REMOTE%/exp/ablation_v2/%%N/images" "%LOCAL%\ablation\%%N\images" 2>nul
)

echo === Pull ablation_destructive (paper-style wo_*) ===
for %%N in (wo_endpoint_adain wo_flow wo_wavelet wo_spectral_ode wo_asg) do (
  echo --- %%N ---
  if not exist "%LOCAL%\ablation\%%N" mkdir "%LOCAL%\ablation\%%N"
  scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_destructive/%%N/full_eval/epoch_0005/metrics.csv" "%LOCAL%\ablation\%%N\metrics.csv" 2>nul
  scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_destructive/%%N/full_eval/epoch_0005/summary.json" "%LOCAL%\ablation\%%N\summary.json" 2>nul
  scp -P %PORT% -o LogLevel=ERROR -r "%HOST%:%REMOTE%/exp/ablation_destructive/%%N/full_eval/epoch_0005/images" "%LOCAL%\ablation\%%N\images" 2>nul
)

echo === Pull ablation_v2 aggregate ===
scp -P %PORT% -o LogLevel=ERROR "%HOST%:%REMOTE%/exp/ablation_v2/_results.json" "%LOCAL%\ablation\_results_v2.json" 2>nul

echo === Pull missing zero_velocity / endpoint images if needed ===
if not exist "%LOCAL%\ablation\zero_velocity\images" (
  scp -P %PORT% -o LogLevel=ERROR -r "%HOST%:%REMOTE%/aaai_submission/final_v6/eval_outputs/d5_750_zero_velocity/images" "%LOCAL%\ablation\zero_velocity\images" 2>nul
)
if not exist "%LOCAL%\ablation\final_only_adain\images" (
  scp -P %PORT% -o LogLevel=ERROR -r "%HOST%:%REMOTE%/eval_outputs/d5_750_endpoint_only/images" "%LOCAL%\ablation\final_only_adain\images" 2>nul
)

echo === Done. Re-run: python scripts/build_submission_pack.py ^&^& python scripts/recompute_aux_tables_cross.py ===
endlocal
