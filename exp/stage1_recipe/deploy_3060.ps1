param(
    [ValidateSet('Audit', 'Plan', 'Execute')]
    [string]$Mode = 'Audit'
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Python = (Get-Command python -ErrorAction Stop).Source
$Config = 'docs/system/optimization/training/stage1_config.json'

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:TORCH_HOME = Join-Path (Split-Path $Repo -Parent) 'eval_cache/torch'
$env:WEAVE_EVAL_CACHE_ROOT = Join-Path (Split-Path $Repo -Parent) 'eval_cache'
Set-Location $Repo
if (-not (Test-Path -LiteralPath $Python)) { throw "Python runtime missing: $Python" }

switch ($Mode) {
    'Audit' {
        & $Python -m exp.stage1_recipe.queue --config $Config --preflight
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python -m exp.stage1_recipe.evaluation.audit --config $Config --validate-training-assets --out runs/stage1_recipe/preparation/s0_cpu_audit.json
        exit $LASTEXITCODE
    }
    'Plan' {
        & $Python -m exp.stage1_recipe.queue --config $Config
        exit $LASTEXITCODE
    }
    'Execute' {
        & $Python -m exp.stage1_recipe.queue --config $Config --execute
        exit $LASTEXITCODE
    }
}
