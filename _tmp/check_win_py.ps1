$ErrorActionPreference = 'Continue'
$cands = @(
    'C:\Program Files\Python312\python.exe',
    'C:\Python312\python.exe',
    'python'
)
foreach ($c in $cands) {
    try {
        $v = & $c -c "import sys; print(sys.version.split()[0])" 2>&1
        Write-Output "PY_OK $c -> $v"
        $h = & $c -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>&1
        Write-Output "   hf_hub: $h"
    } catch {
        Write-Output "PY_FAIL $c : $($_.Exception.Message)"
    }
}
Write-Output "--- disk free on I: ---"
Get-PSDrive I | Select-Object Used, Free | Format-List
