$ErrorActionPreference = 'Continue'
$url = 'https://huggingface.co/mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers/resolve/main/config.json'
try {
    $r = Invoke-WebRequest -Uri $url -Method Head -TimeoutSec 20 -UseBasicParsing
    Write-Output ("WINDOWS_NET_OK status=" + $r.StatusCode)
} catch {
    Write-Output ("WINDOWS_NET_FAIL " + $_.Exception.Message)
}
