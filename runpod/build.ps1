# Windows PowerShell 7 Build & Push script for ghcr.io
$ErrorActionPreference = "Stop"

Write-Host "=== Login to ghcr.io ===" -ForegroundColor Cyan
$token = Read-Host -Prompt "Paste your GitHub Personal Access Token (PAT) with write:packages permission" -AsSecureString
$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($token)
$plainToken = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)

$plainToken | docker login ghcr.io -u effame --password-stdin
$plainToken = $null

Write-Host "`n=== Building Docker Image ===" -ForegroundColor Cyan
docker build -t ghcr.io/effame/stockgen-upscaler:latest .

Write-Host "`n=== Pushing to ghcr.io ===" -ForegroundColor Cyan
docker push ghcr.io/effame/stockgen-upscaler:latest

Write-Host "`n=== Finished Successfully! ===" -ForegroundColor Green
