<#
.SYNOPSIS
  Build & push the StockGen AI upscaler Docker image for RunPod.

  Prerequisites:
    - Docker Desktop (with WSL2) installed
    - `docker login` completed with Docker Hub PAT

  Usage:
    .\scripts\docker-build.ps1              # tag defaults to v2.0.0
    .\scripts\docker-build.ps1 -Tag v2.0.1
#>
param([string]$Tag = "v2.0.0")

$ErrorActionPreference = "Stop"
$IMAGE = "effam/stockgen-upscaler:$Tag"

Write-Host "═══ StockGen Upscaler Docker Build ═══" -ForegroundColor Cyan
Write-Host "Image : $IMAGE"
Write-Host ""

# Step 1 — Build
Write-Host "─── Step 1: Building (linux/amd64) ───" -ForegroundColor Cyan
docker build --platform=linux/amd64 -t $IMAGE -f Dockerfile .
if ($LASTEXITCODE -ne 0) { throw "Build failed" }
Write-Host "Build OK." -ForegroundColor Green

# Step 2 — Push
Write-Host "`n─── Step 2: Pushing to Docker Hub ───" -ForegroundColor Cyan
docker push $IMAGE
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
Write-Host "Push OK." -ForegroundColor Green

Write-Host "`nDone. Image is now available as $IMAGE" -ForegroundColor Green
