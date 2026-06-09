<#
.SYNOPSIS
  StockGen AI — RunPod Network Volume Setup
  Creates a network volume, pre-downloads all model weights,
  and updates the endpoint template to mount it for zero-cold-start workers.
#>

# ── Config ────────────────────────────────────────────────────────────
$DATA_CENTER    = "US-TX"   # Texas — good GPU stock for A4000/A4500
$VOLUME_SIZE_GB = 20        # 5 weights × ~350 MB each + room = 20 GB
$VOLUME_NAME    = "stockgen-upscaler-weights"
$TEMPLATE_NAME  = "stockgen-upscaler-v2-nv"  # new template with network volume
$IMAGE_TAG      = "effam/stockgen-upscaler:v2.0.0"

# ── Step 1: Create the network volume ─────────────────────────────────
Write-Host "─── Step 1: Creating network volume ───" -ForegroundColor Cyan
$vol = runpod_create-network-volume `
  -name $VOLUME_NAME `
  -size $VOLUME_SIZE_GB `
  -dataCenterId $DATA_CENTER

$VOLUME_ID = $vol.id
Write-Host "Created volume: $VOLUME_ID" -ForegroundColor Green
Write-Host "Volume will be mounted at /workspace on any pod/endpoint within $DATA_CENTER" -ForegroundColor Yellow

# ── Step 2: Create a one-time pod to download weights ────────────────
Write-Host "`n─── Step 2: Spawning weight-download pod ───" -ForegroundColor Cyan
Write-Host "This pod will download 5 model weights to the network volume then self-terminate." -ForegroundColor Gray

$POD = runpod_create-pod `
  -name "stockgen-weight-downloader" `
  -imageName $IMAGE_TAG `
  -cloudType "COMMUNITY" `
  -gpuTypeIds @("NVIDIA RTX 4000 Ada Generation") `
  -gpuCount 1 `
  -containerDiskInGb 10 `
  -volumeInGb $VOLUME_SIZE_GB `
  -volumeMountPath "/workspace" `
  -dataCenterIds @($DATA_CENTER) `
  -env @{
    "R2_ACCESS_KEY_ID"     = "<set-from-env>"
    "R2_SECRET_ACCESS_KEY" = "<set-from-env>"
    "R2_ENDPOINT"          = "<set-from-env>"
    "R2_BUCKET_NAME"       = "stockgen-ai"
    "WEIGHTS_DIR"          = "/workspace/weights"
  }
  
Write-Host "Pod created: $($POD.id)" -ForegroundColor Green
Write-Host "Wait for the pod to start, then run inside it:" -ForegroundColor Yellow
Write-Host "  python -c '" -NoNewline
Write-Host @"

import os, requests
WEIGHTS_DIR = \"/workspace/weights\"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
FILES = {
    \"RealESRGAN_x2plus.pth\":         \"https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth\",
    \"RealESRGAN_x4plus.pth\":         \"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth\",
    \"4x-UltraSharp.pth\":             \"https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth\",
    \"RealESRGAN_x4plus_anime_6B.pth\": \"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth\",
    \"GFPGANv1.3.pth\":                \"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth\",
}
for name, url in FILES.items():
    dest = os.path.join(WEIGHTS_DIR, name)
    if not os.path.exists(dest):
        print(f\"Downloading {name}...\")
        r = requests.get(url, timeout=600, stream=True)
        r.raise_for_status()
        with open(dest, \"wb\") as f:
            for c in r.iter_content(65536):
                f.write(c)
        print(f\"  Done ({os.path.getsize(dest) >> 20} MB)\")
    else:
        print(f\"{name} exists, skipping\")
print(\"All weights ready.\")
"@
Write-Host "'" -ForegroundColor Yellow

Write-Host "`nThen terminate the pod to stop billing." -ForegroundColor Yellow

# ── Step 3: Create new template with network volume ──────────────────
Write-Host "`n─── Step 3: Creating endpoint template with network volume ───" -ForegroundColor Cyan
$TEMPLATE = runpod_create-template `
  -name $TEMPLATE_NAME `
  -imageName $IMAGE_TAG `
  -isServerless $true `
  -containerDiskInGb 10 `
  -volumeInGb $VOLUME_SIZE_GB `
  -volumeMountPath "/workspace" `
  -env @{
    "WEIGHTS_DIR" = "/workspace/weights"
  }

Write-Host "Template created: $($TEMPLATE.id)" -ForegroundColor Green

# ── Step 4: Update endpoint to use new template ──────────────────────
Write-Host "`n─── Step 4: Updating endpoint to use new template ───" -ForegroundColor Cyan
Write-Host "RunPod endpoint does NOT support changing template on an existing endpoint." -ForegroundColor Yellow
Write-Host "You must create a NEW endpoint using template $TEMPLATE_NAME:" -ForegroundColor Yellow
Write-Host @"
  runpod_create-endpoint `
    -name "stockgen-upscaler-endpoint-v3" `
    -templateId "$($TEMPLATE.id)" `
    -computeType "GPU" `
    -gpuTypeIds @("NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation") `
    -gpuCount 1 `
    -workersMin 0 `
    -workersMax 10
"@ -ForegroundColor White

Write-Host "`nAfter creating the endpoint, delete the old v2 endpoint and update .env.local." -ForegroundColor Yellow
Write-Host "`nDone." -ForegroundColor Green
