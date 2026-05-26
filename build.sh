#!/bin/bash
set -e
echo "=== Login to ghcr.io ==="
echo -n "Paste PAT: "
read -s TOKEN
echo ""
echo "$TOKEN" | docker login ghcr.io -u effame --password-stdin
unset TOKEN

echo "=== Build ==="
docker build -t ghcr.io/effame/stockgen-upscaler:latest .

echo "=== Push ==="
docker push ghcr.io/effame/stockgen-upscaler:latest

echo "=== Done ==="
