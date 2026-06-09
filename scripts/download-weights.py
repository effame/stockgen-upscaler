"""
Pre-download all model weights to a given directory.
Run this ONCE inside a RunPod pod with a mounted network volume (or locally).
Usage:
    python scripts/download-weights.py /workspace/weights
"""
import os, sys, requests

WEIGHT_URLS = {
    "RealESRGAN_x2plus.pth":
        "https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "4x-UltraSharp.pth":
        "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
    "RealESRGAN_x4plus_anime_6B.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "GFPGANv1.3.pth":
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
}

def main(weights_dir: str):
    os.makedirs(weights_dir, exist_ok=True)
    for name, url in WEIGHT_URLS.items():
        dest = os.path.join(weights_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            mb = os.path.getsize(dest) >> 20
            print(f"✓ {name} ({mb} MB) — exists, skipping")
            continue
        print(f"↓ Downloading {name}...")
        r = requests.get(url, timeout=600, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        mb = os.path.getsize(dest) >> 20
        print(f"  Done ({mb} MB)")
    print("All weights ready.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/weights")
