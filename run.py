"""
Upscale images using Real-ESRGAN + PyTorch CUDA
Usage (Colab):
    !python run.py --image input.png --output output.png --scale 4
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import torch


def download_weights(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading weights from {url}...")
        if sys.platform == "win32":
            import urllib.request
            urllib.request.urlretrieve(url, dest)
        else:
            subprocess.check_call(["curl", "-SL", url, "-o", dest], close_fds=False)
        print("Download complete.")


def upscale(image_path: str, scale: float = 4, tile: int = 0, output_path: str = None) -> str:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "RealESRGAN_x4plus.pth")

    download_weights(MODEL_URL, MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=tile or 400,
        tile_pad=10,
        pre_pad=0,
        half=device == "cuda",
        gpu_id=0 if device == "cuda" else None,
    )

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    print(f"Upscaling {image_path} (scale={scale})...")
    output, _ = upsampler.enhance(img, outscale=scale)
    print(f"Output size: {output.shape[1]}x{output.shape[0]}")

    if output_path is None:
        out_dir = tempfile.mkdtemp()
        output_path = os.path.join(out_dir, "output.png")

    cv2.imwrite(output_path, output)
    print(f"Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN Image Upscaler")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    parser.add_argument("--scale", "-s", type=float, default=4, help="Upscale factor (default: 4)")
    parser.add_argument("--tile", "-t", type=int, default=0, help="Tile size (0=auto)")
    args = parser.parse_args()

    upscale(args.image, args.scale, args.tile, args.output)


if __name__ == "__main__":
    main()
