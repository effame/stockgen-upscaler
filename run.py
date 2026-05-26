"""
Real-ESRGAN Image Upscaler (official RealESRGANer)
"""
import argparse
import os
import sys
import tempfile
import urllib.request
from collections import OrderedDict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch basicsr compatibility (torchvision >= 0.15 removes functional_tensor)
import torchvision.transforms.functional as F_tv
import types
shim = types.ModuleType("torchvision.transforms.functional_tensor")
shim.rgb_to_grayscale = F_tv.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = shim

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ─── Model registry ───

MODELS = OrderedDict()
MODELS["x2plus"] = {
    "url": "https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth",
    "file": "RealESRGAN_x2plus.pth",
    "scale": 2,
    "desc": "Real-ESRGAN x2plus (2x)",
}
MODELS["x4plus"] = {
    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "file": "RealESRGAN_x4plus.pth",
    "scale": 4,
    "desc": "Real-ESRGAN x4plus (4x, high quality)",
}
MODELS["ultrasharp"] = {
    "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
    "file": "4x-UltraSharp.pth",
    "scale": 4,
    "desc": "4x-UltraSharp (4x, sharp)",
}


def download_weights(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading model weights...")
        urllib.request.urlretrieve(url, dest)
        print("Done.")


def upscale(image_path, tile=0, output_path=None, model_key="x2plus",
            fmt="jpg", quality=95, bgr=False):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    cfg = MODELS[model_key]
    scale = cfg["scale"]
    print(f"Model: {cfg['desc']}")

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    model_path = os.path.join(weights_dir, cfg["file"])
    download_weights(cfg["url"], model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile or 400,
        tile_pad=10,
        pre_pad=10,
        half=device.type == "cuda",
        device=device,
    )

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    print(f"Input: {w}x{h}")

    if bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output, _ = upsampler.enhance(img, outscale=scale)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    else:
        output, _ = upsampler.enhance(img, outscale=scale)

    oh, ow = output.shape[:2]
    print(f"Output: {ow}x{oh}")

    if output_path is None:
        out_dir = tempfile.mkdtemp()
        output_path = os.path.join(out_dir, f"output.{fmt}")
    elif "." not in os.path.basename(output_path):
        output_path = f"{output_path}.{fmt}"

    params = []
    if fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    cv2.imwrite(output_path, output, params)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN Upscaler")
    parser.add_argument("--image", "-i", help="Image path or URL")
    parser.add_argument("--output", "-o", default=None, help="Output path (no extension)")
    parser.add_argument("--tile", "-t", type=int, default=0, help="Tile size (0=auto)")
    parser.add_argument("--model", "-m", default="x2plus", choices=list(MODELS.keys()))
    parser.add_argument("--format", "-f", default="jpg", choices=["png", "jpg"], help="Output format")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--bgr", action="store_true", help="BGR model (skip color conversion)")
    args = parser.parse_args()

    if args.image is None:
        print("Usage: python run.py -i <image> -o output -m x2plus -f jpg")
        print("Models:", ", ".join(f"{k} ({v['desc']})" for k, v in MODELS.items()))
        return

    image_path = args.image
    if image_path.startswith(("http://", "https://")):
        tmp = tempfile.mkdtemp()
        ext = os.path.splitext(image_path.split("/")[-1])[1] or ".png"
        local_path = os.path.join(tmp, f"input{ext}")
        print(f"Downloading image...")
        urllib.request.urlretrieve(image_path, local_path)
        image_path = local_path

    upscale(image_path, args.tile, args.output, args.model,
            args.format, args.quality, args.bgr)


if __name__ == "__main__":
    main()
