"""
Real-ESRGAN Image Upscaler — รองรับหลายโมเดล
"""
import argparse
import os
import tempfile

import cv2
import torch
import urllib.request
from collections import OrderedDict

MODELS = {
    "default": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "desc": "Real-ESRGAN x4plus (สมดุลทั่วไป)",
    },
    "ultrasharp": {
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        "file": "4x-UltraSharp.pth",
        "desc": "4x-UltraSharp (คมชัด, ภาพคน, ผลิตภัณฑ์)",
    },
    "nmkd": {
        "url": "https://icedrive.net/1/43GNBihZyi",
        "file": "4x_NMKD-Superscale.pth",
        "desc": "NMKD Superscale (ภาพ realistic, noise)",
    },
}


def download_weights(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading model weights...")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.HTTPError as e:
            print(f"Download failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            raise
        print("Done.")


def load_model(model_path, device):
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v

    model.load_state_dict(new_state)
    model = model.to(device)
    model.eval()
    return model


def upscale(image_path, scale=4, tile=0, output_path=None, model_key="default"):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    cfg = MODELS[model_key]
    print(f"Model: {cfg['desc']}")

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    model_path = os.path.join(weights_dir, cfg["file"])
    download_weights(cfg["url"], model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = load_model(model_path, device)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    print(f"Input: {w}x{h}")
    print(f"Upscaling x{scale}...")

    tile_size = tile or 400

    output = inference_tiled(model, img, scale, tile_size, tile_pad=10, device=device)

    oh, ow = output.shape[:2]
    print(f"Output: {ow}x{oh}")

    if output_path is None:
        out_dir = tempfile.mkdtemp()
        output_path = os.path.join(out_dir, "output.png")

    cv2.imwrite(output_path, output)
    print(f"Saved: {output_path}")
    return output_path


def inference_tiled(model, img, scale, tile_size, tile_pad=10, device="cuda"):
    import math
    import torch.nn.functional as F
    import numpy as np

    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    h, w = img_t.shape[2:]

    if h * w < tile_size * tile_size:
        with torch.no_grad():
            out = model(img_t)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
        return out.astype(np.uint8)[:, :, ::-1]

    oh, ow = h * scale, w * scale
    output = torch.zeros(1, 3, oh, ow, device=device)

    tile_h = min(tile_size, h)
    tile_w = min(tile_size, w)

    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
            y0 = max(0, y - tile_pad)
            y1 = min(h, y + tile_h + tile_pad)
            x0 = max(0, x - tile_pad)
            x1 = min(w, x + tile_w + tile_pad)

            tile_in = img_t[:, :, y0:y1, x0:x1]
            with torch.no_grad():
                tile_out = model(tile_in)

            out_h = tile_out.shape[2]
            out_w = tile_out.shape[3]

            sy0 = (y - y0) * scale
            sy1 = sy0 + min(tile_h, h - y) * scale
            sx0 = (x - x0) * scale
            sx1 = sx0 + min(tile_w, w - x) * scale

            output[:, :, y * scale:y * scale + min(tile_h, h - y) * scale,
                   x * scale:x * scale + min(tile_w, w - x) * scale] = tile_out[:, :, sy0:sy1, sx0:sx1]

    out = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
    return out.astype(np.uint8)[:, :, ::-1]


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN Upscaler")
    parser.add_argument("--image", "-i", help="Image path or URL")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--scale", "-s", type=float, default=4, help="Scale factor (default: 4)")
    parser.add_argument("--tile", "-t", type=int, default=0, help="Tile size (0=auto)")
    parser.add_argument("--model", "-m", default="default", choices=list(MODELS.keys()),
                        help="Model to use")
    args = parser.parse_args()

    if args.image is None:
        print("Usage: python run.py -i <image> -o output.png -s 4 -m <model>")
        print()
        print("Available models:")
        for k, v in MODELS.items():
            print(f"  {k}: {v['desc']}")
        return

    image_path = args.image
    if image_path.startswith(("http://", "https://")):
        tmp = tempfile.mkdtemp()
        ext = os.path.splitext(image_path.split("/")[-1])[1] or ".png"
        local_path = os.path.join(tmp, f"input{ext}")
        print(f"Downloading image from {image_path}...")
        urllib.request.urlretrieve(image_path, local_path)
        image_path = local_path

    upscale(image_path, args.scale, args.tile, args.output, args.model)


if __name__ == "__main__":
    main()
