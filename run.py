"""
Real-ESRGAN Image Upscaler — รองรับหลายโมเดล
"""
import argparse
import os
import tempfile

import cv2
import torch
import urllib.request

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
    "remacri": {
        "url": "https://huggingface.co/Kim2091/Remacri/resolve/main/4x-Remacri.pth",
        "file": "4x-Remacri.pth",
        "desc": "4x-Remacri (ภาพทั่วไป, คมชัด)",
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
        urllib.request.urlretrieve(url, dest)
        print("Done.")


def download_image(url, dest):
    print(f"Downloading image from {url}...")
    urllib.request.urlretrieve(url, dest)
    return dest


def upscale(image_path, scale=4, tile=0, output_path=None, model_key="default"):
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    cfg = MODELS[model_key]
    print(f"Model: {cfg['desc']}")

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    model_path = os.path.join(weights_dir, cfg["file"])
    download_weights(cfg["url"], model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile or 400,
        tile_pad=10,
        pre_pad=0,
        half=(device == "cuda"),
        gpu_id=0 if device == "cuda" else None,
    )

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    print(f"Input: {w}x{h}")
    print(f"Upscaling x{scale}...")

    output, _ = upsampler.enhance(img, outscale=scale)

    oh, ow = output.shape[:2]
    print(f"Output: {ow}x{oh}")

    if output_path is None:
        out_dir = tempfile.mkdtemp()
        output_path = os.path.join(out_dir, "output.png")

    cv2.imwrite(output_path, output)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN Upscaler")
    parser.add_argument("--image", "-i", help="Image path or URL")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--scale", "-s", type=float, default=4, help="Scale factor (default: 4)")
    parser.add_argument("--tile", "-t", type=int, default=0, help="Tile size (0=auto)")
    parser.add_argument("--model", "-m", default="default", choices=list(MODELS.keys()),
                        help="Model to use (default: standard Real-ESRGAN)")
    args = parser.parse_args()

    if args.image is None:
        print("Usage: python run.py -i <image_path_or_url> -o output.png -s 4 -m <model>")
        print()
        print("Available models:")
        for k, v in MODELS.items():
            print(f"  {k}: {v['desc']}")
        print()
        print("Colab quick start:")
        print("  from google.colab import files")
        print("  uploaded = files.upload()")
        print('  !python run.py -i list(uploaded.keys())[0] -o output.png -m ultrasharp')
        return

    image_path = args.image
    if image_path.startswith(("http://", "https://")):
        tmp = tempfile.mkdtemp()
        ext = os.path.splitext(image_path.split("/")[-1])[1] or ".png"
        local_path = os.path.join(tmp, f"input{ext}")
        download_image(image_path, local_path)
        image_path = local_path

    upscale(image_path, args.scale, args.tile, args.output, args.model)


if __name__ == "__main__":
    main()
