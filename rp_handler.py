import os
import sys
import base64
import tempfile
import urllib.request

import cv2
import torch
import runpod

# Patch basicsr compatibility
import torchvision.transforms.functional as F_tv
import types
shim = types.ModuleType("torchvision.transforms.functional_tensor")
shim.rgb_to_grayscale = F_tv.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = shim

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

MODELS = {
    "x2plus": {
        "url": "https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth",
        "file": "RealESRGAN_x2plus.pth",
        "scale": 2,
    },
}

WEIGHTS_DIR = "/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def load_upsampler(model_key):
    cfg = MODELS[model_key]
    dest = os.path.join(WEIGHTS_DIR, cfg["file"])
    if not os.path.exists(dest):
        print(f"Downloading {cfg['file']}...")
        urllib.request.urlretrieve(cfg["url"], dest)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=cfg["scale"])
    return RealESRGANer(
        scale=cfg["scale"],
        model_path=dest,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=10,
        half=torch.cuda.is_available(),
    )


# Preload all models at cold start
upsamplers = {key: load_upsampler(key) for key in MODELS}


def handler(job):
    job_input = job["input"]
    image_url = job_input.get("image")
    model_name = job_input.get("model", "x2plus")
    bgr = job_input.get("bgr", False)

    if not image_url:
        return {"error": "Missing 'image' URL"}

    # Download input
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, "input.jpg")
    urllib.request.urlretrieve(image_url, input_path)

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not read image from URL"}

    upsampler = upsamplers.get(model_name, upsamplers["x2plus"])
    s = MODELS[model_name]["scale"]

    if bgr:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output, _ = upsampler.enhance(img_rgb, outscale=s)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    else:
        output, _ = upsampler.enhance(img, outscale=s)

    out_path = os.path.join(tmp_dir, "output.jpg")
    cv2.imwrite(out_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    h, w = img.shape[:2]
    oh, ow = output.shape[:2]

    return {
        "image": b64,
        "image_format": "jpg",
        "model": model_name,
        "scale": s,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": ow, "height": oh},
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
