import os
import sys
import base64
import tempfile
import urllib.request

import cv2
import numpy as np
import torch
import runpod

# Patch basicsr compatibility with newer PyTorch/torchvision
import torchvision.transforms.functional as F_tv
import types
shim = types.ModuleType("torchvision.transforms.functional_tensor")
shim.rgb_to_grayscale = F_tv.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = shim

from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

MODELS = {
    "x2plus": {
        "url": "https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth",
        "file": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
        "desc": "Real-ESRGAN x2plus (2x)",
    },
    "x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "desc": "Real-ESRGAN x4plus (4x, high quality)",
    },
    "ultrasharp": {
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        "file": "4x-UltraSharp.pth",
        "scale": 4,
        "num_block": 23,
        "desc": "4x-UltraSharp (4x, sharp)",
    },
    "anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
        "desc": "Real-ESRGAN anime (4x, optimized for illustrations)",
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
        
    # ─── 🩹 Patch: Auto-convert state dict for community models like 4x-UltraSharp to fix KeyError: 'params' ───
    try:
        loadnet = torch.load(dest, map_location="cpu")
        if "params" not in loadnet and "params_ema" not in loadnet:
            print(f"🩹 Auto-converting community model {model_key} state dict to fit RealESRGANer structure...")
            torch.save({"params": loadnet}, dest)
            print("✅ Successfully reformatted and saved weights wrapper!")
    except Exception as e:
        print(f"⚠️ Warning checking state dict for community model: {str(e)}")
        
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=cfg["num_block"], num_grow_ch=32, scale=cfg["scale"])
    return RealESRGANer(
        scale=cfg["scale"],
        model_path=dest,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=10,
        half=torch.cuda.is_available(),
    )


# Preload all upscale models at cold start to maximize speed
upsamplers = {key: load_upsampler(key) for key in MODELS}

# Preload GFPGAN (Face Enhancement) models mapped by upscale scales to avoid inference latency
GFPGAN_PATH = os.path.join(WEIGHTS_DIR, "GFPGANv1.3.pth")

def load_face_enhancer(scale, upsampler):
    if not os.path.exists(GFPGAN_PATH):
        print("Downloading GFPGANv1.3.pth...")
        urllib.request.urlretrieve(
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            GFPGAN_PATH
        )
    return GFPGANer(
        model_path=GFPGAN_PATH,
        upscale=scale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler
    )

# Establish dynamic preloaded face enhancers mapped to each upscale model
face_enhancers = {}
for key, upsampler in upsamplers.items():
    s = MODELS[key]["scale"]
    print(f"Preloading face enhancer (GFPGAN) for model {key} with scale {s}...")
    face_enhancers[key] = load_face_enhancer(s, upsampler)


def handler(job):
    job_input = job["input"]
    image_source = job_input.get("image") or job_input.get("source_image")
    model_name = job_input.get("model", "x4plus")
    face_enhance = job_input.get("face_enhance", False)
    bgr = job_input.get("bgr", False)

    if not image_source:
        return {"error": "Missing 'image' or 'source_image' input"}

    # Mapping model name to prevent KeyError
    # Supports variations like RealESRGAN_x4plus, real-esrgan, swinir -> x4plus
    if model_name in ["RealESRGAN_x4plus", "real-esrgan", "swinir", "x4plus"]:
        model_name = "x4plus"
    elif model_name in ["RealESRGAN_x2plus", "x2plus"]:
        model_name = "x2plus"
    elif model_name not in MODELS:
        # Fallback to x4plus if not found
        model_name = "x4plus"

    # Process input image
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, "input_img.png")

    if isinstance(image_source, str) and (image_source.startswith("http://") or image_source.startswith("https://")):
        print(f"📥 [Web Fetch] Downloading image from URL: {image_source}")
        try:
            urllib.request.urlretrieve(image_source, input_path)
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"error": f"Failed to download image from URL: {str(e)}"}
    elif isinstance(image_source, str) and os.path.exists(image_source):
        print(f"📂 [Local Load] Loading local image file: {image_source}")
        img = cv2.imread(image_source, cv2.IMREAD_COLOR)
    else:
        # Decode base64 representation
        try:
            print("🔑 [Base64 Decode] Decoding input image from base64 string...")
            if isinstance(image_source, str) and "," in image_source:
                image_source = image_source.split(",")[1]
            img_data = base64.b64decode(image_source)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"error": f"Invalid source image format. Must be URL, Path, or Base64: {str(e)}"}

    if img is None:
        return {"error": "Could not decode or load image from provided source"}

    upsampler = upsamplers.get(model_name, upsamplers["x2plus"])
    s = MODELS[model_name]["scale"]

    # Process based on face enhancement flag
    if face_enhance:
        face_enhancer = face_enhancers.get(model_name, face_enhancers["x2plus"])
        print(f"Processing with GFPGAN (Face Enhancement) using model {model_name}...")
        if bgr:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, _, output = face_enhancer.enhance(img_rgb, has_aligned=False, only_center_face=False, paste_back=True)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        else:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        print(f"Processing normal upscale with model {model_name}...")
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
        "face_enhance_applied": face_enhance,
        "scale": s,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": ow, "height": oh},
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
