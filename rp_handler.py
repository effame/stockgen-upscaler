import os
import sys
import base64
import io
import requests
import boto3

import cv2
import numpy as np
import torch
import runpod
from PIL import Image

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# S3 (Cloudflare R2) — lazy init
# ---------------------------------------------------------------------------
s3_client = None


def get_s3_client():
    global s3_client
    if s3_client is None:
        access_key = os.environ.get("R2_ACCESS_KEY_ID")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        endpoint = os.environ.get("R2_ENDPOINT")
        if all([access_key, secret_key, endpoint]):
            try:
                from botocore.config import Config

                s3_client = boto3.client(
                    "s3",
                    endpoint_url=endpoint,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    config=Config(signature_version="s3v4"),
                    region_name="auto",
                )
                print("[R2] S3 Client initialized successfully.")
            except Exception as e:
                print(f"[R2] Failed to initialize S3 Client: {e}")
    return s3_client


def upload_to_r2(buffer, key, content_type="image/jpeg"):
    client = get_s3_client()
    if client is None:
        print("[R2] Upload failed: S3 Client is None (check Env Vars)")
        return None

    bucket = os.environ.get("R2_BUCKET_NAME", "stockgen-ai")
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer,
            ContentType=content_type,
            CacheControl="public, max-age=31536000",
        )
        custom_domain = os.environ.get("R2_CUSTOM_DOMAIN")
        if custom_domain:
            return f"{custom_domain.rstrip('/')}/{key}"
        endpoint = os.environ.get("R2_ENDPOINT", "").rstrip("/")
        return f"{endpoint}/{bucket}/{key}"
    except Exception as e:
        print(f"[R2] Upload Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Patch basicsr compatibility with newer PyTorch / torchvision
# ---------------------------------------------------------------------------
import torchvision.transforms.functional as F_tv
import types

shim = types.ModuleType("torchvision.transforms.functional_tensor")
shim.rgb_to_grayscale = F_tv.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = shim

from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "x2plus": {
        "url": "https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth",
        "file": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
        "max_output": 8000,
        "tier": "normal",
    },
    "x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "max_output": 8000,
        "tier": "normal",
    },
    "ultrasharp": {
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        "file": "4x-UltraSharp.pth",
        "scale": 4,
        "num_block": 23,
        "max_output": 12000,
        "tier": "ultra",
    },
    "anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
        "max_output": 8000,
        "tier": "normal",
    },
}

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/workspace/weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Weight download helpers
# ---------------------------------------------------------------------------
def download_file(url, dest, timeout=300):
    for attempt in (1, 2):
        try:
            print(f"Downloading {os.path.basename(dest)}... (attempt {attempt})")
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16 * 1024):
                    f.write(chunk)
            return
        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
            if attempt == 2:
                raise


def safe_torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except Exception:
        print(f"weights_only=True failed for {os.path.basename(path)}, falling back")
        return torch.load(path, weights_only=False, **kwargs)


# ---------------------------------------------------------------------------
# State-dict converter (classic ESRGAN keys -> RRDBNet keys)
# ---------------------------------------------------------------------------
def convert_state_dict(state_dict, num_block):
    if not any(k.startswith("model.0.") for k in state_dict.keys()):
        return state_dict

    print("Classic ESRGAN keys detected. Converting state dict...")
    new = {}
    for k, v in state_dict.items():
        if k.startswith("model.0."):
            new[k.replace("model.0.", "conv_first.")] = v
        elif k.startswith("model.1.sub."):
            parts = k.split(".")
            try:
                idx = int(parts[3])
                if idx == num_block:
                    new[k.replace(f"model.1.sub.{idx}.", "conv_body.")] = v
                else:
                    nk = k.replace("model.1.sub.", "body.")
                    nk = nk.replace(".RDB", ".rdb")
                    for i in range(1, 6):
                        nk = nk.replace(f".conv{i}.0.", f".conv{i}.")
                    new[nk] = v
            except ValueError:
                new[k] = v
        elif k.startswith("model.3."):
            new[k.replace("model.3.", "conv_up1.")] = v
        elif k.startswith("model.6."):
            new[k.replace("model.6.", "conv_up2.")] = v
        elif k.startswith("model.8."):
            new[k.replace("model.8.", "conv_hr.")] = v
        elif k.startswith("model.10."):
            new[k.replace("model.10.", "conv_last.")] = v
        else:
            new[k] = v
    return new


# ---------------------------------------------------------------------------
# Upsampler & face-enhancer loader
# ---------------------------------------------------------------------------
def load_upsampler(model_key):
    cfg = MODELS[model_key]
    dest = os.path.join(WEIGHTS_DIR, cfg["file"])

    if not os.path.exists(dest):
        download_file(cfg["url"], dest)

    try:
        sd = safe_torch_load(dest, map_location="cpu")
        raw = sd.get("params") or sd.get("params_ema") or sd
        raw = convert_state_dict(raw, cfg["num_block"])
        torch.save({"params": raw}, dest)
    except Exception as e:
        print(f"Weight prep warning for {model_key}: {e}")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=cfg["num_block"],
        num_grow_ch=32,
        scale=cfg["scale"],
    )
    return RealESRGANer(
        scale=cfg["scale"],
        model_path=dest,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=10,
        half=False,
    )


GFPGAN_PATH = os.path.join(WEIGHTS_DIR, "GFPGANv1.3.pth")

def load_face_enhancer(scale, upsampler):
    if not os.path.exists(GFPGAN_PATH):
        download_file(
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            GFPGAN_PATH,
        )
    enhancer = GFPGANer(
        model_path=GFPGAN_PATH,
        upscale=scale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )
    if torch.cuda.is_available():
        enhancer.gfpgan.half()
        if hasattr(enhancer, "arcface"):
            enhancer.arcface.half()
        orig_forward = enhancer.gfpgan.forward
        def half_forward(x):
            out = orig_forward(x.half())
            if isinstance(out, torch.Tensor):
                return out.float()
            return tuple(t.float() if isinstance(t, torch.Tensor) else t for t in out)
        enhancer.gfpgan.forward = half_forward
    return enhancer


_upsamplers = {}
_face_enhancers = {}

def get_upsampler(model_key):
    if model_key not in _upsamplers:
        print(f"Loading upsampler: {model_key}...")
        _upsamplers[model_key] = load_upsampler(model_key)
    return _upsamplers[model_key]

def get_face_enhancer(model_key, scale, upsampler):
    if model_key not in _face_enhancers:
        print(f"Loading face enhancer: {model_key}...")
        _face_enhancers[model_key] = load_face_enhancer(scale, upsampler)
    return _face_enhancers[model_key]


# ---------------------------------------------------------------------------
# EXIF auto-orient
# ---------------------------------------------------------------------------
def apply_exif_orientation(img_bgr, raw_bytes):
    """Detect EXIF orientation from raw bytes and rotate via OpenCV to avoid JPEG re-compression."""
    if raw_bytes is None:
        return img_bgr
    try:
        pil = Image.open(io.BytesIO(raw_bytes))
        orientation = pil.getexif().get(0x0112)
        rot_map = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if orientation in rot_map:
            img_bgr = cv2.rotate(img_bgr, rot_map[orientation])
    except Exception:
        pass
    return img_bgr


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------
def load_image(image_source):
    """Return BGR ndarray."""
    raw = None

    if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
        resp = requests.get(
            image_source,
            headers={"User-Agent": "StockGen-AI/1.0 (upscaler)"},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.content

    elif isinstance(image_source, str) and os.path.exists(image_source):
        with open(image_source, "rb") as f:
            raw = f.read()

    else:
        try:
            if isinstance(image_source, str) and "," in image_source:
                image_source = image_source.split(",", 1)[1]
            raw = base64.b64decode(image_source)
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from source")
    return apply_exif_orientation(img, raw)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------
def handler(job):
    job_input = job["input"]

    image_source = job_input.get("image") or job_input.get("source_image")
    model_name = job_input.get("model", "x4plus")
    face_enhance = job_input.get("face_enhance", False)

    if not image_source:
        return {"error": "Missing 'image' or 'source_image' input"}

    # Normalise model name
    model_map = {
        "RealESRGAN_x4plus": "x4plus",
        "real-esrgan": "x4plus",
        "swinir": "x4plus",
        "x4plus": "x4plus",
        "RealESRGAN_x2plus": "x2plus",
        "x2plus": "x2plus",
        "ultrasharp": "ultrasharp",
        "4x-UltraSharp": "ultrasharp",
        "anime": "anime",
    }
    model_name = model_map.get(model_name, "x4plus")

    # Load image
    try:
        runpod.serverless.progress_update(job, {"progress": 15, "statusMessage": "Downloading image..."})
        img = load_image(image_source)
    except ValueError as e:
        return {"error": str(e)}
    except requests.RequestException as e:
        return {"error": f"Failed to fetch image from URL: {e}"}

    h, w = img.shape[:2]
    cfg = MODELS[model_name]
    max_in = cfg["max_output"] // cfg["scale"]
    if w > max_in or h > max_in:
        return {
            "error": (
                f"Input image too large ({w}x{h}) for {model_name} ({cfg['tier']} tier). "
                f"Max input: {max_in}x{max_in} = {max_in * max_in // 1_000_000} MP "
                f"(output cap: {cfg['max_output']}x{cfg['max_output']} at {cfg['scale']}x scale)."
            )
        }

    # Upscale
    upsampler = get_upsampler(model_name)
    s = MODELS[model_name]["scale"]

    # Both RealESRGANer.enhance() and GFPGANer.enhance() expect BGR input
    # (they convert to RGB internally and back to BGR on output).
    # OpenCV images are BGR by default — no extra conversion needed.
    if face_enhance:
        runpod.serverless.progress_update(job, {"progress": 30, "statusMessage": "Enhancing face + upscaling..."})
        enhancer = get_face_enhancer(model_name, s, upsampler)
        with torch.inference_mode():
            _, _, output = enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
    else:
        runpod.serverless.progress_update(job, {"progress": 40, "statusMessage": "AI upscaling..."})
        with torch.inference_mode():
            output, _ = upsampler.enhance(img, outscale=s)

    # Encode to JPEG
    success, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        return {"error": "Failed to encode output image"}

    out_bytes = encoded.tobytes()

    # Direct-to-R2 upload
    r2_url = None
    r2_key = job_input.get("r2_key")
    if r2_key:
        runpod.serverless.progress_update(job, {"progress": 85, "statusMessage": "Uploading to Cloudflare R2..."})
        r2_url = upload_to_r2(out_bytes, r2_key)
        if r2_url:
            runpod.serverless.progress_update(job, {"progress": 95, "statusMessage": "Done"})

    b64 = None
    if not r2_url:
        b64 = base64.b64encode(out_bytes).decode("utf-8")

    oh, ow = output.shape[:2]

    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Job processed successfully.")
    return {
        "image": b64,
        "r2_url": r2_url,
        "image_format": "jpg",
        "model": model_name,
        "face_enhance_applied": face_enhance,
        "scale": s,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": ow, "height": oh},
    }


if __name__ == "__main__":
    print("--- Starting Serverless Worker | Version 2.1.1 ---")
    runpod.serverless.start({"handler": handler})
