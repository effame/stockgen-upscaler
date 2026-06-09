import os
import sys
import base64
import urllib.request
import requests
import boto3
from botocore.config import Config

import cv2
import numpy as np
import torch
import runpod

# Initialize S3 client for R2 (Lazy)
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
                print("✅ [R2] S3 Client initialized successfully.")
            except Exception as e:
                print(f"❌ [R2] Failed to initialize S3 Client: {str(e)}")
    return s3_client

def upload_to_r2(buffer, key, content_type="image/jpeg"):
    client = get_s3_client()
    if client is None:
        print("❌ [R2] Upload failed: S3 Client is None (check Env Vars)")
        return None
    
    bucket = os.environ.get("R2_BUCKET_NAME", "stockgen-ai")
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer,
            ContentType=content_type,
            CacheControl="public, max-age=31536000"
        )
        custom_domain = os.environ.get("R2_CUSTOM_DOMAIN")
        if custom_domain:
            return f"{custom_domain.rstrip('/')}/{key}"
        endpoint = os.environ.get('R2_ENDPOINT', '').rstrip('/')
        return f"{endpoint}/{bucket}/{key}"
    except Exception as e:
        print(f"❌ [R2] Upload Error: {str(e)}")
        return None

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

WEIGHTS_DIR = "/workspace/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def load_upsampler(model_key):
    cfg = MODELS[model_key]
    dest = os.path.join(WEIGHTS_DIR, cfg["file"])
    if not os.path.exists(dest):
        print(f"Downloading {cfg['file']}...")
        urllib.request.urlretrieve(cfg["url"], dest)
        
    # ─── 🩹 Patch: Auto-convert state dict for community models like 4x-UltraSharp to fix KeyError: 'params' and architecture mismatch ───
    try:
        loadnet = torch.load(dest, map_location="cpu")
        
        # Determine the source state dict
        if "params" in loadnet:
            state_dict = loadnet["params"]
        elif "params_ema" in loadnet:
            state_dict = loadnet["params_ema"]
        else:
            state_dict = loadnet
            
        # Detect if it's using classic ESRGAN keys (e.g. model.0.weight)
        is_classic_esrgan = any(k.startswith("model.0.") for k in state_dict.keys())
        if is_classic_esrgan:
            print(f"🩹 Classic ESRGAN keys detected in {model_key}. Converting state dict to match RRDBNet structure...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model.0."):
                    new_k = k.replace("model.0.", "conv_first.")
                elif k.startswith("model.1.sub."):
                    # Parse block index to identify the conv_body layer
                    parts = k.split(".")
                    try:
                        block_idx = int(parts[3])
                        if block_idx == cfg["num_block"]:
                            new_k = k.replace(f"model.1.sub.{block_idx}.", "conv_body.")
                        else:
                            new_k = k.replace("model.1.sub.", "body.")
                            new_k = new_k.replace(".RDB", ".rdb")
                            new_k = new_k.replace(".conv1.0.", ".conv1.")
                            new_k = new_k.replace(".conv2.0.", ".conv2.")
                            new_k = new_k.replace(".conv3.0.", ".conv3.")
                            new_k = new_k.replace(".conv4.0.", ".conv4.")
                            new_k = new_k.replace(".conv5.0.", ".conv5.")
                    except ValueError:
                        new_k = k
                elif k.startswith("model.3."):
                    new_k = k.replace("model.3.", "conv_up1.")
                elif k.startswith("model.6."):
                    new_k = k.replace("model.6.", "conv_up2.")
                elif k.startswith("model.8."):
                    new_k = k.replace("model.8.", "conv_hr.")
                elif k.startswith("model.10."):
                    new_k = k.replace("model.10.", "conv_last.")
                else:
                    new_k = k
                new_state_dict[new_k] = v
            state_dict = new_state_dict
            
        # Re-save with 'params' key as expected by RealESRGANer
        torch.save({"params": state_dict}, dest)
        print(f"✅ Successfully converted and wrapped weights for {model_key}!")
    except Exception as e:
        print(f"⚠️ Warning checking state dict for model {model_key}: {str(e)}")
        
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


# Lazy loading containers
upsamplers = {}
face_enhancers = {}

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

def get_upsampler_lazy(model_key):
    if model_key not in upsamplers:
        print(f"🚀 Lazy loading upsampler for model: {model_key}...")
        upsamplers[model_key] = load_upsampler(model_key)
    return upsamplers[model_key]

def get_face_enhancer_lazy(model_key, scale, upsampler):
    if model_key not in face_enhancers:
        print(f"🎭 Lazy loading face enhancer for model: {model_key}...")
        face_enhancers[model_key] = load_face_enhancer(scale, upsampler)
    return face_enhancers[model_key]


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
    img = None

    if isinstance(image_source, str) and (image_source.startswith("http://") or image_source.startswith("https://")):
        print(f"📥 [Web Fetch] Downloading image from URL: {image_source}")
        runpod.serverless.progress_update(job, {"progress": 15, "statusMessage": "📥 กำลังดาวน์โหลดรูปภาพต้นฉบับ..."})
        try:
            headers = {"User-Agent": "StockGen-AI/1.0 (upscaler)"}
            resp = requests.get(image_source, headers=headers, timeout=30)
            resp.raise_for_status()
            img_array = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"error": f"Failed to download image from URL: {str(e)}"}
    elif isinstance(image_source, str) and os.path.exists(image_source):
        print(f"📂 [Local Load] Loading local image file: {image_source}")
        runpod.serverless.progress_update(job, {"progress": 15, "statusMessage": "📂 กำลังโหลดรูปภาพจากระบบ..."})
        img = cv2.imread(image_source, cv2.IMREAD_COLOR)
    else:
        # Decode base64 representation
        try:
            print("🔑 [Base64 Decode] Decoding input image from base64 string...")
            runpod.serverless.progress_update(job, {"progress": 15, "statusMessage": "🔑 กำลังประมวลผลข้อมูลภาพ..."})
            if isinstance(image_source, str) and "," in image_source:
                image_source = image_source.split(",")[1]
            img_data = base64.b64decode(image_source)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"error": f"Invalid source image format. Must be URL, Path, or Base64: {str(e)}"}

    if img is None:
        return {"error": "Could not decode or load image from provided source"}

    upsampler = get_upsampler_lazy(model_name)
    s = MODELS[model_name]["scale"]

    # Process based on face enhancement flag
    if face_enhance:
        face_enhancer = get_face_enhancer_lazy(model_name, s, upsampler)
        print(f"Processing with GFPGAN (Face Enhancement) using model {model_name}...")
        runpod.serverless.progress_update(job, {"progress": 30, "statusMessage": "🎭 กำลังปรับปรุงใบหน้าและขยายขนาดภาพ..."})
        if bgr:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, _, output = face_enhancer.enhance(img_rgb, has_aligned=False, only_center_face=False, paste_back=True)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        else:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        print(f"Processing normal upscale with model {model_name}...")
        runpod.serverless.progress_update(job, {"progress": 40, "statusMessage": "🚀 AI กำลังขยายขนาดรูปภาพความละเอียดสูง..."})
        if bgr:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output, _ = upsampler.enhance(img_rgb, outscale=s)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        else:
            output, _ = upsampler.enhance(img, outscale=s)

    # Encode output directly to Base64 in-memory
    success, encoded_img = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        return {"error": "Failed to encode output image"}
    
    # Direct-to-R2 Upload Logic
    r2_url = None
    r2_key = job_input.get("r2_key") # Target path like "users/uid/batches/bid/jid.jpg"
    
    if r2_key:
        print(f"☁️ [Direct-to-R2] Uploading result to: {r2_key}")
        runpod.serverless.progress_update(job, {"progress": 85, "statusMessage": "☁️ กำลังส่งไฟล์ไปยัง Cloudflare R2..."})
        r2_url = upload_to_r2(encoded_img.tobytes(), r2_key)
        if r2_url:
            print(f"✅ [Direct-to-R2] Success: {r2_url}")
            runpod.serverless.progress_update(job, {"progress": 95, "statusMessage": "✅ อัปโหลดสำเร็จ กำลังปิดงาน..."})

    # 🚀 OPTIMIZATION: Skip Base64 if R2 upload is successful to prevent worker hanging/memory bloat
    b64 = None
    if not r2_url:
        print("📥 [Fallback] R2 failed or not requested, encoding Base64...")
        b64 = base64.b64encode(encoded_img).decode("utf-8")

    h, w = img.shape[:2]
    oh, ow = output.shape[:2]

    print("🏁 [Final] Job processed successfully. Returning results.")
    return {
        "image": b64, # None if R2 success
        "r2_url": r2_url,
        "image_format": "jpg",
        "model": model_name,
        "face_enhance_applied": face_enhance,
        "scale": s,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": ow, "height": oh},
    }


if __name__ == "__main__":
    print("--- Starting Serverless Worker | Version 1.9.4 ---")
    runpod.serverless.start({"handler": handler})
