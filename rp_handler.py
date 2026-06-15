# StockGen Upscaler v2.4.1
import base64
import torch
import runpod
from utils import load_image, encode_image, inject_dpi, upload_to_r2
from models import MODELS, get_upsampler, get_face_enhancer

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

def handler(job):
    job_input = job["input"]
    image_source = job_input.get("image") or job_input.get("source_image")
    model_name = job_input.get("model", "x4plus")
    face_enhance = job_input.get("face_enhance", False)
    use_half = job_input.get("half", True)

    if not image_source:
        return {"error": "Missing 'image' or 'source_image' input"}

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

    try:
        runpod.serverless.progress_update(job, {"progress": 15, "statusMessage": "Downloading image..."})
        img = load_image(image_source)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to fetch image: {e}"}

    h, w = img.shape[:2]
    cfg = MODELS[model_name]
    max_in = cfg["max_output"] // cfg["scale"]
    if w > max_in or h > max_in:
        return {"error": f"Input image too large ({w}x{h}). Max input: {max_in}x{max_in}."}

    upsampler = get_upsampler(model_name, use_half)
    s = cfg["scale"]

    if face_enhance:
        runpod.serverless.progress_update(job, {"progress": 30, "statusMessage": "Enhancing face + upscaling..."})
        enhancer = get_face_enhancer(model_name, s, upsampler)
        with torch.inference_mode():
            _, _, output = enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        runpod.serverless.progress_update(job, {"progress": 40, "statusMessage": "AI upscaling..."})
        with torch.inference_mode():
            output, _ = upsampler.enhance(img, outscale=s)

    image_format = job_input.get("image_format", "jpg")
    out_bytes = encode_image(output, image_format)

    if out_bytes is None:
        return {"error": "Failed to encode output image"}

    if image_format == "jpg":
        out_bytes = inject_dpi(out_bytes, dpi=300)

    r2_url = None
    r2_key = job_input.get("r2_key")
    if r2_key:
        runpod.serverless.progress_update(job, {"progress": 85, "statusMessage": "Uploading to Cloudflare R2..."})
        content_type = "image/png" if image_format == "png" else "image/jpeg"
        r2_url = upload_to_r2(out_bytes, r2_key, content_type)

    b64 = None
    if not r2_url:
        b64 = base64.b64encode(out_bytes).decode("utf-8")

    oh, ow = output.shape[:2]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "image": b64,
        "r2_url": r2_url,
        "image_format": image_format,
        "model": model_name,
        "face_enhance_applied": face_enhance,
        "scale": s,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": ow, "height": oh},
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
