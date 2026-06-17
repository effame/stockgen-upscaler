import os
import time
import base64
import numpy as np
import cv2
import torch
import runpod
from PIL import Image
from utils import load_image, encode_image, inject_dpi, upload_to_r2
from models import MODELS, get_upsampler, get_face_enhancer
from scripts.trt_inference_handler import StockGenInference

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    rembg_session = new_session("u2net")
except ImportError:
    REMBG_AVAILABLE = False

_engines = {}
MODELS_DIR = os.environ.get("MODELS_DIR", "/workspace/models")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_trt_engine(model_name):
    if model_name not in _engines:
        engine_path = os.path.join(MODELS_DIR, f"{model_name}.engine")
        if os.path.exists(engine_path):
            print(f"Loading TRT engine: {model_name}")
            _engines[model_name] = StockGenInference(engine_path)
        else:
            return None
    return _engines[model_name]


@torch.inference_mode()
def handler(job):
    job_input = job["input"]
    img_data = job_input.get("image")
    model_name = job_input.get("model", "x4plus")
    scale = job_input.get("scale", 4)
    face_enhance = job_input.get("face_enhance", False)
    image_format = job_input.get("image_format", "jpg")
    use_half = job_input.get("half", True)
    remove_bg = job_input.get("remove_bg", False)
    r2_key = job_input.get("r2_key")

    if not img_data:
        return {"error": "No image data provided"}

    runpod.serverless.progress_update(job, {"progress": 10, "statusMessage": "Loading image..."})
    try:
        img = load_image(img_data)
    except Exception as e:
        return {"error": f"Failed to decode image: {e}"}
    if img is None:
        return {"error": "Failed to decode image"}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    h, w = img_pil.height, img_pil.width

    alpha_mask = None
    if remove_bg:
        if REMBG_AVAILABLE:
            runpod.serverless.progress_update(job, {"progress": 25, "statusMessage": "Removing background..."})
            try:
                result = remove(img, session=rembg_session, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_size=10)
                if result is not None:
                    if result.shape[2] == 4:
                        alpha_mask = result[:, :, 3]
                        img_rgb = cv2.cvtColor(result[:, :, :3], cv2.COLOR_RGBA2RGB)
                        img_pil = Image.fromarray(img_rgb)
                        h, w = img_pil.height, img_pil.width
            except Exception as e:
                print(f"rembg failed, skipping: {e}")

    engine = get_trt_engine(model_name)
    if engine is not None:
        runpod.serverless.progress_update(job, {"progress": 40, "statusMessage": f"AI Upscaling ({model_name}) via TensorRT..."})
        try:
            output_img, inference_time = engine.upscale(img_rgb)
        except Exception as e:
            return {"error": f"TRT inference failed: {e}"}
    else:
        runpod.serverless.progress_update(job, {"progress": 40, "statusMessage": f"AI Upscaling ({model_name}) via PyTorch..."})
        try:
            upsampler = get_upsampler(model_name, use_half)
            upsampler.to(device)
            upsampler.eval()

            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            if use_half:
                img_t = img_t.half()
            else:
                img_t = img_t.float()
            img_t = img_t / 255.0

            torch.cuda.synchronize()
            start = time.time()
            output = upsampler(img_t)
            torch.cuda.synchronize()
            inference_time = time.time() - start

            output = output.squeeze(0).permute(1, 2, 0).float().mul_(255.0).clamp_(0, 255).to(torch.uint8).cpu().numpy()
            output_img = output
        except Exception as e:
            return {"error": f"PyTorch inference failed: {e}"}

    oh, ow = output_img.shape[:2]

    if face_enhance:
        runpod.serverless.progress_update(job, {"progress": 60, "statusMessage": "Enhancing faces..."})
        try:
            img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            face_enhancer = get_face_enhancer(use_half)
            _, _, face_output = face_enhancer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
            if face_output is not None:
                output_img = cv2.cvtColor(face_output, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Face enhancement failed, skipping: {e}")

    if alpha_mask is not None and remove_bg:
        runpod.serverless.progress_update(job, {"progress": 70, "statusMessage": "Reapplying alpha matting..."})
        try:
            alpha_resized = cv2.resize(alpha_mask, (ow, oh), interpolation=cv2.INTER_LINEAR)
            if image_format == "jpg":
                white_bg = np.full((oh, ow, 3), 255, dtype=np.uint8)
                alpha_norm = alpha_resized.astype(np.float32)[:, :, None] / 255.0
                output_img = (output_img.astype(np.float32) * alpha_norm + white_bg.astype(np.float32) * (1.0 - alpha_norm)).round().astype(np.uint8)
        except Exception as e:
            print(f"Alpha matting failed, skipping: {e}")

    runpod.serverless.progress_update(job, {"progress": 80, "statusMessage": "Finalizing image..."})
    try:
        out_bytes = StockGenInference.finalize_image(output_img, image_format=image_format)
    except Exception as e:
        return {"error": f"Image encode failed: {e}"}

    r2_url = None
    if r2_key:
        runpod.serverless.progress_update(job, {"progress": 90, "statusMessage": "Uploading to Cloudflare R2..."})
        try:
            r2_url = upload_to_r2(out_bytes, r2_key, f"image/{image_format}")
        except Exception as e:
            print(f"R2 upload failed, continuing: {e}")

    oh, ow = output_img.shape[:2]
    return {
        "r2_url": r2_url,
        "image": base64.b64encode(out_bytes).decode("utf-8") if not r2_url else None,
        "inference_time": f"{inference_time:.4f}s",
        "model": model_name,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": ow, "height": oh},
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
