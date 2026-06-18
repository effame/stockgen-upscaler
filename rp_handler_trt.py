import os, sys, base64, time, json
import numpy as np
import cv2, torch, runpod
from PIL import Image

sys.path.insert(0, "/scripts")
from trt_inference_handler import StockGenInference

from utils import load_image, upload_to_r2
from models import MODELS, get_upsampler, get_face_enhancer

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
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
        path = os.path.join(MODELS_DIR, f"{model_name}.engine")
        if os.path.exists(path):
            _engines[model_name] = StockGenInference(path)
        else:
            return None
    return _engines[model_name]

@torch.inference_mode()
def handler(job):
    inp = job["input"]
    img_data = inp.get("image")
    model_name = inp.get("model", "x4plus")
    scale = inp.get("scale", 4)
    face_enhance = inp.get("face_enhance", False)
    image_format = inp.get("image_format", "jpg")
    use_half = inp.get("half", True)
    remove_bg = inp.get("remove_bg", False)
    r2_key = inp.get("r2_key")

    if not img_data:
        return {"error": "No image data"}

    runpod.serverless.progress_update(job, {"progress": 10})
    img = load_image(img_data)
    if img is None:
        return {"error": "Failed to decode"}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if remove_bg:
        try:
            session = new_session('u2net')
            result = remove(img, session=session, alpha_matting=True)
            if result is not None and result.shape[2] == 4:
                img_rgb = cv2.cvtColor(result[:,:,:3], cv2.COLOR_RGBA2RGB)
        except:
            pass

    engine = get_trt_engine(model_name)
    if engine:
        runpod.serverless.progress_update(job, {"progress": 40})
        output_img, inf_time = engine.upscale(img_rgb)
    else:
        runpod.serverless.progress_update(job, {"progress": 40})
        upsampler = get_upsampler(model_name, use_half).to(device).eval()
        img_t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device)
        if use_half: img_t = img_t.half()
        else: img_t = img_t.float()
        img_t /= 255.0
        torch.cuda.synchronize()
        output = upsampler(img_t)
        torch.cuda.synchronize()
        output_img = output.squeeze(0).permute(1,2,0).float().mul_(255).clamp_(0,255).to(torch.uint8).cpu().numpy()

    oh, ow = output_img.shape[:2]

    if face_enhance:
        runpod.serverless.progress_update(job, {"progress": 60})
        try:
            img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            fe = get_face_enhancer(use_half)
            _, _, fo = fe.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
            if fo is not None:
                output_img = cv2.cvtColor(fo, cv2.COLOR_BGR2RGB)
        except:
            pass

    out_bytes = StockGenInference.finalize_image(output_img, image_format=image_format)

    r2_url = None
    if r2_key:
        try:
            r2_url = upload_to_r2(out_bytes, r2_key, f"image/{image_format}")
        except:
            pass

    return {
        "r2_url": r2_url,
        "image": base64.b64encode(out_bytes).decode() if not r2_url else None,
        "model": model_name,
        "input_size": {"w": img_rgb.shape[1], "h": img_rgb.shape[0]},
        "output_size": {"w": ow, "h": oh},
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
