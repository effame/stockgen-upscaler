import os, base64, numpy as np
import cv2, torch, runpod
from PIL import Image
from io import BytesIO
from utils import load_image, upload_to_r2, inject_dpi
from models import MODELS, get_upsampler, get_face_enhancer

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

    img_bgr = img
    alpha_mask = None

    if remove_bg:
        try:
            from rembg import remove, new_session
            result = remove(img, session=new_session('u2net'), alpha_matting=True)
            if result is not None and result.shape[2] == 4:
                alpha_mask = result[:,:,3]
                img_bgr = result[:,:,:3]
        except:
            pass

    runpod.serverless.progress_update(job, {"progress": 40, "statusMessage": f"AI Upscaling ({model_name})..."})
    upsampler = get_upsampler(model_name, use_half)
    output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
    oh, ow = output_bgr.shape[:2]

    if face_enhance:
        runpod.serverless.progress_update(job, {"progress": 60, "statusMessage": "Enhancing faces..."})
        try:
            enhancer = get_face_enhancer("gfpgan", scale, None)
            _, _, fo = enhancer.enhance(output_bgr, has_aligned=False, only_center_face=False, paste_back=True)
            if fo is not None:
                output_bgr = fo
        except:
            pass

    if alpha_mask is not None and remove_bg:
        try:
            alpha_up = cv2.resize(alpha_mask, (ow, oh), interpolation=cv2.INTER_CUBIC)
            alpha_up[alpha_up < 10] = 0
            out_rgba = np.zeros((oh, ow, 4), dtype=np.uint8)
            out_rgba[:,:,:3] = output_bgr[:,:,::-1]
            out_rgba[:,:,3] = alpha_up
            pil_img = Image.fromarray(out_rgba, "RGBA")
            buf = BytesIO()
            if image_format == "png":
                pil_img.save(buf, "PNG")
            else:
                white = Image.new("RGB", (ow, oh), (255,255,255))
                white.paste(pil_img, mask=pil_img.split()[3])
                pil_img = white
                buf = BytesIO()
                pil_img.save(buf, "JPEG", quality=95)
            out_bytes = buf.getvalue()
        except:
            out_bytes = None
    else:
        out_bytes = inject_dpi(cv2.imencode(f".{image_format}", output_bgr)[1].tobytes(), image_format)

    r2_url = None
    if r2_key and out_bytes:
        try:
            r2_url = upload_to_r2(out_bytes, r2_key, f"image/{image_format}")
        except:
            pass

    return {
        "r2_url": r2_url,
        "image": base64.b64encode(out_bytes).decode() if out_bytes and not r2_url else None,
        "model": model_name,
        "output_size": {"w": ow, "h": oh},
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
