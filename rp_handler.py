# StockGen Upscaler v2.4.3
import cv2
import torch
import runpod
import gc
import numpy as np
from PIL import Image
from utils import load_image, encode_image, inject_dpi, upload_to_r2
from models import MODELS, get_upsampler, get_face_enhancer

# Optional: Import rembg for background removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    # Pre-warm session to avoid delay on first request
    rembg_session = new_session('u2net')
except ImportError:
    REMBG_AVAILABLE = False

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

    # 1. Load image
    runpod.serverless.progress_update(job, {"progress": 10, "statusMessage": "Loading image..."})
    try:
        img = load_image(img_data)
    except Exception as e:
        return {"error": f"Failed to decode image: {e}"}
        
    if img is None:
        return {"error": "Failed to decode image"}

    # load_image returns BGR numpy; convert to RGB PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    h, w = img_pil.height, img_pil.width

    # 2. Background Removal (Smart Pipeline: Run on original small image)
    alpha_mask = None
    if remove_bg:
        if REMBG_AVAILABLE:
            runpod.serverless.progress_update(job, {"progress": 25, "statusMessage": "Removing background (Alpha Matting)..."})
            # Use Alpha Matting for high-quality edges on the small image
            pil_nobg = remove(
                img_pil, 
                session=rembg_session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10
            )
            # If removing background, we must output PNG for transparency
            image_format = "png"
            # Extract the Alpha mask as numpy array
            alpha_mask = np.array(pil_nobg)[:, :, 3]
        else:
            print("WARNING: Background removal requested but rembg not installed")

    # 3. Upscaling (Run on original RGB image)
    output_bgr = np.array(img_pil)[:, :, ::-1] # Convert RGB PIL to BGR Numpy
    
    # Anti-Halo technique: Pre-multiply alpha (mask out background to black before upscaling)
    if remove_bg and alpha_mask is not None:
        mask_3ch = np.stack([alpha_mask]*3, axis=2) / 255.0
        output_bgr = (output_bgr * mask_3ch).astype(np.uint8)
        
    s = 1
    if model_name != "none":
        upsampler = get_upsampler(model_name, use_half)
        s = scale

        if face_enhance:
            runpod.serverless.progress_update(job, {"progress": 40, "statusMessage": "Enhancing face + upscaling..."})
            enhancer = get_face_enhancer("gfpgan", s, upsampler)
            _, _, output_bgr = enhancer.enhance(output_bgr, has_aligned=False, only_center_face=False, paste_back=True)
            if isinstance(output_bgr, list): # GFPGAN returns a list of faces sometimes, but enhance usually returns tuple
                pass
        else:
            runpod.serverless.progress_update(job, {"progress": 50, "statusMessage": "AI upscaling..."})
            output_bgr, _ = upsampler.enhance(output_bgr, outscale=s)
    # Free PyTorch VRAM from upscaling intermediates
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
            
    # 4. Handle Alpha Upscaling & Merge (Smart Pipeline)
    final_output = output_bgr
    if remove_bg and alpha_mask is not None:
        runpod.serverless.progress_update(job, {"progress": 70, "statusMessage": "Merging high-res transparency mask..."})
        oh, ow = output_bgr.shape[:2]
        # Upscale alpha channel using INTER_CUBIC (softer edges, less harsh than LANCZOS4 for masking)
        alpha_up = cv2.resize(alpha_mask, (ow, oh), interpolation=cv2.INTER_CUBIC)
        # Defringing: Hard cut-off for very low opacity pixels that might cause faint halos
        alpha_up[alpha_up < 10] = 0
        
        # Merge BGR output with Alpha into RGBA
        output_rgba = np.zeros((oh, ow, 4), dtype=np.uint8)
        output_rgba[:, :, :3] = output_bgr[:, :, ::-1] # BGR to RGB
        output_rgba[:, :, 3] = alpha_up
        # Convert back to PIL
        final_output_pil = Image.fromarray(output_rgba, 'RGBA')
    else:
        # Convert BGR back to RGB PIL
        final_output_pil = Image.fromarray(output_bgr[:, :, ::-1])

    ow, oh = final_output_pil.size

    # 5. Output processing (DPI Injection)
    runpod.serverless.progress_update(job, {"progress": 80, "statusMessage": "Injecting 300 DPI metadata..."})
    # Encode PIL Image to JPEG bytes first, then inject DPI
    import io
    buf = io.BytesIO()
    save_format = "PNG" if image_format == "png" else "JPEG"
    final_output_pil.save(buf, format=save_format, quality=95)
    encoded_bytes = buf.getvalue()
    final_output_bytes = inject_dpi(encoded_bytes, dpi=300)

    if final_output_bytes is None:
         final_output_bytes = encoded_bytes

    # 6. Upload to R2 (required — no base64 fallback to avoid 400 Bad Request)
    if not r2_key:
        return {"error": "Missing r2_key"}
    runpod.serverless.progress_update(job, {"progress": 90, "statusMessage": "Uploading to Cloudflare R2..."})
    r2_url = None
    try:
        r2_url = upload_to_r2(final_output_bytes, r2_key, image_format)
    except Exception as e:
        print(f"R2 Upload Error: {e}")

    if not r2_url:
        return {"error": "R2 upload failed"}

    # Clean up VRAM before next job
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "r2Url": r2_url,
        "model": model_name,
        "remove_bg": remove_bg,
        "face_enhance": face_enhance,
        "scale": scale,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": ow, "height": oh},
        "output_file_size": len(final_output_bytes),
        "output_dpi": 300,
        "output_color_space": "sRGB",
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
