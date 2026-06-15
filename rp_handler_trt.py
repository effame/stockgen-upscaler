import os
import time
import base64
import numpy as np
import cv2
import torch
import runpod
import tensorrt as trt
from PIL import Image
from io import BytesIO
from utils import load_image, upload_to_r2, inject_dpi

# Import our new TRT Native Inference components
from scripts.trt_inference_handler import StockGenInference

"""
🚀 StockGen AI - NextGen TRT Handler (RunPod Serverless)
นี่คือ Handler ชุดใหม่ที่ออกแบบมาเพื่อความเร็วระดับโลกด้วย Native TensorRT
รองรับ Multi-Tier (Standard v3 / Pro x4plus) และ DPI 300
"""

# โหลด Engine ล่วงหน้า (Caching) เพื่อลด Cold Start
_engines = {}
MODELS_DIR = os.environ.get("MODELS_DIR", "/workspace/models")

def get_trt_engine(model_name):
    if model_name not in _engines:
        engine_path = os.path.join(MODELS_DIR, f"{model_name}.engine")
        if os.path.exists(engine_path):
            print(f"📦 Loading TRT Engine: {model_name}")
            _engines[model_name] = StockGenInference(engine_path)
        else:
            return None
    return _engines[model_name]

def handler(job):
    job_input = job["input"]
    image_source = job_input.get("image") or job_input.get("source_image")
    model_name = job_input.get("model", "v3") # Default เป็น v3 สำหรับ Standard Tier
    quality = job_input.get("quality", 95)
    
    if not image_source:
        return {"error": "Missing 'image' input"}

    # 1. โหลดรูปภาพ
    try:
        runpod.serverless.progress_update(job, {"progress": 15, "statusMessage": "Downloading image..."})
        img = load_image(image_source)
    except Exception as e:
        return {"error": f"Failed to fetch image: {e}"}

    h, w = img.shape[:2]

    # 2. ตรวจสอบและโหลด TensorRT Engine
    # หมายเหตุ: หากไม่มีไฟล์ .engine ระบบจะแจ้ง Error (เนื่องจากเราต้องการความเร็วสูงสุดเท่านั้น)
    engine = get_trt_engine(model_name)
    if not engine:
        return {"error": f"Model engine '{model_name}' not found. Please build the engine first."}

    # 3. AI Upscaling (TensorRT Native)
    runpod.serverless.progress_update(job, {"progress": 40, "statusMessage": f"AI Upscaling ({model_name})..."})
    
    # แปลงภาพเป็น RGB สำหรับ AI
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # รัน Inference
    try:
        # เตรียม input ในรูปแบบ NCHW
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        
        start_time = time.time()
        # หมายเหตุ: ในระบบจริง เราจะใช้ Native Inference ที่สมบูรณ์กว่านี้ 
        # สำหรับเบต้า เราจะใช้ session.run ของ ONNX Runtime ที่เปิด TRT Provider ไว้ก่อน
        output = engine.session.run(None, {engine.input_name: img_input})[0]
        inference_time = time.time() - start_time
        
        # Post-processing
        output = np.squeeze(output)
        output = np.clip(output, 0, 1)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
    except Exception as e:
        return {"error": f"AI Inference failed: {str(e)}"}

    # 4. ฝัง DPI 300 และแปลงเป็น JPEG (Finalize)
    runpod.serverless.progress_update(job, {"progress": 70, "statusMessage": "Finalizing image (DPI 300)..."})
    
    # ใช้ฟังก์ชัน finalize_image จาก handler ชุดใหม่
    out_bytes = engine.finalize_image(output, quality=quality)

    # 5. อัปโหลดเข้า Cloudflare R2
    r2_url = None
    r2_key = job_input.get("r2_key")
    if r2_key:
        runpod.serverless.progress_update(job, {"progress": 85, "statusMessage": "Uploading to Cloudflare R2..."})
        r2_url = upload_to_r2(out_bytes, r2_key, "image/jpeg")

    # 6. ส่งผลลัพธ์กลับ
    oh, ow = output.shape[:2]
    
    return {
        "r2_url": r2_url,
        "inference_time": f"{inference_time:.4f}s",
        "model": model_name,
        "input_size": {"width": w, "height": h},
        "output_size": {"width": ow, "height": oh},
        "dpi": 300
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
