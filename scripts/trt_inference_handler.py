import os
import time
import numpy as np
import cv2
import torch
import piexif
import onnxruntime as ort
from PIL import Image
from io import BytesIO

"""
🚀 StockGen AI - NextGen Inference Handler (TensorRT + DPI 300)
ระบบประมวลผลรุ่นใหม่ที่รองรับการแบ่ง Tier โมเดล และฝัง DPI 300 อัตโนมัติ
"""

class StockGenInference:
    def __init__(self, model_path, device_id=0):
        print(f"⚙️ Initializing TensorRT Engine from: {model_path}")
        
        # 1. ตั้งค่า TensorRT Execution Provider
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': device_id,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache'
            }),
            'CUDAExecutionProvider'
        ]
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def upscale(self, input_image_path):
        # --- [Step 1: Pre-processing] ---
        img = cv2.imread(input_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # เตรียมภาพเข้าระบบ AI (NCHW Format)
        img_input = img.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)

        # --- [Step 2: AI Inference (TensorRT)] ---
        start_time = time.time()
        output = self.session.run(None, {self.input_name: img_input})[0]
        inference_time = time.time() - start_time
        
        # --- [Step 3: Post-processing] ---
        output = np.squeeze(output)
        output = np.clip(output, 0, 1)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        return output, inference_time

    def finalize_image(self, numpy_image, quality=95):
        """
        แปลงเป็น JPEG และฝัง DPI 300 (ไม่มีการฝัง Metadata อื่นๆ ตามที่คุยกัน)
        """
        pil_img = Image.fromarray(numpy_image)
        
        # สร้าง EXIF สำหรับ DPI 300
        zeroth_ifd = {
            piexif.ImageIFD.XResolution: (300, 1),
            piexif.ImageIFD.YResolution: (300, 1),
            piexif.ImageIFD.ResolutionUnit: 2, # Inches
        }
        
        exif_bytes = piexif.dump({"0th": zeroth_ifd, "Exif": {}, "GPS": {}})
        
        # บันทึกเป็น Bytes (เพื่อเตรียมอัปโหลด R2)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", exif=exif_bytes, quality=quality, subsampling=0)
        
        return buffer.getvalue()

# --- [ตัวอย่างการใช้งาน] ---
if __name__ == "__main__":
    # จำลองการเลือกโมเดลตาม Tier
    # model_path = "models/realesr-v3.onnx" # สำหรับ Standard
    # model_path = "models/realesr-x4plus.onnx" # สำหรับ Pro
    
    print("💡 ระบบพร้อมใช้งาน กรุณาระบุพาธโมเดล ONNX และรูปภาพเพื่อทดสอบ")
