import tensorrt as trt
import os

"""
🛠️ StockGen AI - NVIDIA TensorRT Engine Builder
หน้าที่: แปลงไฟล์ ONNX ให้กลายเป็นไฟล์ .engine (TensorRT Native) 
เพื่อความเร็วระดับ "เร็วที่สุดในโลก" บน GPU เฉพาะรุ่น
"""

def build_engine(onnx_file_path, engine_file_path, fp16=True):
    print(f"🏗️ Starting Engine Build for: {onnx_file_path}")
    
    # 1. สร้าง Logger และ Builder ของ NVIDIA
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # 2. ตั้งค่า Network และ Parser
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 3. อ่านไฟล์ ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # 4. ตั้งค่าการบิวด์ (Optimization Profile)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB Workspace
    
    # เปิดโหมด FP16 (ความลับของความเร็ว โดยที่คุณภาพแทบไม่ลด)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("⚡ FP16 Mode Enabled (Fast Processing)")

    # 5. สร้าง Engine
    print("⏳ Building serialized engine... (This may take 2-5 minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ ERROR: Failed to build the engine.")
        return False

    # 6. บันทึกลง Disk (Network Volume)
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
        
    print(f"✅ SUCCESS: Engine saved at {engine_file_path}")
    return True

if __name__ == "__main__":
    # ตัวอย่างการเรียกใช้ (จะใช้จริงบน RunPod On-Demand)
    # build_engine("models/realesr-v3.onnx", "models/realesr-v3-rtx4090.engine")
    pass
