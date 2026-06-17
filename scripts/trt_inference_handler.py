import os
import time
import numpy as np
import cv2
import torch
import piexif
import tensorrt as trt
from PIL import Image
from io import BytesIO

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_RUNTIME = trt.Runtime(TRT_LOGGER)


class StockGenInference:
    def __init__(self, engine_path):
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            serialized = f.read()
        self.engine = TRT_RUNTIME.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

    def _allocate(self, shape, dtype=np.float32):
        t = torch.empty(tuple(shape), dtype=torch.float32, device="cuda")
        return t, t.data_ptr()

    def upscale(self, img_rgb: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        input_nchw = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(torch.float32)
            .div_(255.0)
            .contiguous()
            .cuda()
        )

        self.context.set_input_shape(self.input_name, input_nchw.shape)
        output_shape = self.context.get_tensor_shape(self.output_name)

        d_output = torch.empty(tuple(output_shape), dtype=torch.float32, device="cuda")

        self.context.set_tensor_address(self.input_name, input_nchw.data_ptr())
        self.context.set_tensor_address(self.output_name, d_output.data_ptr())

        torch.cuda.synchronize()
        start = time.time()
        success = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        inference_time = time.time() - start

        if not success:
            raise RuntimeError("TensorRT inference failed")

        output_np = d_output.squeeze(0).permute(1, 2, 0).mul_(255.0).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        return output_np, inference_time

    @staticmethod
    def finalize_image(numpy_image: np.ndarray, quality=95, image_format="jpg"):
        pil_img = Image.fromarray(numpy_image)
        zeroth_ifd = {
            piexif.ImageIFD.XResolution: (300, 1),
            piexif.ImageIFD.YResolution: (300, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
        }
        exif_bytes = piexif.dump({"0th": zeroth_ifd, "Exif": {}, "GPS": {}})
        buffer = BytesIO()
        if image_format == "png":
            pil_img.save(buffer, format="PNG")
        else:
            pil_img.save(buffer, format="JPEG", exif=exif_bytes, quality=quality, subsampling=0)
        return buffer.getvalue()

    @staticmethod
    def preprocess_gfpgan(img_bgr: np.ndarray):
        from gfpgan import GFPGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        gfpgan = GFPGANer(
            model_path=None,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _, _, gfpgan_output = gfpgan.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        if gfpgan_output is not None:
            return gfpgan_output
        return img_bgr
