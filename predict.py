import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch
from cog import BasePredictor, Input, Path
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = "/src/weights/RealESRGAN_x4plus.pth"


def download_weights(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)


class Predictor(BasePredictor):
    def setup(self):
        download_weights(MODEL_URL, MODEL_PATH)

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=MODEL_PATH,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0,
        )

        self.face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
        )

    def predict(
        self,
        image: Path = Input(description="Input image to upscale"),
        scale: float = Input(
            description="Upscale factor",
            default=4,
            ge=1,
            le=8,
        ),
        face_enhance: bool = Input(
            description="Enhance faces with GFPGAN",
            default=False,
        ),
        tile: int = Input(
            description="Tile size (0 = auto, 200-400 recommended for large images)",
            default=0,
            ge=0,
            le=1024,
        ),
    ) -> Path:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {image}")

        if tile > 0:
            self.upsampler.tile = tile

        if face_enhance:
            _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = self.upsampler.enhance(img, outscale=scale)

        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / "output.png"
        cv2.imwrite(str(out_path), output)

        return Path(str(out_path))
