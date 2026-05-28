import os
import sys
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch
from cog import BasePredictor, Input, Path

# Patch basicsr compatibility (torchvision >= 0.15 removes functional_tensor)
import torchvision.transforms.functional as F_tv
import types
shim = types.ModuleType("torchvision.transforms.functional_tensor")
shim.rgb_to_grayscale = F_tv.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = shim

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

MODELS = {
    "x2plus": {
        "url": "https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth",
        "file": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
        "desc": "Real-ESRGAN x2plus",
    },
    "x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "desc": "Real-ESRGAN x4plus (4x)",
    },
    "ultrasharp": {
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        "file": "4x-UltraSharp.pth",
        "scale": 4,
        "num_block": 23,
        "desc": "4x-UltraSharp (4x)",
    },
    "anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
        "desc": "Real-ESRGAN anime (4x)",
    },
}

BASE_WEIGHTS = "/src/weights"


def download_weights(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)


class Predictor(BasePredictor):
    def setup(self):
        self.upsamplers = {}
        for key, cfg in MODELS.items():
            dest = os.path.join(BASE_WEIGHTS, cfg["file"])
            download_weights(cfg["url"], dest)
            s = cfg["scale"]
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=cfg["num_block"], num_grow_ch=32, scale=s)
            self.upsamplers[key] = RealESRGANer(
                scale=s,
                model_path=dest,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=10,
                half=torch.cuda.is_available(),
                gpu_id=0,
            )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        model: str = Input(
            description="Model to use",
            default="x2plus",
            choices=list(MODELS.keys()),
        ),
        tile: int = Input(description="Tile size (0=auto)", default=0, ge=0, le=1024),
        bgr: bool = Input(description="BGR model (skip color conversion)", default=False),
    ) -> Path:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {image}")

        upsampler = self.upsamplers[model]
        if tile > 0:
            upsampler.tile_size = tile

        s = MODELS[model]["scale"]
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output, _ = upsampler.enhance(img, outscale=s)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        else:
            output, _ = upsampler.enhance(img, outscale=s)

        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / "output.jpg"
        cv2.imwrite(str(out_path), output, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return Path(str(out_path))
