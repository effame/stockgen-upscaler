import os
import torch
import sys
import types
from utils import download_file

# ---------------------------------------------------------------------------
# Patch basicsr compatibility with newer PyTorch / torchvision
# ---------------------------------------------------------------------------
import torchvision.transforms.functional as F_tv
shim = types.ModuleType("torchvision.transforms.functional_tensor")
shim.rgb_to_grayscale = F_tv.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = shim

from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

MODELS = {
    "x2plus": {
        "url": "https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth",
        "file": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
        "max_output": 8000,
        "tile": 0,
        "tier": "normal",
    },
    "x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "max_output": 8000,
        "tier": "normal",
    },
    "ultrasharp": {
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        "file": "4x-UltraSharp.pth",
        "scale": 4,
        "num_block": 23,
        "max_output": 12000,
        "tier": "ultra",
    },
    "anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
        "max_output": 8000,
        "tier": "normal",
    },
    "v3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "file": "realesr-general-x4v3.pth",
        "scale": 4,
        "num_block": 0,
        "max_output": 8000,
        "tile": 0,
        "tier": "general",
    },
}

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/workspace/weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def safe_torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except Exception:
        print(f"weights_only=True failed for {os.path.basename(path)}, falling back")
        return torch.load(path, weights_only=False, **kwargs)

def convert_state_dict(state_dict, num_block):
    if not any(k.startswith("model.0.") for k in state_dict.keys()):
        return state_dict
    new = {}
    for k, v in state_dict.items():
        if k.startswith("model.0."):
            new[k.replace("model.0.", "conv_first.")] = v
        elif k.startswith("model.1.sub."):
            parts = k.split(".")
            try:
                idx = int(parts[3])
                if idx == num_block:
                    new[k.replace(f"model.1.sub.{idx}.", "conv_body.")] = v
                else:
                    nk = k.replace("model.1.sub.", "body.")
                    nk = nk.replace(".RDB", ".rdb")
                    for i in range(1, 6):
                        nk = nk.replace(f".conv{i}.0.", f".conv{i}.")
                    new[nk] = v
            except ValueError:
                new[k] = v
        elif k.startswith("model.3."):
            new[k.replace("model.3.", "conv_up1.")] = v
        elif k.startswith("model.6."):
            new[k.replace("model.6.", "conv_up2.")] = v
        elif k.startswith("model.8."):
            new[k.replace("model.8.", "conv_hr.")] = v
        elif k.startswith("model.10."):
            new[k.replace("model.10.", "conv_last.")] = v
        else:
            new[k] = v
    return new

def load_upsampler(model_key, use_half=True):
    cfg = MODELS[model_key]
    dest = os.path.join(WEIGHTS_DIR, cfg["file"])
    if not os.path.exists(dest):
        download_file(cfg["url"], dest)
    try:
        loadnet = safe_torch_load(dest, map_location="cpu")
        if any(k.startswith("model.0.") for k in loadnet.keys()):
            loadnet = convert_state_dict(loadnet, cfg["num_block"])
        if 'params' not in loadnet and 'params_ema' not in loadnet:
            torch.save({'params': loadnet}, dest)
    except Exception as e:
        print(f"Weight prep warning for {model_key}: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("arch") == "srvgg":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=cfg["scale"], act_type="prelu").to(device)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=cfg["num_block"], num_grow_ch=32, scale=cfg["scale"]).to(device)
    tile_size = 0 if cfg.get("tile") == 0 else cfg.get("tile", 800)
    return RealESRGANer(scale=cfg["scale"], model_path=dest, model=model, tile=tile_size, half=use_half, device=device)

GFPGAN_PATH = os.path.join(WEIGHTS_DIR, "GFPGANv1.3.pth")

def load_face_enhancer(scale, upsampler):
    if not os.path.exists(GFPGAN_PATH):
        download_file("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth", GFPGAN_PATH)
    enhancer = GFPGANer(model_path=GFPGAN_PATH, upscale=scale, arch="clean", channel_multiplier=2, bg_upsampler=upsampler)
    if torch.cuda.is_available():
        enhancer.gfpgan = enhancer.gfpgan.float()
        if hasattr(enhancer, "arcface"):
            enhancer.arcface.float()
    return enhancer

_upsamplers = {}
_face_enhancers = {}

def get_upsampler(model_key, use_half=True):
    cache_key = f"{model_key}_half{use_half}"
    if cache_key not in _upsamplers:
        _upsamplers[cache_key] = load_upsampler(model_key, use_half)
    return _upsamplers[cache_key]

def get_face_enhancer(model_key, scale, upsampler):
    if model_key not in _face_enhancers:
        _face_enhancers[model_key] = load_face_enhancer(scale, upsampler)
    return _face_enhancers[model_key]
