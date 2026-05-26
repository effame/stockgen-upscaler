"""
Real-ESRGAN Image Upscaler
"""
import argparse
import os
import tempfile
from collections import OrderedDict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request


# ─── RRDBNet architecture ───

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.ModuleList([RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        for block in self.body:
            feat = block(feat)
        feat = self.conv_body(feat)
        feat = self.lrelu(F.interpolate(feat, scale_factor=4, mode="nearest"))
        feat = self.lrelu(self.conv_up1(feat))
        feat = self.lrelu(F.interpolate(feat, scale_factor=2, mode="nearest"))
        feat = self.lrelu(self.conv_up2(feat))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ─── Model registry ───

MODELS = {
    "default": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "desc": "Real-ESRGAN x4plus (สมดุลทั่วไป)",
    },
    "ultrasharp": {
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        "file": "4x-UltraSharp.pth",
        "desc": "4x-UltraSharp (คมชัด, ภาพคน, ผลิตภัณฑ์)",
    },
}


def download_weights(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading model weights...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            if os.path.exists(dest):
                os.remove(dest)
            raise
        print("Done.")


def _remap_ultrasharp_key(k):
    inner = k[6:]
    if inner.startswith("0."):
        return "conv_first." + inner[2:]
    if inner.startswith("1.sub."):
        sub_inner = inner[6:]
        dot = sub_inner.find(".")
        block_idx = int(sub_inner[:dot])
        rest = sub_inner[dot + 1:]
        if block_idx < 23:
            rest = rest.replace("RDB", "rdb").replace(".0", "")
            return f"body.{block_idx}.{rest}"
        if block_idx == 23:
            return f"conv_body.{rest}"
        return None
    if inner.startswith("3."):
        return "conv_up1." + inner[2:]
    if inner.startswith("6."):
        return "conv_up2." + inner[2:]
    if inner.startswith("8."):
        return "conv_hr." + inner[2:]
    if inner.startswith("10."):
        return "conv_last." + inner[3:]
    return None


def load_model(model_path, device):
    model = RRDBNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    for key in ["params", "params_ema", "state_dict"]:
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        state = checkpoint

    new_state = OrderedDict()
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            mapped = _remap_ultrasharp_key(k)
            if mapped is not None:
                new_state[mapped] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def inference_tiled(model, img_bgr, tile_size=400, tile_pad=10):
    import numpy as np
    h, w = img_bgr.shape[:2]
    img = img_bgr[:, :, ::-1].copy()
    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    device = next(model.parameters()).device
    img_t = img_t.to(device)

    # Model is always 4x native
    scale = 4
    oh, ow = h * scale, w * scale
    out = torch.zeros(1, 3, oh, ow, device=device)
    ts = min(tile_size, h // 2 + 1) * scale

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0 = max(0, y - tile_pad)
            y1 = min(h, y + tile_size + tile_pad)
            x0 = max(0, x - tile_pad)
            x1 = min(w, x + tile_size + tile_pad)

            patch = img_t[:, :, y0:y1, x0:x1]
            pred = model(patch)

            dy = min(tile_size, h - y) * scale
            dx = min(tile_size, w - x) * scale
            out[:, :, y * scale:y * scale + dy, x * scale:x * scale + dx] = pred[:, :, :dy, :dx]

    result = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
    result = result[:, :, ::-1].astype(np.uint8)
    return result


def upscale(image_path, scale=2, tile=0, output_path=None, model_key="default",
            fmt="png", quality=92):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    cfg = MODELS[model_key]
    print(f"Model: {cfg['desc']}")

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    model_path = os.path.join(weights_dir, cfg["file"])
    download_weights(cfg["url"], model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = load_model(model_path, device)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    print(f"Input: {w}x{h}")

    # Model is always 4x native. Resize input to match desired output scale.
    if scale != 4:
        factor = scale / 4.0
        new_w, new_h = int(w * factor + 0.5), int(h * factor + 0.5)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = new_h, new_w
        print(f"Resized input → {w}x{h} for x{scale} output")

    print(f"Upscaling 4x...")
    t = tile or 400
    output = inference_tiled(model, img, tile_size=t)

    oh, ow = output.shape[:2]
    print(f"Output: {ow}x{oh}")

    if output_path is None:
        out_dir = tempfile.mkdtemp()
        output_path = os.path.join(out_dir, f"output.{fmt}")

    params = []
    if fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    cv2.imwrite(output_path, output, params)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN Upscaler")
    parser.add_argument("--image", "-i", help="Image path or URL")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--scale", "-s", type=float, default=2, help="Output scale (1-4)")
    parser.add_argument("--tile", "-t", type=int, default=0, help="Tile size (0=auto)")
    parser.add_argument("--model", "-m", default="default", choices=list(MODELS.keys()))
    parser.add_argument("--format", "-f", default="png", choices=["png", "jpg"], help="Output format")
    parser.add_argument("--quality", "-q", type=int, default=92, help="JPEG quality (1-100)")
    args = parser.parse_args()

    if args.image is None:
        print("Usage: python run.py -i <image> -o output -s 2 -m default")
        print("Models:", ", ".join(f"{k} ({v['desc']})" for k, v in MODELS.items()))
        return

    image_path = args.image
    if image_path.startswith(("http://", "https://")):
        tmp = tempfile.mkdtemp()
        ext = os.path.splitext(image_path.split("/")[-1])[1] or ".png"
        local_path = os.path.join(tmp, f"input{ext}")
        print(f"Downloading image...")
        urllib.request.urlretrieve(image_path, local_path)
        image_path = local_path

    upscale(image_path, args.scale, args.tile, args.output, args.model,
            args.format, args.quality)


if __name__ == "__main__":
    main()
