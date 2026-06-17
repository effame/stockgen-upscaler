"""
Standalone ONNX export — no basicsr/realesrgan dependency.
Contains inline RRDBNet + SRVGGNetCompact architectures.
"""
import os, sys, torch, onnx
import torch.nn as nn
from torch.nn import functional as F
import urllib.request

# ── Helpers (from basicsr) ──────────────────────────────────────────────
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for m in module_list:
        for mm in m.modules():
            if isinstance(mm, (nn.Conv2d, nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(mm.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if mm.bias is not None:
                    mm.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

def pixel_unshuffle(x, scale):
    b, c, h, w = x.shape
    x = x.view(b, c, h // scale, scale, w // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * scale * scale, h // scale, w // scale)

# ── RRDBNet Architecture ────────────────────────────────────────────────
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2: num_in_ch = num_in_ch * 4
        elif scale == 1: num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

# ── SRVGGNetCompact Architecture ────────────────────────────────────────
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        activation = nn.PReLU(num_parameters=num_feat)
        self.body.append(activation)
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            activation = nn.PReLU(num_parameters=num_feat)
            self.body.append(activation)
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out

# ── Config ──────────────────────────────────────────────────────────────
WEIGHTS = {
    "v3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "file": "realesr-general-x4v3.pth",
        "scale": 4, "arch": "srvgg", "num_conv": 32,
    },
    "x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "scale": 4, "arch": "rrdb", "num_block": 23,
    },
}

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

def export(model_key, weights_dir="weights", output_dir="models"):
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    cfg = WEIGHTS[model_key]
    weight_path = os.path.join(weights_dir, cfg["file"])
    output_path = os.path.join(output_dir, f"{model_key}.onnx")

    # Download weight
    if not os.path.exists(weight_path):
        print(f"Downloading {cfg['file']}...")
        urllib.request.urlretrieve(cfg["url"], weight_path)

    # Load state dict
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    if "params" in state_dict:
        sd = state_dict["params"]
    elif "params_ema" in state_dict:
        sd = state_dict["params_ema"]
    else:
        sd = state_dict

    # Create model
    if cfg["arch"] == "srvgg":
        sd = convert_state_dict(sd, 0)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_conv=cfg["num_conv"], upscale=cfg["scale"], act_type="prelu")
    else:
        sd = convert_state_dict(sd, cfg["num_block"])
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=cfg["num_block"], num_grow_ch=32, scale=cfg["scale"])

    model.load_state_dict(sd, strict=True)
    model.eval()

    # Export to ONNX
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    dummy = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model, dummy, output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"}
        }
    )
    size_mb = os.path.getsize(output_path) / 1_024_000
    print(f"OK {model_key}.onnx -- {size_mb:.1f} MB")

if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "v3"
    export(model_key)
    print("Done!")
