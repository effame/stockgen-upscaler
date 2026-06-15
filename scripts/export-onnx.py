import os
import torch
import sys
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact

# Add current dir to path to import local models if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "stockgen-upscaler"))

MODELS = {
    "x2plus": {
        "arch": "RRDBNet",
        "file": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
    },
    "x4plus": {
        "arch": "RRDBNet",
        "file": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
    },
    "ultrasharp": {
        "arch": "RRDBNet",
        "file": "4x-UltraSharp.pth",
        "scale": 4,
        "num_block": 23,
    },
    "anime": {
        "arch": "RRDBNet",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
    },
    "v3": {
        "arch": "SRVGGNetCompact",
        "file": "realesr-general-x4v3.pth",
        "scale": 4,
        "num_feat": 64,
        "num_conv": 32,
    }
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

def export_to_onnx(model_key, weights_path, output_path):
    cfg = MODELS[model_key]
    print(f"📦 Exporting {model_key} to {output_path}...")
    
    # Load state dict
    state_dict = torch.load(weights_path, map_location="cpu")
    if "params" in state_dict:
        state_dict = state_dict["params"]
    
    # Initialize model based on architecture
    if cfg["arch"] == "RRDBNet":
        state_dict = convert_state_dict(state_dict, cfg["num_block"])
        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=cfg["num_block"], 
            num_grow_ch=32, 
            scale=cfg["scale"]
        )
    elif cfg["arch"] == "SRVGGNetCompact":
        model = SRVGGNetCompact(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=cfg["num_feat"], 
            num_conv=cfg["num_conv"], 
            upscale=cfg["scale"], 
            act_type='prelu'
        )
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Dummy input (NCHW)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"}
        }
    )
    print(f"✅ Exported {model_key} successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.weights, args.output)
