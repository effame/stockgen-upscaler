FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# System deps for opencv
RUN apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0t64 && rm -rf /var/lib/apt/lists/*

# Install python dependencies including GFPGAN for face enhancement
# Install python dependencies - pin torch to match base image so pip resolver avoids downloading duplicate CUDA bloat
RUN pip install --break-system-packages --no-cache-dir torch==2.11.0 \
 && pip install --break-system-packages --no-cache-dir opencv-python-headless basicsr realesrgan gfpgan runpod boto3 "numpy<2"

# Patch basicsr without importing it (compatibility patch for newer PyTorch/torchvision)
RUN python -c "\
import importlib.util, os;\
spec = importlib.util.find_spec('basicsr');\
path = os.path.join(os.path.dirname(spec.origin), 'data', 'degradations.py');\
content = open(path).read();\
content = content.replace(\
  'from torchvision.transforms.functional_tensor import rgb_to_grayscale',\
  'from torchvision.transforms.functional import rgb_to_grayscale');\
open(path, 'w').write(content);\
print('Patched:', path)\
"

# Pre-download core upscale and face enhancement weights so cold start is fast
RUN mkdir -p /weights && python -c "\
import urllib.request;\
print('Downloading Real-ESRGAN weights...');\
urllib.request.urlretrieve('https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth', '/weights/RealESRGAN_x2plus.pth');\
urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', '/weights/RealESRGAN_x4plus.pth');\
urllib.request.urlretrieve('https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth', '/weights/4x-UltraSharp.pth');\
urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', '/weights/RealESRGAN_x4plus_anime_6B.pth');\
print('Downloading GFPGAN weights...');\
urllib.request.urlretrieve('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', '/weights/GFPGANv1.3.pth');\
print('Pre-download core weights completed!')\
"

# Trigger gfpgan/facexlib auto-download for face helper models during build to prevent cold start latency
RUN python -c "\
import sys, os, types;\
import torchvision.transforms.functional as F_tv;\
shim = types.ModuleType('torchvision.transforms.functional_tensor');\
shim.rgb_to_grayscale = F_tv.rgb_to_grayscale;\
sys.modules['torchvision.transforms.functional_tensor'] = shim;\
from gfpgan import GFPGANer;\
from realesrgan import RealESRGANer;\
from basicsr.archs.rrdbnet_arch import RRDBNet;\
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2);\
upsampler = RealESRGANer(scale=2, model_path='/weights/RealESRGAN_x2plus.pth', model=model);\
GFPGANer(model_path='/weights/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler);\
print('Pre-download face helper detection/parsing models completed!')\
"

COPY rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
