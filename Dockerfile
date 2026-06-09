FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# System deps for opencv
RUN apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0t64 && rm -rf /var/lib/apt/lists/*

# Pin all pip deps for reproducible builds.
# PyTorch already ships in the base image — skip re-installing it.
RUN pip install --break-system-packages --no-cache-dir \
  opencv-python-headless==4.10.0.84 \
  basicsr==1.4.2 \
  realesrgan==0.3.0 \
  gfpgan==1.3.8 \
  runpod==1.7.1 \
  boto3==1.34.106 \
  "numpy<2" \
  "Pillow>=10"

# Patch basicsr: replace torchvision.functional_tensor import (removed in newer torchvision)
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

# Pre-download face helper models (facexlib detection/parsing) during build.
RUN python -c "\
import sys, types;\
import torchvision.transforms.functional as F_tv;\
shim = types.ModuleType('torchvision.transforms.functional_tensor');\
shim.rgb_to_grayscale = F_tv.rgb_to_grayscale;\
sys.modules['torchvision.transforms.functional_tensor'] = shim;\
from gfpgan import GFPGANer;\
print('Pre-download face helper models completed!')\
"

COPY rp_handler.py /rp_handler.py

# WEIGHTS_DIR is typically /workspace/weights when a network volume is mounted at /workspace.
# Without a network volume the worker falls back to downloading weights from R2 at cold start.
ENV WEIGHTS_DIR=/workspace/weights

# Build-time marker: trigger re-build on push
CMD ["python", "-u", "/rp_handler.py"]