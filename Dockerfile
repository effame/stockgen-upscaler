FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# System deps for opencv
RUN apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0t64 && rm -rf /var/lib/apt/lists/*

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

# Pre-download face helper models (facexlib detection/parsing) during build to avoid cold start latency.
# Weights loaded from S3 network volume at runtime; just importing triggers auto-download of small helper models.
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

CMD ["python", "-u", "/rp_handler.py"]
