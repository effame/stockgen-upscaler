FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# System deps for opencv
RUN apt-get update && apt-get install -y libxcb1 libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir opencv-python-headless basicsr realesrgan runpod

# Patch basicsr without importing it (chicken-and-egg: import triggers the error we patch)
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

# Pre-download weights so cold start is fast
RUN mkdir -p /weights && python -c "\
import urllib.request;\
urllib.request.urlretrieve('https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth', '/weights/RealESRGAN_x2plus.pth');\
urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', '/weights/RealESRGAN_x4plus.pth')\
"

COPY rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
