FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y libxcb1 libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir opencv-python-headless basicsr realesrgan gfpgan runpod

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

RUN mkdir -p /weights

COPY rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
