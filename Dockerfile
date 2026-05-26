FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir opencv-python-headless basicsr realesrgan runpod

# Patch basicsr for torchvision >= 0.15
RUN python -c "
import basicsr, os
path = os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py')
content = open(path).read()
content = content.replace(
    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
    'from torchvision.transforms.functional import rgb_to_grayscale'
)
open(path, 'w').write(content)
print('Patched:', path)
"

# Pre-download weights so cold start is fast
RUN mkdir -p /weights && \
    python -c "
import urllib.request
for url in [
    'https://huggingface.co/nateraw/real-esrgan/resolve/main/RealESRGAN_x2plus.pth',
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
]:
    name = url.split('/')[-1]
    print(f'Downloading {name}...')
    urllib.request.urlretrieve(url, f'/weights/{name}')
"

COPY rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
