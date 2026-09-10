# 🚀 StockGen AI - RunPod Serverless GPU Upscaler Worker

[![GitHub Package](https://img.shields.io/badge/Container-GHCR.io-blue?logo=docker)](https://github.com/effame/stockgen-upscaler/pkgs/container/stockgen-upscaler)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%20%7C%20CUDA%2012-red.svg?logo=pytorch)](https://pytorch.org/)
[![RunPod Serverless](https://img.shields.io/badge/RunPod-Serverless%20GPU-purple)](https://runpod.io/)

A high-performance, production-ready **RunPod Serverless GPU Worker** designed for commercial-grade stock photo upscaling (2K to 4K/8K), face restoration, background removal, 300 DPI metadata injection, and direct Cloudflare R2 uploads.

---

## 📁 Repository Structure (Monorepo)

```text
stockgen-upscaler/
  ├── web/                 # 🌐 Next.js 16 Web Application (Batch Upscaler UI)
  ├── scripts/             # 🛠️ Maintenance & Deployment Scripts
  ├── cli/                 # 💻 Python CLI Tool
  ├── colab/               # 📓 Google Colab Notebooks
  ├── experimental/        # 🧪 Experimental & Debug Files
  ├── Dockerfile           # 🐳 RunPod Serverless GPU Worker Docker image
  ├── rp_handler.py        # ⚡ RunPod Serverless Worker Handler
  └── models.py            # 🧠 Model loaders (Real-ESRGAN, GFPGAN)
```

---

## ⚡ Core Capabilities

- 🔍 **AI Super Resolution (Real-ESRGAN & UltraSharp)**:
  - `x4plus` (General 4x photo upscaling)
  - `x2plus` (Fast 2x photo upscaling)
  - `ultrasharp` (Extreme high-definition commercial clarity)
  - `anime` (Real-ESRGAN Anime 6B model)
  - `v3` (Ultra-fast compact general model)
- 👤 **Face Restoration (GFPGAN v1.3)**: Automatic facial defect repair and high-definition reconstruction.
- ✂️ **Smart Alpha Matting (Rembg)**: Background removal with anti-halo defringing.
- 🖨️ **Print-Ready Stock Standards**: Automatically embeds **300 DPI** EXIF metadata.
- ☁️ **Direct Cloudflare R2 Pipeline**: Uploads high-res output directly to R2 CDN, avoiding API payload limitations (bypassing 400 Bad Request).

---

## 🐳 Docker Image

The pre-built Docker image is available on GitHub Container Registry:

```bash
docker pull ghcr.io/effame/stockgen-upscaler:latest
```

---

## 🛠️ RunPod Serverless Setup

1. Go to **[RunPod Serverless](https://www.runpod.io/console/serverless)** and click **"New Endpoint"**.
2. **Container Image**: `ghcr.io/effame/stockgen-upscaler:latest`
3. **GPU Selection**: Minimum 16GB VRAM (e.g. RTX 4000 Ada, RTX 3090, RTX 4090, L4).
4. **Environment Variables**:
   ```env
   R2_ACCESS_KEY_ID=your_cloudflare_r2_access_key
   R2_SECRET_ACCESS_KEY=your_cloudflare_r2_secret_key
   R2_ENDPOINT=https://<account_id>.r2.cloudflarestorage.com
   R2_BUCKET_NAME=your_bucket_name
   R2_CUSTOM_DOMAIN=https://your-custom-domain.com # Optional
   WEIGHTS_DIR=/workspace/weights # Optional network volume path
   ```

---

## 📥 API Input Payload

Send a POST request to your RunPod Serverless endpoint (`/runsync` or `/run`):

```json
{
  "input": {
    "image": "https://example.com/image.jpg", 
    "scale": 4,
    "model": "x4plus",
    "face_enhance": true,
    "remove_bg": false,
    "r2_key": "upscaled/photo_4k_01.jpg"
  }
}
```

### Parameters:
| Field | Type | Default | Description |
|---|---|---|---|
| `image` | string | *required* | Base64 string or public image URL |
| `scale` | number | `4` | Scale factor (`2` or `4`) |
| `model` | string | `"x4plus"` | Model: `x4plus`, `x2plus`, `ultrasharp`, `anime`, `v3` |
| `face_enhance`| boolean | `false` | Enable GFPGAN v1.3 face enhancement |
| `remove_bg` | boolean | `false` | Enable Rembg alpha cutout (outputs PNG) |
| `r2_key` | string | *required* | Target destination key in Cloudflare R2 |

---

## 📤 API Response

```json
{
  "r2Url": "https://your-custom-domain.com/upscaled/photo_4k_01.jpg",
  "model": "x4plus",
  "scale": 4,
  "remove_bg": false,
  "face_enhance": true,
  "input_size": { "width": 1024, "height": 1024 },
  "output_size": { "width": 4096, "height": 4096 },
  "output_file_size": 1845210,
  "output_dpi": 300,
  "output_color_space": "sRGB"
}
```

---

## 🌐 Connected Ecosystem

- **Frontend Client**: [Batch Upscaler Web (Open Source)](https://github.com/effame/batch-upscaler) — [Live Demo](https://batch-upscaler.vercel.app)
- **Flagship Application**: [StockGen AI](https://stockgen-ai.com) — Full-suite AI Stock Media Creator & Metadata Generator.

---

## 📜 Acknowledgements & Open Source Credits

This project builds upon and integrates extraordinary open-source research and tools. We would like to give sincere thanks and full credit to:

1. **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** by Xintao Wang et al. — Practical Algorithms for General Image Restoration.
2. **[GFPGAN](https://github.com/TencentARC/GFPGAN)** by Tencent ARC Lab / Xintao Wang — Towards Real-World Blind Face Restoration with Generative Facial Prior.
3. **[4x-UltraSharp](https://huggingface.co/Kim2091/UltraSharp)** by Kim2091 — Outstanding high-detail upscaling weights.
4. **[Rembg](https://github.com/danielgatis/rembg)** by Daniel Gatis — Background removal tool based on U2-Net.
5. **[RunPod Python SDK](https://github.com/runpod/runpod-python)** — Serverless cloud computing infrastructure.

---

## ⚖️ License

Distributed under the [MIT License](LICENSE).
