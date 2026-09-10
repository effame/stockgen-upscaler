# ⚡ Batch Upscaler AI

[![CI Build Check](https://github.com/effame/batch-upscaler/actions/workflows/ci.yml/badge.svg)](https://github.com/effame/batch-upscaler/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![RunPod Serverless](https://img.shields.io/badge/RunPod-Serverless%20GPU-purple)](https://runpod.io/)

A fast, lightweight, and open-source batch image upscaling web tool powered by **RunPod Serverless GPU (Real-ESRGAN)**.

Upload 10 to 50+ images at once, upscale to 4K UHD, fix faces, remove backgrounds, and download everything as a single ZIP file in one click.

---

## ✨ Features

- 🚀 **Batch Processing**: Drag & drop 10–50+ images simultaneously.
- 🔍 **Ultra-Sharp 4K**: Powered by Real-ESRGAN (2x & 4x scaling).
- 👤 **Face Fix (GFPGAN)**: Restore and enhance facial details automatically.
- ✂️ **Background Removal**: Clean alpha cutout powered by Rembg.
- 📦 **1-Click ZIP Download**: Package all upscaled images into a single `.zip`.
- 🔒 **Zero Data Retention**: Pure client-to-GPU tool. No databases, no login, no accounts required.

---

## 🛠️ Tech Stack

- **Frontend**: Next.js 16 (App Router), React 19, TypeScript
- **Styling**: Tailwind CSS v4, Lucide Icons
- **Compression**: JSZip & FileSaver
- **Backend Worker**: RunPod Serverless GPU (PyTorch, Real-ESRGAN, GFPGAN)

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/effame/batch-upscaler.git
cd batch-upscaler
```

### 2. Install dependencies
```bash
npm install
```

### 3. Configure Environment Variables
Create a `.env.local` file in the root directory:
```env
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_runpod_serverless_endpoint_id
```

### 4. Run the development server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
