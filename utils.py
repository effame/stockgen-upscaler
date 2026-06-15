import os
import io
import base64
import tempfile
import requests
import boto3
import cv2
import numpy as np
from PIL import Image

# TurboJPEG — ~1.5-2x faster encode than OpenCV
try:
    from turbojpeg import TurboJPEG
    _tj = TurboJPEG()
except Exception:
    _tj = None

# piexif — DPI metadata injection (safe fallback)
try:
    import piexif
except Exception:
    piexif = None

# ---------------------------------------------------------------------------
# S3 (Cloudflare R2) — lazy init
# ---------------------------------------------------------------------------
s3_client = None

def get_s3_client():
    global s3_client
    if s3_client is None:
        access_key = os.environ.get("R2_ACCESS_KEY_ID")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        endpoint = os.environ.get("R2_ENDPOINT")
        if all([access_key, secret_key, endpoint]):
            try:
                from botocore.config import Config
                s3_client = boto3.client(
                    "s3",
                    endpoint_url=endpoint,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    config=Config(signature_version="s3v4"),
                    region_name="auto",
                )
                print("[R2] S3 Client initialized successfully.")
            except Exception as e:
                print(f"[R2] Failed to initialize S3 Client: {e}")
    return s3_client

def upload_to_r2(buffer, key, content_type="image/jpeg"):
    client = get_s3_client()
    if client is None:
        print("[R2] Upload failed: S3 Client is None (check Env Vars)")
        return None

    bucket = os.environ.get("R2_BUCKET_NAME", "stockgen-ai")
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer,
            ContentType=content_type,
            CacheControl="public, max-age=31536000",
        )
        custom_domain = os.environ.get("R2_CUSTOM_DOMAIN")
        if custom_domain:
            return f"{custom_domain.rstrip('/')}/{key}"
        endpoint = os.environ.get("R2_ENDPOINT", "").rstrip("/")
        return f"{endpoint}/{bucket}/{key}"
    except Exception as e:
        print(f"[R2] Upload Error: {e}")
        return None

# ---------------------------------------------------------------------------
# Weight download helpers
# ---------------------------------------------------------------------------
def download_file(url, dest, timeout=300):
    for attempt in (1, 2):
        try:
            print(f"Downloading {os.path.basename(dest)}... (attempt {attempt})")
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16 * 1024):
                    f.write(chunk)
            return
        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
            if attempt == 2:
                raise

# ---------------------------------------------------------------------------
# Image Helpers
# ---------------------------------------------------------------------------
def apply_exif_orientation(img_bgr, raw_bytes):
    if raw_bytes is None:
        return img_bgr
    try:
        pil = Image.open(io.BytesIO(raw_bytes))
        orientation = pil.getexif().get(0x0112)
        rot_map = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if orientation in rot_map:
            img_bgr = cv2.rotate(img_bgr, rot_map[orientation])
    except Exception:
        pass
    return img_bgr

def inject_dpi(jpeg_bytes, dpi=300):
    if piexif is None:
        return jpeg_bytes
    try:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
        try:
            exif_dict = piexif.load(jpeg_bytes)
        except ValueError:
            pass
        exif_dict["0th"][piexif.ImageIFD.XResolution] = (dpi, 1)
        exif_dict["0th"][piexif.ImageIFD.YResolution] = (dpi, 1)
        exif_dict["0th"][piexif.ImageIFD.ResolutionUnit] = 2
        exif_bytes = piexif.dump(exif_dict)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(jpeg_bytes)
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            out_path = f.name
        try:
            piexif.insert(exif_bytes, in_path, out_path)
            with open(out_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(in_path)
            except OSError:
                pass
            try:
                os.remove(out_path)
            except OSError:
                pass
    except Exception as e:
        print(f"[DPI] Warning: failed to embed DPI {dpi}: {e}")
        return jpeg_bytes

def encode_image(bgr_array, image_format, quality=95):
    if image_format == "png":
        success, encoded = cv2.imencode(".png", bgr_array, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            return None
        return encoded.tobytes()

    if _tj is not None:
        try:
            return _tj.encode(bgr_array, quality=quality)
        except Exception as e:
            print(f"[JPEG] TurboJPEG failed, falling back to OpenCV: {e}")

    success, encoded = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None
    return encoded.tobytes()

def load_image(image_source):
    raw = None
    if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
        resp = requests.get(image_source, headers={"User-Agent": "StockGen-AI/1.0 (upscaler)"}, timeout=60)
        resp.raise_for_status()
        raw = resp.content
    elif isinstance(image_source, str) and os.path.exists(image_source):
        with open(image_source, "rb") as f:
            raw = f.read()
    else:
        try:
            if isinstance(image_source, str) and "," in image_source:
                image_source = image_source.split(",", 1)[1]
            raw = base64.b64decode(image_source)
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from source")
    return apply_exif_orientation(img, raw)
