import os
import piexif
from PIL import Image
import requests
from io import BytesIO

"""
🧪 StockGen AI - Python DPI 300 Metadata Embedding Test
วัตถุประสงค์: ทดสอบการฝังค่า DPI 300 ในระดับ Worker (Python) เพื่อให้รูปภาพพร้อมขายก่อนอัปโหลดเข้า R2
"""

def test_python_dpi_embedding():
    print("===============================================================")
    print("🧪 PYTHON DPI 300 METADATA EMBEDDING TEST")
    print("===============================================================\n")

    # 1. เตรียมรูปภาพจำลอง
    img_url = "https://images.unsplash.com/photo-1579546929518-9e396f3cc809?w=300&q=80"
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    
    # 2. สร้าง EXIF Data สำหรับ DPI 300
    # XResolution, YResolution เป็นแบบ Rational (numerator, denominator)
    # ResolutionUnit: 2 = Inches
    zeroth_ifd = {
        piexif.ImageIFD.XResolution: (300, 1),
        piexif.ImageIFD.YResolution: (300, 1),
        piexif.ImageIFD.ResolutionUnit: 2,
        piexif.ImageIFD.ImageDescription: "StockGen AI - Python Worker Test".encode("utf-8")
    }
    
    exif_dict = {"0th": zeroth_ifd, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_bytes = piexif.dump(exif_dict)

    # 3. บันทึกไฟล์พร้อมฝัง EXIF
    output_path = "temp_python_dpi_output.jpg"
    img.save(output_path, "jpeg", exif=exif_bytes, quality=95)
    print(f"✨ Embedded DPI 300 and saved to: {output_path}")

    # 4. ตรวจสอบข้อมูลกลับ (Verification)
    verify_img = Image.open(output_path)
    info = verify_img.info.get("exif")
    if info:
        exif_data = piexif.load(info)
        x_res = exif_data["0th"].get(piexif.ImageIFD.XResolution)
        y_res = exif_data["0th"].get(piexif.ImageIFD.YResolution)
        unit = exif_data["0th"].get(piexif.ImageIFD.ResolutionUnit)

        print("\n📊 VERIFICATION REPORT:")
        print(f"📌 X-Resolution: {x_res[0]}/{x_res[1]} ({x_res[0]/x_res[1]} DPI)")
        print(f"📌 Y-Resolution: {y_res[0]}/{y_res[1]} ({y_res[0]/y_res[1]} DPI)")
        print(f"📌 Resolution Unit: {'2 (Inches)' if unit == 2 else unit}")
        
        if x_res == (300, 1) and unit == 2:
            print("\n🎉 SUCCESS: PYTHON DPI 300 VERIFICATION PASSED!")
        else:
            print("\n❌ ERROR: Verification failed!")
    else:
        print("\n❌ ERROR: No EXIF data found in output image!")

if __name__ == "__main__":
    test_python_dpi_embedding()
