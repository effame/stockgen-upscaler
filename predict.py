import subprocess
import sys
import tempfile
from pathlib import Path
from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def predict(
        self,
        image: Path = Input(description="Input image to upscale"),
        scale: int = Input(
            description="Upscale factor (2, 3, or 4)",
            default=4,
            choices=[2, 3, 4],
        ),
        tile: int = Input(
            description="Tile size for segmented processing (0 = off, 200-400 recommended for large images)",
            default=0,
            ge=0,
            le=1024,
        ),
        output_format: str = Input(
            description="Output image format",
            default="png",
            choices=["png", "jpg", "webp"],
        ),
    ) -> Path:
        sys.stderr.write(f"Image type: {type(image)}, value: {image!r}\n")
        sys.stderr.write(f"Image is_file: {Path(image).is_file()}\n")
        sys.stderr.write(f"Models: {list(Path('/src/models').iterdir())}\n")

        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / f"output.{output_format}"

        cmd = [
            "upscayl-bin",
            "-i", str(image),
            "-o", str(out_path),
            "-s", str(scale),
            "-m", "/src/models",
            "-n", "high-fidelity-4x",
            "-f", output_format,
            "-c", "0",
        ]

        if tile > 0:
            cmd.extend(["-t", str(tile)])

        sys.stderr.write(f"Running: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=False, text=True)

        sys.stderr.write(f"Return code: {result.returncode}\n")

        if not out_path.exists():
            sys.stderr.write(f"Output file not found at {out_path}\n")
            sys.stderr.write(
                f"Directory contents: {[str(p) for p in out_dir.iterdir()]}\n"
            )
            raise FileNotFoundError(f"Output file not created at {out_path}")

        sys.stderr.write(f"Output size: {out_path.stat().st_size} bytes\n")

        return Path(out_path)
