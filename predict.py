import pathlib
import subprocess
import sys
import tempfile
from cog import BasePredictor, Input, Path


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
            description="Tile size (0 = off, 200-400 for large images)",
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
        input_path = pathlib.Path(str(image))
        sys.stderr.write(f"Input: {input_path}, exists: {input_path.exists()}, size: {input_path.stat().st_size}\n")

        out_dir = pathlib.Path(tempfile.mkdtemp())
        out_path = out_dir / f"output.{output_format}"

        cmd = [
            "upscayl-bin",
            "-i", str(input_path),
            "-o", str(out_path),
            "-s", str(scale),
            "-m", "/src/models",
            "-n", "high-fidelity-4x",
            "-f", output_format,
            "-c", "0",
        ]

        if tile > 0:
            cmd.extend(["-t", str(tile)])

        sys.stderr.write(f"Cmd: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=False, text=True)

        sys.stderr.write(f"Return code: {result.returncode}\n")

        if not out_path.exists():
            ls = [str(p) for p in out_dir.iterdir()]
            sys.stderr.write(f"Output not found. Dir: {ls}\n")
            raise FileNotFoundError(f"Output not created at {out_path}")

        sys.stderr.write(f"Output: {out_path.stat().st_size} bytes\n")
        return Path(str(out_path))
