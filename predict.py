import subprocess
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

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        return Path(out_path)
