import os
import subprocess
import tempfile
from pathlib import Path

from cog import BasePredictor, Input, Path


MODEL_DIR = "/src/models"
BIN_DIR = "/src/bin"


class Predictor(BasePredictor):
    def setup(self):
        self.bin_path = os.path.join(BIN_DIR, "upscayl-bin")
        self.model_path = os.path.join(MODEL_DIR, "high-fidelity-4x")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: int = Input(
            description="Upscale scale factor",
            default=4,
            ge=2,
            le=8,
        ),
    ) -> Path:
        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / "output.png"

        cmd = [
            self.bin_path,
            "-i", str(image),
            "-o", str(out_path),
            "-m", self.model_path,
            "-n", "high-fidelity-4x",
            "-s", str(scale),
            "-z",
        ]
        subprocess.run(cmd, check=True)

        return Path(str(out_path))
