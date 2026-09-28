from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings:
    name: str = os.getenv("MODELSERVE_NAME", "iris-random-forest")
    version: str = os.getenv("MODELSERVE_VERSION", "1.0.0")
    artifact_dir: Path = Path(os.getenv("MODELSERVE_ARTIFACT_DIR", ROOT_DIR / "artifacts"))
    max_batch_size: int = int(os.getenv("MODELSERVE_MAX_BATCH_SIZE", "250"))

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "iris_model.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / "model_metadata.json"


settings = Settings()
