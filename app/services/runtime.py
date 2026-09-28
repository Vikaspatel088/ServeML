import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ModelRuntime:
    def __init__(self, model_path: Path, metadata_path: Path):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model: Any | None = None
        self.metadata: dict[str, Any] | None = None

    def load(self) -> None:
        if not self.model_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found in {self.model_path.parent}. "
                "Run `python -m app.training.train` before starting the API."
            )
        model = joblib.load(self.model_path)
        with self.metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise ValueError("Loaded artifact does not provide predict and predict_proba")
        if [str(value) for value in model.classes_] != [str(value) for value in metadata["target_classes"]]:
            raise ValueError("Model classes do not match the artifact metadata")
        self.model = model
        self.metadata = metadata

    @property
    def ready(self) -> bool:
        return self.model is not None and self.metadata is not None

    def infer(self, rows: list[list[float]]) -> list[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("Model runtime is not ready")
        features = np.asarray(rows, dtype=float)
        labels = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        classes = [str(value) for value in self.model.classes_]
        return [
            {
                "prediction": str(label),
                "class_index": classes.index(str(label)),
                "probabilities": {name: float(score) for name, score in zip(classes, scores)},
            }
            for label, scores in zip(labels, probabilities)
        ]
