import json
from datetime import datetime, timezone

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import settings


def train() -> dict:
    dataset = load_iris()
    labels = dataset.target_names[dataset.target]
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    estimator = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced")
    pipeline = Pipeline([("classifier", estimator)])
    pipeline.fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, pipeline.predict(x_test)))
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.model_path, compress=3)
    metadata = {
        "name": settings.name,
        "version": settings.version,
        "framework": f"scikit-learn {__import__('sklearn').__version__}",
        "algorithm": "RandomForestClassifier",
        "dataset": "Iris (scikit-learn bundled dataset)",
        "feature_names": list(dataset.feature_names),
        "target_classes": [str(name) for name in dataset.target_names],
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy": accuracy,
        "evaluation": {"method": "stratified holdout", "test_size": 0.2, "random_state": 42, "test_samples": int(len(y_test))},
        "n_estimators": int(pipeline.named_steps["classifier"].n_estimators),
        "training_samples": int(len(y_train)),
        "artifact": {"filename": settings.model_path.name, "size_bytes": settings.model_path.stat().st_size},
        "status": "trained",
    }
    settings.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
