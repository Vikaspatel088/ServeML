import json

import joblib
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from app.services.runtime import ModelRuntime


def test_runtime_loads_and_predicts_from_serialized_estimator(tmp_path):
    dataset = load_iris()
    model = RandomForestClassifier(n_estimators=5, random_state=9).fit(dataset.data, dataset.target)
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps({"target_classes": [str(c) for c in model.classes_]}), encoding="utf-8")
    runtime = ModelRuntime(model_path, metadata_path)
    runtime.load()
    output = runtime.infer([[5.1, 3.5, 1.4, 0.2]])[0]
    assert output["prediction"] == str(model.predict([[5.1, 3.5, 1.4, 0.2]])[0])
    assert output["probabilities"] == dict(zip([str(c) for c in model.classes_], model.predict_proba([[5.1, 3.5, 1.4, 0.2]])[0]))


def test_runtime_rejects_missing_artifact(tmp_path):
    runtime = ModelRuntime(tmp_path / "missing.joblib", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="python -m app.training.train"):
        runtime.load()
