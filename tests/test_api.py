from fastapi.testclient import TestClient

from app.main import app


def test_health_and_metadata():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        metadata = client.get("/api/v1/model/metadata").json()
        assert metadata["algorithm"] == "RandomForestClassifier"
        assert len(metadata["feature_names"]) == 4
        assert len(metadata["target_classes"]) == 3


def test_single_prediction_uses_model_response():
    with TestClient(app) as client:
        result = client.post("/api/v1/predict", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
        assert result.status_code == 200
        body = result.json()
        assert body["prediction"] == "setosa"
        assert body["class_index"] == 0
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-9
        assert body["model"]["version"] == "1.0.0"
        assert body["latency_ms"] >= 0


def test_batch_prediction_and_metrics():
    rows = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
    ]
    with TestClient(app) as client:
        result = client.post("/api/v1/predict/batch", json={"instances": rows})
        assert result.status_code == 200
        assert result.json()["count"] == 2
        metrics = client.get("/api/v1/metrics").json()
        assert metrics["total_predictions"] == 2
        assert metrics["successful_predictions"] == 2


def test_invalid_and_extra_input_are_rejected():
    with TestClient(app) as client:
        invalid = client.post("/api/v1/predict", json={"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
        extra = client.post("/api/v1/predict", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"})
        empty_batch = client.post("/api/v1/predict/batch", json={"instances": []})
        assert invalid.status_code == extra.status_code == empty_batch.status_code == 422
        assert invalid.json()["error"]["code"] == "VALIDATION_ERROR"
