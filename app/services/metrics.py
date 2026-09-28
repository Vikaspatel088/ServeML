from collections import Counter, deque
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
from typing import Any


class Metrics:
    """Process-local inference telemetry; intentionally not durable."""

    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self._lock = Lock()
        self._latencies: deque[float] = deque(maxlen=500)
        self._recent: deque[dict[str, Any]] = deque(maxlen=12)
        self._distribution: Counter[str] = Counter()
        self.total = 0
        self.successful = 0
        self.failed = 0

    def record_success(self, result: dict[str, Any], latency: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.total += 1
            self.successful += 1
            self._latencies.append(latency)
            self._distribution[result["prediction"]] += 1
            self._recent.appendleft({"prediction": result["prediction"], "confidence": max(result["probabilities"].values()), "latency_ms": round(latency, 3), "timestamp": now})

    def record_failure(self) -> None:
        with self._lock:
            self.total += 1
            self.failed += 1

    def snapshot(self, model_online: bool) -> dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies)
            total, success, failed = self.total, self.successful, self.failed
            recent, distribution = list(self._recent), dict(self._distribution)
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return {
            "model_status": "online" if model_online else "offline",
            "total_predictions": total,
            "successful_predictions": success,
            "failed_predictions": failed,
            "success_rate": round(success / total * 100, 2) if total else 100.0,
            "average_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "uptime_seconds": round(uptime, 1),
            "latest_prediction": recent[0] if recent else None,
            "recent_predictions": recent,
            "prediction_distribution": distribution,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }


def timed() -> float:
    return perf_counter()
