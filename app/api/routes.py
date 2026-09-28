from time import perf_counter

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.schemas.prediction import BatchRequest, BatchResponse, IrisFeatures, ModelRef, PredictionResponse

router = APIRouter(prefix="/api/v1")


@router.get("/docs", include_in_schema=False)
def api_docs():
    return RedirectResponse(url="/docs")


def model_ref(request: Request) -> ModelRef:
    return ModelRef(name=request.app.state.runtime.metadata["name"], version=request.app.state.runtime.metadata["version"])


@router.get("/model", tags=["model"])
def model_summary(request: Request):
    metadata = request.app.state.runtime.metadata
    return {"name": metadata["name"], "version": metadata["version"], "status": "online"}


@router.get("/model/metadata", tags=["model"])
def model_metadata(request: Request):
    metadata = dict(request.app.state.runtime.metadata)
    metadata["status"] = "serving" if request.app.state.runtime.ready else "offline"
    metadata["artifact"]["size_bytes"] = request.app.state.runtime.model_path.stat().st_size
    return metadata


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: IrisFeatures, request: Request):
    started = perf_counter()
    try:
        result = request.app.state.runtime.infer([payload.as_row()])[0]
        latency = (perf_counter() - started) * 1000
        request.app.state.metrics.record_success(result, latency)
        return {**result, "model": model_ref(request), "latency_ms": round(latency, 3)}
    except Exception as exc:
        request.app.state.metrics.record_failure()
        raise HTTPException(status_code=503, detail={"code": "INFERENCE_UNAVAILABLE", "message": "The model could not complete this inference."}) from exc


@router.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(payload: BatchRequest, request: Request):
    if len(payload.instances) > request.app.state.settings.max_batch_size:
        raise HTTPException(status_code=422, detail={"code": "BATCH_TOO_LARGE", "message": f"Maximum batch size is {request.app.state.settings.max_batch_size}."})
    started = perf_counter()
    try:
        results = request.app.state.runtime.infer([row.as_row() for row in payload.instances])
        latency = (perf_counter() - started) * 1000
        per_item_latency = latency / len(results)
        for result in results:
            request.app.state.metrics.record_success(result, per_item_latency)
        ref = model_ref(request)
        return {"predictions": [{**result, "model": ref, "latency_ms": round(per_item_latency, 3)} for result in results], "count": len(results), "model": ref, "latency_ms": round(latency, 3)}
    except Exception as exc:
        request.app.state.metrics.record_failure()
        raise HTTPException(status_code=503, detail={"code": "INFERENCE_UNAVAILABLE", "message": "The model could not complete this batch."}) from exc


@router.get("/metrics", tags=["observability"])
def metrics(request: Request):
    return request.app.state.metrics.snapshot(request.app.state.runtime.ready)
