from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import ROOT_DIR, settings
from app.services.metrics import Metrics
from app.services.runtime import ModelRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = ModelRuntime(settings.model_path, settings.metadata_path)
    runtime.load()
    app.state.runtime = runtime
    app.state.metrics = Metrics()
    app.state.settings = settings
    yield


app = FastAPI(
    title="ModelServe API",
    description="A validated inference service for a trained Iris Random Forest model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_failure()
    return JSONResponse(status_code=422, content=jsonable_encoder({"error": {"code": "VALIDATION_ERROR", "message": "The request did not match the expected schema.", "details": exc.errors()}}))


@app.get("/health", tags=["health"])
def health(request: Request):
    ready = getattr(request.app.state, "runtime", None)
    online = bool(ready and ready.ready)
    return JSONResponse(status_code=200 if online else 503, content={"status": "ok" if online else "unavailable", "model_status": "online" if online else "offline"})


app.include_router(router)
app.mount("/", StaticFiles(directory=ROOT_DIR / "app" / "static", html=True), name="console")
