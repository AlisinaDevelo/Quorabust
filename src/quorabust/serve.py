from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from quorabust.explain import explain_pair_features
from quorabust.model import predict_proba_duplicate
from quorabust.persist import load_classifier

DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_MAX_BATCH_SIZE = 256
API_KEY_HEADER = "X-Quorabust-API-Key"
REQUEST_ID_HEADER = "X-Request-ID"
HTTP_LOGGER = logging.getLogger("quorabust.http")


def _request_id(raw: str | None) -> str:
    if raw:
        try:
            parsed = uuid.UUID(raw)
        except (AttributeError, ValueError):
            pass
        else:
            canonical = str(parsed)
            if canonical == raw.lower():
                return canonical
    return str(uuid.uuid4())


def _log_http_event(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    payload = {
        "duration_ms": round(duration_ms, 3),
        "event": "http.request",
        "method": method,
        "path": path,
        "request_id": request_id,
        "status_code": status_code,
    }
    message = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    if status_code >= 500:
        HTTP_LOGGER.error(message)
    elif status_code >= 400:
        HTTP_LOGGER.warning(message)
    else:
        HTTP_LOGGER.info(message)


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "<unmatched>"


def _dist_version() -> str:
    try:
        return version("Quorabust")
    except PackageNotFoundError:
        return "0.0.0"


def _env_decision_threshold() -> float:
    raw = os.environ.get("QUORABUST_DECISION_THRESHOLD")
    if raw is None:
        return DEFAULT_DECISION_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_DECISION_THRESHOLD
    if 0.0 <= value <= 1.0:
        return value
    return DEFAULT_DECISION_THRESHOLD


def _model_decision_threshold(meta: dict[str, Any], fallback: float) -> float:
    raw = meta.get("decision_threshold")
    if isinstance(raw, int | float) and 0.0 <= float(raw) <= 1.0:
        return float(raw)
    return fallback


def _env_max_batch_size() -> int:
    raw = os.environ.get("QUORABUST_MAX_BATCH_SIZE")
    if raw is None:
        return DEFAULT_MAX_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_BATCH_SIZE
    return value if value > 0 else DEFAULT_MAX_BATCH_SIZE


class PredictBody(BaseModel):
    """Batch of question pairs; ``question1[i]`` is paired with ``question2[i]``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question1": [
                        "How do I learn Python?",
                        "Best pizza in NYC?",
                    ],
                    "question2": [
                        "What is the best way to learn Python?",
                        "Where is good pizza in New York?",
                    ],
                }
            ]
        }
    )

    question1: list[str] = Field(
        ...,
        min_length=1,
        description="First question in each pair (same length as question2).",
    )
    question2: list[str] = Field(
        ...,
        min_length=1,
        description="Second question in each pair (same length as question1).",
    )


class PredictOut(BaseModel):
    """Per-row duplicate probabilities and which model variant served the request."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "proba_duplicate": [0.82, 0.31],
                    "is_duplicate": [True, False],
                    "decision_threshold": 0.5,
                    "variant": "a",
                    "features": [
                        {
                            "cos": 0.73,
                            "jaccard": 0.42,
                            "len_ratio": 0.8,
                            "abs_len_diff": 1.0,
                            "len_sum": 9.0,
                        }
                    ],
                }
            ]
        }
    )

    proba_duplicate: list[float] = Field(
        ...,
        description="P(duplicate) for each pair, same order as the request lists.",
    )
    is_duplicate: list[bool] = Field(
        ...,
        description=(
            "Thresholded duplicate decision for each pair, same order as the request lists."
        ),
    )
    decision_threshold: float = Field(
        ...,
        description=(
            "Probability threshold used to compute is_duplicate. Request threshold wins over "
            "artifact metadata; artifact metadata wins over QUORABUST_DECISION_THRESHOLD."
        ),
    )
    variant: str = Field(..., description="Scoring variant (a or b) after A/B fallback rules.")
    features: list[dict[str, float]] | None = Field(
        default=None,
        description=(
            "Optional per-pair model input feature values when `explain=true`. "
            "These are feature values, not causal explanations."
        ),
    )


_PUBLIC_META_KEYS = {
    "feature_backend",
    "feature_schema",
    "n_train",
    "n_eval",
    "seed",
    "quorabust_version",
    "git_revision",
    "csv_sha256",
    "reference_feature_means",
    "decision_threshold",
    "decision_threshold_source",
    "decision_threshold_metric",
    "decision_threshold_metrics",
}


def _public_model_meta(meta: dict[str, Any]) -> dict[str, Any]:
    public = {k: meta[k] for k in _PUBLIC_META_KEYS if k in meta}
    eval_metrics = {
        k.removeprefix("eval_"): v
        for k, v in sorted(meta.items())
        if k.startswith("eval_") and isinstance(v, int | float)
    }
    if eval_metrics:
        public["eval_metrics"] = eval_metrics
    return public


def create_app(
    model_path_a: str | None = None,
    model_path_b: str | None = None,
    *,
    api_key: str | None = None,
    max_batch_size: int | None = None,
) -> FastAPI:
    path_a = model_path_a or os.environ.get("QUORABUST_MODEL_PATH", "")
    path_b = model_path_b or os.environ.get("QUORABUST_MODEL_B", "")
    default_threshold = _env_decision_threshold()
    configured_api_key = api_key if api_key is not None else os.environ.get("QUORABUST_API_KEY")
    configured_api_key = configured_api_key or None
    configured_max_batch_size = (
        max_batch_size if max_batch_size is not None else _env_max_batch_size()
    )
    if configured_max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1")

    api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

    def require_api_key(
        provided_key: str | None = Depends(api_key_header),
    ) -> None:
        if configured_api_key and (
            provided_key is None or not hmac.compare_digest(provided_key, configured_api_key)
        ):
            raise HTTPException(
                status_code=401,
                detail="invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    registry = CollectorRegistry()
    predictions = Counter(
        "quorabust_predictions_total",
        "Scoring requests",
        ["variant"],
        registry=registry,
    )
    latency = Histogram(
        "quorabust_predict_latency_seconds",
        "Predict latency",
        ["variant"],
        registry=registry,
    )
    http_requests = Counter(
        "quorabust_http_requests_total",
        "HTTP requests by method, path, and status code",
        ["method", "path", "status_code"],
        registry=registry,
    )
    http_latency = Histogram(
        "quorabust_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "path"],
        registry=registry,
    )

    def observe_http(
        request: Request,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        http_requests.labels(request.method, path, str(status_code)).inc()
        http_latency.labels(request.method, path).observe(duration_seconds)

    state: dict[str, tuple[Any, Any, dict[str, Any]]] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if path_a:
            pa = Path(path_a)
            if pa.is_file():
                state["a"] = load_classifier(pa)
        if path_b:
            pb = Path(path_b)
            if pb.is_file():
                state["b"] = load_classifier(pb)
        yield

    app = FastAPI(
        title=f"Quorabust {_dist_version()}",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "scoring", "description": "Duplicate probability for question pairs."},
            {
                "name": "operations",
                "description": "Liveness, readiness, and Prometheus metrics.",
            },
        ],
    )

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = _request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.quorabust_request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_seconds = time.perf_counter() - started
            path = _route_path(request)
            observe_http(request, path, 500, duration_seconds)
            _log_http_event(
                request_id=request_id,
                method=request.method,
                path=path,
                status_code=500,
                duration_ms=duration_seconds * 1000,
            )
            raise
        duration_seconds = time.perf_counter() - started
        path = _route_path(request)
        observe_http(request, path, response.status_code, duration_seconds)
        response.headers[REQUEST_ID_HEADER] = request_id
        _log_http_event(
            request_id=request_id,
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_ms=duration_seconds * 1000,
        )
        return response

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    def ready() -> dict[str, str]:
        if "a" not in state:
            raise HTTPException(status_code=503, detail="model a not loaded")
        return {"status": "ready"}

    @app.get(
        "/models",
        dependencies=[Depends(require_api_key)],
        tags=["operations"],
        summary="List loaded model metadata",
        description=(
            "Returns allowlisted metadata for loaded variants. Local artifact paths and "
            "training CSV paths are intentionally omitted. If `QUORABUST_API_KEY` is "
            "configured, send it in the `X-Quorabust-API-Key` header. Every response "
            "includes an `X-Request-ID` UUID for support and log correlation."
        ),
        responses={
            200: {
                "description": "Loaded model metadata",
                "headers": {
                    REQUEST_ID_HEADER: {
                        "description": "Request correlation UUID.",
                        "schema": {"type": "string", "format": "uuid"},
                    }
                },
            },
            401: {"description": "API key required or invalid"},
            503: {"description": "No models loaded"},
        },
    )
    def models() -> dict[str, dict[str, dict[str, Any]]]:
        if not state:
            raise HTTPException(status_code=503, detail="no models loaded")
        variants = {
            name: _public_model_meta(meta)
            for name, (_, _, meta) in state.items()
        }
        return {"variants": variants}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["operations"])
    def metrics() -> PlainTextResponse:
        data = generate_latest(registry)
        return PlainTextResponse(
            data.decode("utf-8"),
            media_type="text/plain; version=0.0.4",
        )

    @app.post(
        "/predict",
        dependencies=[Depends(require_api_key)],
        response_model=PredictOut,
        tags=["scoring"],
        summary="Predict duplicate probability",
        description=(
            "Scores one or more question pairs. Optional header "
            "`X-Quorabust-Variant: b` selects the B artifact when configured. "
            "Set query parameter `explain=true` to return input feature values. "
            "Set `threshold` to override the duplicate decision cutoff for this request. "
            "Deployments can configure `QUORABUST_API_KEY` and "
            "`QUORABUST_MAX_BATCH_SIZE` for access control and bounded work. Every response "
            "includes an `X-Request-ID` UUID for support and log correlation."
        ),
        responses={
            200: {
                "description": "Duplicate probabilities and decisions",
                "headers": {
                    REQUEST_ID_HEADER: {
                        "description": "Request correlation UUID.",
                        "schema": {"type": "string", "format": "uuid"},
                    }
                },
            },
            401: {"description": "API key required or invalid"},
            400: {"description": "question1 and question2 length mismatch"},
            413: {"description": "request batch exceeds configured maximum"},
            503: {"description": "Model not loaded or variant unavailable"},
        },
    )
    def predict(
        body: PredictBody,
        explain: bool = False,
        threshold: float | None = Query(
            default=None,
            ge=0.0,
            le=1.0,
            description=(
                "Optional probability cutoff for is_duplicate. Defaults to artifact metadata "
                "decision_threshold when present, otherwise QUORABUST_DECISION_THRESHOLD or 0.5."
            ),
        ),
        x_quorabust_variant: str | None = Header(
            default=None,
            alias="X-Quorabust-Variant",
            description="A/B variant: `a` (default) or `b` if a second model is loaded.",
        ),
    ) -> PredictOut:
        v = (x_quorabust_variant or "a").lower().strip()
        if v == "b" and "b" not in state:
            v = "a"
        if v not in state:
            raise HTTPException(status_code=503, detail="requested variant not available")
        bld, clf, _meta = state[v]
        effective_threshold = (
            threshold
            if threshold is not None
            else _model_decision_threshold(_meta, default_threshold)
        )
        if len(body.question1) != len(body.question2):
            raise HTTPException(status_code=400, detail="question1 and question2 length mismatch")
        if len(body.question1) > configured_max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"batch size exceeds configured maximum of {configured_max_batch_size}",
            )
        t0 = time.perf_counter()
        try:
            proba = predict_proba_duplicate(bld, clf, body.question1, body.question2)[:, 1]
        finally:
            latency.labels(v).observe(time.perf_counter() - t0)
        predictions.labels(v).inc()
        features = None
        if explain:
            schema = _meta.get("feature_schema")
            features = explain_pair_features(
                bld,
                body.question1,
                body.question2,
                feature_schema=schema if isinstance(schema, list) else None,
            )
        probs = [float(x) for x in proba]
        return PredictOut(
            proba_duplicate=probs,
            is_duplicate=[p >= effective_threshold for p in probs],
            decision_threshold=effective_threshold,
            variant=v,
            features=features,
        )

    return app


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="quorabust-serve")
    ap.add_argument("--host", default=os.environ.get("QUORABUST_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("QUORABUST_PORT", "8000")))
    ap.add_argument(
        "--model",
        default=os.environ.get("QUORABUST_MODEL_PATH", ""),
        help="Primary artifact (.pkl); or set QUORABUST_MODEL_PATH",
    )
    ap.add_argument(
        "--model-b",
        default=os.environ.get("QUORABUST_MODEL_B", ""),
        help="Optional second artifact for A/B; or set QUORABUST_MODEL_B",
    )
    ns = ap.parse_args()
    app = create_app(ns.model or None, ns.model_b or None)
    uvicorn.run(app, host=ns.host, port=ns.port)
