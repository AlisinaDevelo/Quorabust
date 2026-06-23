from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from quorabust.model import predict_proba_duplicate
from quorabust.persist import load_classifier


def _dist_version() -> str:
    try:
        return version("Quorabust")
    except PackageNotFoundError:
        return "0.0.0"


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
                    "variant": "a",
                }
            ]
        }
    )

    proba_duplicate: list[float] = Field(
        ...,
        description="P(duplicate) for each pair, same order as the request lists.",
    )
    variant: str = Field(..., description="Scoring variant (a or b) after A/B fallback rules.")


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
) -> FastAPI:
    path_a = model_path_a or os.environ.get("QUORABUST_MODEL_PATH", "")
    path_b = model_path_b or os.environ.get("QUORABUST_MODEL_B", "")

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
        tags=["operations"],
        summary="List loaded model metadata",
        description=(
            "Returns allowlisted metadata for loaded variants. Local artifact paths and "
            "training CSV paths are intentionally omitted."
        ),
        responses={503: {"description": "No models loaded"}},
    )
    def models() -> dict[str, dict[str, dict[str, Any]]]:
        if not state:
            raise HTTPException(status_code=503, detail="no models loaded")
        variants = {
            name: _public_model_meta(meta)
            for name, (_, _, meta) in state.items()
        }
        return {"variants": variants}

    @app.get("/metrics", tags=["operations"])
    def metrics() -> PlainTextResponse:
        data = generate_latest(registry)
        return PlainTextResponse(
            data.decode("utf-8"),
            media_type="text/plain; version=0.0.4",
        )

    @app.post(
        "/predict",
        response_model=PredictOut,
        tags=["scoring"],
        summary="Predict duplicate probability",
        description=(
            "Scores one or more question pairs. Optional header "
            "`X-Quorabust-Variant: b` selects the B artifact when configured."
        ),
        responses={
            400: {"description": "question1 and question2 length mismatch"},
            503: {"description": "Model not loaded or variant unavailable"},
        },
    )
    def predict(
        body: PredictBody,
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
        if len(body.question1) != len(body.question2):
            raise HTTPException(status_code=400, detail="question1 and question2 length mismatch")
        t0 = time.perf_counter()
        try:
            proba = predict_proba_duplicate(bld, clf, body.question1, body.question2)[:, 1]
        finally:
            latency.labels(v).observe(time.perf_counter() - t0)
        predictions.labels(v).inc()
        return PredictOut(proba_duplicate=[float(x) for x in proba], variant=v)

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
