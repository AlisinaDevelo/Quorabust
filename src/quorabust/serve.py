from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from quorabust.model import predict_proba_duplicate
from quorabust.persist import load_classifier


class PredictBody(BaseModel):
    question1: list[str] = Field(..., min_length=1)
    question2: list[str] = Field(..., min_length=1)


class PredictOut(BaseModel):
    proba_duplicate: list[float]
    variant: str


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

    app = FastAPI(title="Quorabust", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if "a" not in state:
            raise HTTPException(status_code=503, detail="model a not loaded")
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        data = generate_latest(registry)
        return PlainTextResponse(
            data.decode("utf-8"),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/predict", response_model=PredictOut)
    def predict(
        body: PredictBody,
        x_quorabust_variant: str | None = Header(default=None, alias="X-Quorabust-Variant"),
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
