import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from starlette.testclient import TestClient

from quorabust.retrieval import CatalogQuestion, TfidfCatalogRetriever, rerank_candidates
from quorabust.serve import create_app
from quorabust.tracing import _otel_modules, span


def test_optional_spans_capture_safe_http_retrieval_and_errors():
    _otel_modules.cache_clear()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    retriever = TfidfCatalogRetriever().fit(
        [CatalogQuestion("q1", "How do I learn Python?"), CatalogQuestion("q2", "Other")]
    )
    candidates = retriever.search("learn Python", k=2)
    rerank_candidates("learn Python", candidates, lambda _q1, _q2: [0.9, 0.1])

    with TestClient(create_app()) as client:
        response = client.get(
            "/health",
            headers={
                "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "X-Request-ID": "123e4567-e89b-42d3-a456-426614174000",
            },
        )

    with pytest.raises(RuntimeError, match="trace failure"):
        with span("quorabust.test.error"):
            raise RuntimeError("trace failure")

    spans = exporter.get_finished_spans()
    by_name = {item.name: item for item in spans}
    http_span = by_name["GET /health"]
    assert response.status_code == 200
    assert http_span.attributes["http.request.method"] == "GET"
    assert http_span.attributes["http.route"] == "/health"
    assert http_span.attributes["http.response.status_code"] == 200
    assert http_span.attributes["quorabust.request_id"] == "123e4567-e89b-42d3-a456-426614174000"
    assert http_span.parent is not None
    assert http_span.parent.trace_id == 0x4BF92F3577B34DA6A3CE929D0E0E4736
    assert all("question" not in key for key in http_span.attributes)
    assert "quorabust.retrieval" in by_name
    assert "quorabust.rerank" in by_name
    error_span = by_name["quorabust.test.error"]
    assert error_span.status.status_code.name == "ERROR"
    assert any(event.name == "exception" for event in error_span.events)
