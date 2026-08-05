from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _otel_modules() -> tuple[Any, ...] | None:
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        return None
    try:
        from opentelemetry import context, propagate, trace
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        return None
    return context, propagate, trace, SpanKind, Status, StatusCode


def tracing_available() -> bool:
    """Return whether the optional OpenTelemetry API can create spans."""
    return _otel_modules() is not None


@contextmanager
def span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    kind: str = "internal",
) -> Iterator[Any | None]:
    """Create an optional span and record raised exceptions without leaking inputs."""
    modules = _otel_modules()
    if modules is None:
        yield None
        return

    _context, _propagate, trace, span_kind, _status, _status_code = modules
    kinds = {
        "internal": span_kind.INTERNAL,
        "server": span_kind.SERVER,
    }
    tracer = trace.get_tracer("quorabust")
    with tracer.start_as_current_span(
        name,
        kind=kinds.get(kind, span_kind.INTERNAL),
        attributes=dict(attributes or {}),
    ) as current:
        try:
            yield current
        except Exception as exc:
            record_error(current, exc)
            raise


@contextmanager
def extracted_trace_context(headers: Mapping[str, str]) -> Iterator[None]:
    """Attach incoming W3C trace context when the optional SDK is available."""
    modules = _otel_modules()
    if modules is None:
        yield
        return

    context, propagate, _trace, _span_kind, _status, _status_code = modules
    carrier = {str(key).lower(): str(value) for key, value in headers.items()}
    extracted = propagate.extract(carrier)
    token = context.attach(extracted)
    try:
        yield
    finally:
        context.detach(token)


def set_attributes(current: Any | None, attributes: Mapping[str, Any]) -> None:
    """Set a reviewed attribute allow-list on a live span."""
    if current is None:
        return
    for key, value in attributes.items():
        current.set_attribute(key, value)


def record_error(current: Any | None, exc: BaseException) -> None:
    """Record exception details using the OpenTelemetry error guidance."""
    if current is None:
        return
    modules = _otel_modules()
    if modules is None:
        return
    _context, _propagate, _trace, _span_kind, status, status_code = modules
    current.record_exception(exc)
    current.set_status(status(status_code.ERROR))
    current.set_attribute("error.type", type(exc).__name__)


def record_http_response(current: Any | None, *, route: str, status_code: int) -> None:
    """Add low-cardinality HTTP response data and mark server failures."""
    set_attributes(
        current,
        {
            "http.response.status_code": int(status_code),
            "http.route": route,
        },
    )
    if current is None or status_code < 500:
        return
    modules = _otel_modules()
    if modules is None:
        return
    _context, _propagate, _trace, _span_kind, status, status_code_type = modules
    current.set_status(status(status_code_type.ERROR))
    current.set_attribute("error.type", str(status_code))
