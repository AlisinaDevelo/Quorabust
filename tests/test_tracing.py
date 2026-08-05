from quorabust.tracing import extracted_trace_context, span, tracing_available


def test_tracing_helpers_are_safe_without_configuration():
    assert isinstance(tracing_available(), bool)
    with extracted_trace_context({"traceparent": "invalid"}):
        with span("quorabust.test", attributes={"quorabust.test": True}) as current:
            assert current is None or hasattr(current, "set_attribute")
