# OpenTelemetry traces

Tracing is optional. The default install keeps the current Prometheus behavior and does not
import or export OpenTelemetry data. Install the extra when the deployment owns a configured
OpenTelemetry provider:

```bash
pip install -e ".[observability]"
```

Quorabust uses the global OpenTelemetry Python provider, so a platform bootstrap or vendor
distribution can configure the exporter and collector without putting backend-specific code
in the application. For a small OTLP HTTP setup, install the exporter and configure the
provider in the deployment bootstrap:

```bash
pip install "opentelemetry-exporter-otlp-proto-http>=1.28,<2"
export OTEL_SERVICE_NAME=quorabust
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

The provider setup belongs to the deployment because exporters, batching, sampling, and
credentials vary by platform. The OpenTelemetry Python SDK documentation shows the provider
and OTLP exporter wiring: <https://opentelemetry.io/docs/languages/python/exporters/>.

## Span contract

- HTTP server spans use the method, scheme, low-cardinality route, response status, and the
  existing canonical request ID. Incoming W3C trace context is extracted when available.
- Model-load spans record only variant, whether an artifact digest was pinned, and the feature
  backend.
- Feature-build and explanation spans record variant and bounded batch/output counts.
- Retrieval and rerank spans record the retriever/model name, catalog size, k, and candidate
  counts.
- Raised exceptions record an exception event, `error.type`, and error status. HTTP 5xx spans
  are marked errors; normal HTTP 4xx responses are left unset per the server semantic
  conventions.

Question text, raw model inputs, API keys, authorization headers, and arbitrary request paths
are not span attributes. The HTTP route template is used instead of a user-controlled path.

To disable tracing even when the optional packages are installed, set:

```bash
export OTEL_SDK_DISABLED=true
```

The manual integration test runs in the `Observability smoke` workflow, which is intentionally
separate from the minimal three-version CI matrix.
