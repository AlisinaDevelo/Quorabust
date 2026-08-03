# Grafana dashboard

## Metrics

`quorabust-serve` exposes Prometheus text at **`/metrics`**. Relevant series:

- **`quorabust_predictions_total{variant="a"|"b"}`** — Counter incremented per successful `POST /predict`.
- **`quorabust_predict_latency_seconds_*{variant=...}`** — Histogram of handler latency (seconds).
- **`quorabust_http_requests_total{method=...,path=...,status_code=...}`** — Counter for every
  completed request, including handled 4xx responses.
- **`quorabust_http_request_duration_seconds_*{method=...,path=...}`** — Histogram for
  endpoint latency and p50/p95 calculations.

HTTP metric labels use matched route templates. Unknown paths share the `<unmatched>` label
so user-controlled URLs cannot create unbounded time series.

Scrape the service with Prometheus (job pointing at `http://<host>:8000/metrics`), then attach Grafana to that Prometheus data source.

## Import

The local Compose demo in [DEMO.md](DEMO.md) provisions Prometheus and this dashboard
automatically.

For manual import:

1. Grafana → **Dashboards** → **New** → **Import**.
2. Upload **`grafana/dashboards/quorabust-serving.json`** (or paste its contents).
3. The dashboard expects a Prometheus data source with UID **`prometheus`**. Either
   create the data source with that UID or replace `prometheus` in the JSON with your
   existing Prometheus data source UID before importing.

Panels cover prediction rate by variant, prediction latency, total predictions, HTTP request
rate by route, HTTP p95 latency by route, and 4xx/5xx rate. Adjust time range and thresholds
to match your environment.

## Notes

- Histogram buckets are the Prometheus client defaults; for stricter SLOs, configure custom buckets in code and re-export the dashboard queries if names change.
- If only variant **a** is deployed, graphs still work; variant **b** series appear when `QUORABUST_MODEL_B` is configured.
