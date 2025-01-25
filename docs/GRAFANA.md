# Grafana dashboard

## Metrics

`quorabust-serve` exposes Prometheus text at **`/metrics`**. Relevant series:

- **`quorabust_predictions_total{variant="a"|"b"}`** — Counter incremented per successful `POST /predict`.
- **`quorabust_predict_latency_seconds_*{variant=...}`** — Histogram of handler latency (seconds).

Scrape the service with Prometheus (job pointing at `http://<host>:8000/metrics`), then attach Grafana to that Prometheus data source.

## Import

1. Grafana → **Dashboards** → **New** → **Import**.
2. Upload **`grafana/dashboards/quorabust-serving.json`** (or paste its contents).
3. When prompted, pick your **Prometheus** data source for **`DS_PROMETHEUS`**.

Panels cover request rate by variant, p50/p95 latency, total predictions, and a single-stat p95 for SLO spot-checks. Adjust time range and thresholds to match your environment.

## Notes

- Histogram buckets are the Prometheus client defaults; for stricter SLOs, configure custom buckets in code and re-export the dashboard queries if names change.
- If only variant **a** is deployed, graphs still work; variant **b** series appear when `QUORABUST_MODEL_B` is configured.
