# Load testing and SLO hints

## OpenAPI

With the API running, open **`/docs`** (Swagger UI) or fetch **`/openapi.json`**. The `POST /predict` body includes **documented examples** for batch scoring.

## k6

1. Install [k6](https://k6.io/docs/get-started/installation/).
2. Train a model and start the server with `QUORABUST_MODEL_PATH` set.
3. Run:

```bash
export QUORABUST_MODEL_PATH=models/quorabust.pkl
quorabust-serve --port 8000 &
k6 run -e BASE_URL=http://127.0.0.1:8000 loadtests/k6_predict.js
```

Edit **`loadtests/k6_predict.js`** `thresholds` to match your SLO (e.g. stricter `p(95)` latency or lower error rate). Pair k6 results with **`GET /metrics`** (Prometheus histogram `quorabust_predict_latency_seconds`) in Grafana or similar.
