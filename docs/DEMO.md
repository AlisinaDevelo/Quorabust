# Local serving demo

Use this stack to inspect Quorabust as a small ML-backed service: FastAPI scoring,
Prometheus metrics, and Grafana dashboards.

The smoke model below is not a quality benchmark. It uses `examples/smoke_pairs.csv` only
to demonstrate the API and observability contract.

For a quick static inspection without starting Docker, see the generated JSON snapshots
in [demo-assets](demo-assets). Regenerate them with:

```bash
quorabust-demo-assets --out docs/demo-assets
quorabust-demo-assets --out docs/demo-assets --check
```

## Build a smoke model

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

mkdir -p models
quorabust-train \
  --csv examples/smoke_pairs.csv \
  --out models/quorabust-smoke.pkl \
  --metadata-out models/quorabust-smoke.meta.json \
  --eval-fraction 0 \
  --seed 7
```

## Start the stack

```bash
docker compose up --build
```

Open:

- FastAPI docs: http://localhost:8000/docs
- Readiness: http://localhost:8000/ready
- Model metadata: http://localhost:8000/models
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000/d/quorabust-serving/quorabust-serving

## Send a prediction

```bash
curl -s 'http://localhost:8000/predict?explain=true&threshold=0.9' \
  -H 'content-type: application/json' \
  -d '{
    "question1": ["How do I learn Python?", "How should I cache API responses?"],
    "question2": ["What is the best way to learn Python?", "Where can I buy train tickets?"]
  }' | python -m json.tool
```

The response includes `proba_duplicate`, thresholded `is_duplicate`,
`decision_threshold`, and optional feature values. The example uses `threshold=0.9` to
show both positive and negative decisions with the smoke model.

## Stop

```bash
docker compose down
```
