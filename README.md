# Quorabust

Production-minded **Quora-style duplicate question detection**: reproducible training,
artifact metadata, FastAPI serving, Prometheus metrics, drift helpers, A/B model routing,
load tests, and Markdown model-card reporting.

Quorabust is intentionally small enough to inspect quickly while still showing the
operational shape of an ML-backed backend service.

📐 **Architecture & diagrams:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — component
overview, training pipeline, train-vs-serve sequence, and the artifact/registry contract.

## What it demonstrates

- Pairwise text features with TF-IDF or optional sentence-transformer embeddings
- XGBoost training with optional holdout evaluation and early stopping
- Saved artifacts that include lineage, feature schema, dataset checksum, and metrics
- FastAPI inference with health/readiness checks, OpenAPI docs, Prometheus metrics, and A/B routing
- Drift helper utilities, JSONL model registry, k6 load test, and Grafana dashboard starter
- `quorabust-report` model-card generation for artifact review and benchmark summaries

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras: `pip install -e ".[viz,notebooks]"` for Matplotlib, Seaborn, and Jupyter; `".[nlp]"` for sentence-transformer features; `".[serve]"` matches the API stack (also included in `dev`).

## Usage

```python
import pandas as pd
from quorabust import (
    clean_text,
    load_classifier,
    predict_proba_duplicate,
    train_duplicate_classifier,
)

df = pd.read_csv("data/raw/train.csv")  # Kaggle Quora Question Pairs
df.columns = [c.strip() for c in df.columns]
builder, clf = train_duplicate_classifier(df.head(5000))

proba = predict_proba_duplicate(
    builder,
    clf,
    ["How do I learn Python?"],
    ["What is the best way to learn Python?"],
)
print(proba[:, 1])  # P(duplicate)
```

### Train from the terminal

With `data/raw/train.csv` in place:

```bash
quorabust-train --csv data/raw/train.csv --out models/quorabust.pkl
python -m quorabust --csv data/raw/train.csv --out models/quorabust.pkl   # equivalent
```

Options: `--max-rows N`, `--eval-fraction 0.1` (default), `--eval-fraction 0` to train on all rows without a holdout, `--seed`, `--feature-backend {tfidf,embedding}`, `--embedding-model …`, `--registry-dir` (JSONL registry), `--metadata-out` (JSON sidecar for reviewing artifact lineage without loading the pickle).

### Generate a model card

```bash
quorabust-report \
  --model models/quorabust.pkl \
  --eval-csv data/processed/holdout.csv \
  --out reports/quorabust-model-card.md
```

The report includes artifact metadata, persisted training/eval metrics, optional holdout
metrics, a confusion matrix, and a precision/recall/F1 threshold sweep. Use
`--format json` for machine-readable CI or release artifacts. Use a real held-out CSV for
comparable model claims; the command accepts the same `question1`, `question2`,
`is_duplicate` column contract as training.
See [docs/REPORTING.md](docs/REPORTING.md) for the CI smoke workflow and
real-evaluation checklist.

### Load a saved model

```python
from quorabust import load_classifier, predict_proba_duplicate

builder, clf, meta = load_classifier("models/quorabust.pkl")
print(meta)  # n_train, metrics, csv_sha256, reference_feature_means, …
```

### HTTP API (monitoring + A/B)

```bash
export QUORABUST_MODEL_PATH=models/quorabust.pkl
quorabust-serve --host 0.0.0.0 --port 8000
# optional second artifact: export QUORABUST_MODEL_B=models/other.pkl
```

`GET /metrics` exposes Prometheus text; `POST /predict` accepts `{"question1":[...],"question2":[...]}` and optional header `X-Quorabust-Variant: b`. Add `?explain=true` to return per-pair input feature values. Interactive docs: **`/docs`**. Load testing: [docs/LOAD_TESTING.md](docs/LOAD_TESTING.md). Grafana: [docs/GRAFANA.md](docs/GRAFANA.md).
`GET /models` returns allowlisted metadata for loaded variants without leaking local
artifact paths or training CSV paths.

## Project layout

| Path | Purpose |
|------|---------|
| `src/quorabust/` | Package: `preprocess`, `features`, `embedding_features`, `model`, `persist`, `cli`, `serve`, `dataio`, `registry`, `drift` |
| `tests/` | Pytest suite |
| `data/raw/` | Original CSVs (not committed; see `data/README.md`) |
| `data/processed/` | Cleaned splits |
| `models/` | Saved `.pkl` artifacts (gitignored) |
| `notebooks/` | Exploratory work (optional) |

## Development

```bash
ruff check src tests
mypy src/quorabust
pytest -q --cov=quorabust --cov-fail-under=70
pre-commit run --all-files   # optional
```

Design notes: [docs/NOTES.md](docs/NOTES.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## Enterprise / operations

Governance (security policy, Dependabot, audits), containers, serving, and release expectations are summarized in [docs/ENTERPRISE.md](docs/ENTERPRISE.md) and [docs/SCALING.md](docs/SCALING.md). Saved model pickles include `meta` (CSV checksum, git revision, package version, feature means for drift, metrics); only load trusted artifacts. Use `quorabust-train --metadata-out ...` when you only need lineage and metrics.

## License

MIT — see [LICENSE](LICENSE).
