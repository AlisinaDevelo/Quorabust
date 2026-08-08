# Quorabust

Production-minded **semantic duplicate detection service** for Quora-style question
pairs: reproducible training, artifact metadata, thresholded API decisions, FastAPI
serving, Prometheus metrics, drift helpers, A/B model routing, load tests, and Markdown
model-card reporting.

Quorabust is intentionally small enough to inspect quickly while still showing the
operational shape of an ML-backed backend service.

📐 **Architecture & diagrams:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — component
overview, training pipeline, train-vs-serve sequence, and the artifact/registry contract.
**Enterprise positioning:** [docs/PRODUCT.md](docs/PRODUCT.md) — product surface,
buyer-facing use cases, method strategy, and production gaps.
**Artifact trust policy:** [docs/ARTIFACTS.md](docs/ARTIFACTS.md) — safe `.qmodel` export,
pickle boundary, parity checks, and promotion threat model.
**Local demo stack:** [docs/DEMO.md](docs/DEMO.md) — FastAPI, Prometheus, and Grafana
via Docker Compose.
**API snapshots:** [docs/demo-assets](docs/demo-assets) — generated request/response and
OpenAPI excerpts for quick inspection without running the service.
**Hosted demo checklist:** [docs/HOSTED_DEMO.md](docs/HOSTED_DEMO.md) — safe public demo
requirements and portfolio capture notes.

## What it demonstrates

- Pairwise text features with TF-IDF, optional sentence-transformer embeddings, or
  optional cross-encoder pair scoring
- Leakage-safe lexical hard-negative mining for cross-encoder training experiments
- Leakage-aware question-component holdouts when `qid1`/`qid2` identifiers are available
- XGBoost training with optional holdout evaluation and early stopping
- Saved artifacts that include lineage, feature schema, dataset checksum, and metrics
- FastAPI inference with health/readiness checks, thresholded decisions, OpenAPI docs,
  Prometheus metrics, A/B routing, request IDs, and structured HTTP events
- Drift helper utilities, JSONL model registry, k6 load test, and Grafana dashboard starter
- `quorabust-report` model-card generation for artifact review and benchmark summaries

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras: `pip install -e ".[viz,notebooks]"` for Matplotlib, Seaborn, and Jupyter;
`".[nlp]"` for sentence-transformer features; `".[observability]"` for optional
OpenTelemetry spans; `".[serve]"` matches the API stack (also included in `dev`). See
[docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) for collector setup and the privacy contract.

### Audit data before training

Run the dataset preflight before training or comparing artifacts:

```bash
quorabust-audit-data \
  --csv data/raw/train.csv \
  --out reports/train-data-audit.json
```

The JSON manifest records the input SHA-256, schema and label checks, empty or repeated
pair signals, and whether complete `qid1`/`qid2` columns are available for leakage-aware
splitting. It exits non-zero for missing required columns, an empty dataset, or labels
outside `{0, 1}`. Missing question IDs and duplicate pairs are reported as warnings, not
silently treated as benchmark evidence.

Before publishing a real-data comparison, freeze the source, split roles, threshold policy,
and runtime provenance in a protocol manifest, then validate it with
`quorabust-validate-protocol --protocol reports/quorabust-benchmark-protocol.json`.
The contract and real-data handoff are documented in
[docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md); the repository smoke fixture is
explicitly not benchmark evidence.

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

Options: `--max-rows N`, `--eval-fraction 0.1` (default), `--eval-fraction 0` to train on all rows without a holdout, `--eval-out` to export the exact holdout used by training, `--require-question-ids` to fail if leakage-safe IDs are absent, `--seed`, `--feature-backend {tfidf,embedding,cross-encoder}`, `--embedding-model …`, `--cross-encoder-model …`, `--thresholds`, `--threshold-metric {accuracy,precision,recall,f1}` for holdout-based decision-threshold selection, `--registry-dir` (JSONL registry), `--metadata-out` (JSON sidecar for reviewing artifact lineage without loading the pickle).

To generate lexical hard negatives for training or tuning experiments, use only the
permitted training/tuning material and keep the calibration and final holdout files out:

```bash
quorabust-mine-hard-negatives \
  --csv data/processed/train.csv \
  --out data/derived/train-hard-negatives.csv \
  --metadata-out data/derived/train-hard-negatives.meta.json \
  --candidate-k 50 \
  --negatives-per-positive 2 \
  --seed 42
```

The command requires complete question IDs, builds the known-positive graph from label-1
edges, and excludes every question in an anchor's positive component. It emits label-0
pairs with retrieval rank and score, plus a sidecar containing input/output hashes,
configuration, and runtime provenance. These are candidate training examples, not a
quality benchmark or evidence that the model improved; validate them on the frozen protocol
before making a model-card claim.

The default `--retriever tfidf` is the deterministic, dependency-free control. For a
semantic candidate-generation experiment, install `.[nlp]` and select
`--retriever embedding --embedding-model sentence-transformers/all-MiniLM-L6-v2`; the
sidecar records the selected model name. Dense retrieval may produce a different candidate
set, but it is still a hypothesis until measured on the frozen real-data protocol.

To calibrate a trained artifact, keep the calibration and threshold-selection CSVs
independent from training and from each other:

```bash
quorabust-calibrate \
  --model models/quorabust.pkl \
  --calibration-csv data/processed/calibration.csv \
  --threshold-csv data/processed/threshold.csv \
  --calibration-method sigmoid \
  --out models/quorabust-calibrated.pkl \
  --metadata-out models/quorabust-calibrated.meta.json
```

The command stores calibration and threshold-data hashes, calibration diagnostics, and the
selected calibrated decision threshold. Keep a final untouched holdout for the model card.

For a catalog lookup, the lexical first stage is available without the optional NLP extra:

```bash
quorabust-retrieve \
  --catalog-csv data/catalog/questions.csv \
  --query "How do I cache API responses?" \
  --query "Where is the cache configured?" \
  --k 10 \
  --out reports/retrieval.json
```

The JSON result contains stable question IDs and retrieval scores. With the optional NLP
extra, `--retriever embedding` adds dense retrieval and `--reranker-model` applies a
cross-encoder only to the bounded candidate set. Those reranker scores are ranking signals,
not calibrated duplicate probabilities.

Benchmark a catalog against a qrels-style CSV (`query,question_id,relevance`) with the same
control or optional NLP stages:

```bash
quorabust-retrieve-benchmark \
  --catalog-csv data/catalog/questions.csv \
  --qrels-csv data/processed/retrieval-qrels.csv \
  --ks 1,5,10 \
  --candidate-k 50 \
  --out reports/retrieval-benchmark.json
```

The JSON report separates first-stage and final recall/MRR/NDCG, retrieval/rerank/end-to-end
p50/p95/p99 latency, bounded reranker work, source hashes, model names, runtime, and the exact
command. It also reports populated latency strata for short (1-5), medium (6-15), and long
(16+) whitespace-token queries. The checked-in `examples/retrieval_*.csv` files are structural
smoke fixtures, not real quality evidence; use a permitted frozen dataset before publishing
model claims.

For process-level rollout sizing, `quorabust-retrieve-profile` repeats the benchmark in fresh
subprocesses and records process-to-report startup time, warm metrics, input sizes, and optional
local artifact hashes. It is an operational profile, not model-quality evidence.

Use `quorabust-validate-retrieval` in release jobs to validate report provenance, measurement
counts, latency summaries, and query-length strata, then apply caller-owned policies such as
`--min-final-recall-at-k 10=0.95`, `--max-end-to-end-p95-ms 100`,
`--max-cold-start-p95-ms`, `--max-peak-rss-bytes`, and
`--max-total-artifact-bytes`. Cold-start and artifact-size policies require a profile, and the
artifact-size policy requires at least one profiled `--artifact`. The gate compares supplied
evidence; it does not turn the smoke fixture into a public quality claim or define universal
quality, latency, memory, or packaging targets.

### Generate a model card

```bash
quorabust-report \
  --model models/quorabust.pkl \
  --eval-csv data/processed/holdout.csv \
  --out reports/quorabust-model-card.md
```

The report includes artifact metadata, persisted training/eval metrics, optional holdout
metrics, a confusion matrix, a precision/recall/F1 threshold sweep, and probability
calibration diagnostics. Use `--calibration-bins` to tune the calibration table and
`--format json` for machine-readable CI or release artifacts. Use a real held-out CSV for
comparable model claims; the command accepts the same `question1`, `question2`,
`is_duplicate` column contract as training.
Reports with `--eval-csv` also carry an evaluation manifest with dataset/artifact hashes,
label counts, threshold policy, runtime, commit, and exact invocation. Use
`--manifest-out` for a separate sidecar. Use
`quorabust-validate-report --require-holdout --require-calibration --require-manifest`
to fail release jobs when a JSON model card is missing benchmark evidence or its audit
record. Add `--require-question-component-split` for a public benchmark release that must
prove the evaluation holdout used complete question IDs and a leakage-safe component split.
For a protocol-bound release, add
`--protocol reports/quorabust-benchmark-protocol.json`; this also requires the holdout,
calibration, manifest, and component-split checks and binds the report to the declared
source/final-holdout hashes, seed, threshold grid, and metric.
Use repeated `--compare-model label=path` flags to compare TF-IDF, embedding, and
cross-encoder artifacts against the same holdout split.
For caller-supplied evaluation slices, repeat `--slice-column` (for example,
`--slice-column language --slice-column domain`). The report emits bounded per-label
counts, rates, threshold metrics, log loss, and calibration diagnostics in JSON or
Markdown, and records the requested columns in the evaluation manifest. Slice labels are
provided by the dataset owner; Quorabust does not infer language or domain membership and
slice output is not a quality claim without permitted, representative data.
For a benchmark release, add `--slice-manifest reports/holdout-slices.json` to bind those
columns to the exact evaluated CSV hash, source reference, row count, and labeling method;
the report then records observed per-label row counts. The sidecar schema is documented in
[REPORTING.md](docs/REPORTING.md), and synthetic CI coverage is contract-only.
Slice rows also include 95% Wilson intervals for rate metrics; undefined rates are null,
and log loss, F1, ROC-AUC, Brier score, and ECE remain point estimates unless a permitted
resampling protocol is supplied.
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
# optional integrity pin: use the digest recorded in the metadata sidecar/registry
export QUORABUST_MODEL_SHA256=<64-hex-artifact-sha256>
quorabust-serve --host 0.0.0.0 --port 8000
# optional second artifact: export QUORABUST_MODEL_B=models/other.pkl
# optional A/B integrity pin: export QUORABUST_MODEL_B_SHA256=<64-hex-artifact-sha256>
```

`GET /metrics` exposes Prometheus text; `POST /predict` accepts `{"question1":[...],"question2":[...]}` and optional header `X-Quorabust-Variant: b`. Responses include `proba_duplicate`, thresholded `is_duplicate`, and `decision_threshold`. Every completed response includes an `X-Request-ID` UUID; clients may provide a canonical UUID to correlate retries and support cases. The `quorabust.http` logger emits one-line JSON request events with method, path, status, duration, and request ID, without question text, headers, or secrets. Add `?threshold=0.7` to override the duplicate cutoff for one request; otherwise serving uses the holdout-selected artifact `decision_threshold` when present, then `QUORABUST_DECISION_THRESHOLD`, then `0.5`. Add `?explain=true` to return per-pair input feature values. Interactive docs: **`/docs`**. Local demo: [docs/DEMO.md](docs/DEMO.md). API snapshots: [docs/demo-assets](docs/demo-assets). Load testing: [docs/LOAD_TESTING.md](docs/LOAD_TESTING.md). Grafana: [docs/GRAFANA.md](docs/GRAFANA.md).
The same endpoint exposes `quorabust_http_requests_total{method,path,status_code}` and
`quorabust_http_request_duration_seconds_*` for RED dashboards. Labels use matched route
templates, with unknown paths grouped under `<unmatched>` so arbitrary URLs cannot create
unbounded Prometheus series.
`GET /models` returns allowlisted metadata for loaded variants without leaking local
artifact paths or training CSV paths. It includes the actual `artifact_sha256` of each
loaded artifact so deployment checks can confirm which immutable bytes are serving.
Client errors use one stable envelope: `{"error":{"code":"...","message":"...","request_id":"..."}}`.
Codes include `invalid_request`, `validation_error`, `unauthorized`, `batch_too_large`,
`request_too_large`, `model_unavailable`, and `not_found`; the `request_id` matches the
`X-Request-ID` response header.
For a deployment-level boundary, set `QUORABUST_API_KEY` to require the
`X-Quorabust-API-Key` header on `/predict` and `/models`, and use
`QUORABUST_MAX_BATCH_SIZE` to bound request work (default 256), and
`QUORABUST_MAX_TEXT_LENGTH` to reject an individual question that is too large (default 8192
characters). The text limit applies to both sides of every pair and is checked before feature
construction; the API does not log or echo rejected text. Set `QUORABUST_MAX_REQUEST_BYTES` to
bound the raw HTTP body before JSON/Pydantic parsing (default 8 MiB). The byte limit is an
application boundary; keep TLS, rate limiting, quotas, and key rotation at the gateway.
For artifact integrity, set `QUORABUST_MODEL_SHA256` and optionally
`QUORABUST_MODEL_B_SHA256`; serving verifies each digest before unpickling and fails
startup on a mismatch. `quorabust-train --metadata-out ...` records the post-save
`artifact_sha256` in the sidecar, and `--registry-dir` records it in `models.jsonl`.

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

Design notes: [docs/NOTES.md](docs/NOTES.md). Enterprise positioning:
[docs/PRODUCT.md](docs/PRODUCT.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## Enterprise / operations

Governance (security policy, Dependabot, audits), containers, serving, and release expectations are summarized in [docs/ENTERPRISE.md](docs/ENTERPRISE.md) and [docs/SCALING.md](docs/SCALING.md). Saved model pickles include `meta` (CSV checksum, git revision, package version, feature means for drift, metrics); only load trusted artifacts. Use `quorabust-train --metadata-out ...` when you only need lineage and metrics.

## License

MIT — see [LICENSE](LICENSE).
