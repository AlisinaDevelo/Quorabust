# Changelog

## Unreleased

### Added
- `quorabust-report` for generating Markdown model cards from saved artifacts and
  optional labeled holdout CSVs.
- `GET /models` serving endpoint for safe loaded-model metadata.
- `examples/smoke_pairs.csv` and CI coverage for the train-to-report workflow.
- JSON output for `quorabust-report --format json`.
- Precision, recall, F1, and configurable threshold sweeps in model reports.
- `POST /predict?explain=true` feature-value explanations for scored pairs.
- `quorabust-train --metadata-out` for JSON metadata sidecars.

### Fixed
- Capped NumPy below 2.5 so Mypy can parse dependency stubs with the configured
  Python target.

## 0.3.2

### Added
- Importable Grafana dashboard `grafana/dashboards/quorabust-serving.json` and [docs/GRAFANA.md](docs/GRAFANA.md).

## 0.3.1

### Added
- OpenAPI examples, descriptions, and tags for scoring vs operations endpoints.
- `loadtests/k6_predict.js` and [docs/LOAD_TESTING.md](docs/LOAD_TESTING.md) for SLO-oriented load checks.

## 0.3.0

### Added
- Chunked CSV reader (`dataio`), JSONL model registry, drift mean-shift helpers.
- Optional **embedding** backend (`PairEmbeddingBuilder`, `[nlp]` extra) and `quorabust-train --feature-backend`.
- **FastAPI** serving (`quorabust-serve`): health/ready, predict, Prometheus `/metrics`, A/B header routing.
- Training `meta`: `feature_schema`, `reference_feature_means`; optional `--registry-dir`.
- Docs: [docs/SCALING.md](docs/SCALING.md); CI runs `quorabust-serve --help`.

## 0.2.0

### Added
- `SECURITY.md`, `NOTICE`, `CODEOWNERS`, Dependabot, pip-audit workflow, pre-commit (Ruff + Mypy).
- PEP 621 packaging (`pyproject.toml` only), Mypy in CI, coverage floor 70%, training lineage in CLI meta.
- `eval_classification_metrics`, `lineage` helpers, Docker image and build workflow.
- `CONTRIBUTING.md`, `docs/ENTERPRISE.md`.

## 0.1.0

Initial published layout: `quorabust` package, tests, training CLI, documentation.
