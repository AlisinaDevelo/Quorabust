# Enterprise operations

## Supply chain

- **Dependabot** updates GitHub Actions and pip dependencies weekly (see `.github/dependabot.yml`).
- **Audit** workflow runs `pip-audit` on pushes, PRs, and weekly; install pins `pip>=26` before auditing.
- **Pre-commit** (optional locally): Ruff and Mypy on `src/quorabust` (see `.pre-commit-config.yaml`).

## Builds

- **Python package**: PEP 621 metadata in `pyproject.toml`; install with `pip install .` or `pip install -e ".[dev]"`.
- **Container**: `docker build -t quorabust .` then mount data and pass CLI flags, e.g.  
  `docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/models:/models" quorabust --csv /data/raw/train.csv --out /models/model.pkl`

## Lineage and artifacts

Training writes `csv_sha256`, `git_revision`, `quorabust_version`, `feature_schema`,
`reference_feature_means` (for drift checks), holdout-selected `decision_threshold`
when an eval split exists, and metric fields into the pickle `meta` dict. Treat `.pkl`
files as **trusted** (pickle); load only from controlled storage.

Use `quorabust-train --metadata-out models/quorabust.meta.json` to write the same
lineage and metric metadata as JSON. Reviewers and release tooling can inspect that
sidecar without loading executable pickle content. The sidecar is not a replacement for
the model artifact; it is a safer inspection path. For untrusted artifact distribution,
prefer a non-pickle format such as `skops` or ONNX in a future release.

## Serving and SLOs

- **`quorabust-serve`**: FastAPI with `/health`, `/ready`, `/predict`, `/metrics` (Prometheus). Configure **`QUORABUST_MODEL_PATH`** and optional **`QUORABUST_MODEL_B`** for A/B; clients may send **`X-Quorabust-Variant: b`**.
- **Decisioning**: `/predict` returns both `proba_duplicate` and thresholded
  `is_duplicate`. Clients can pass `?threshold=0.7`; otherwise serving uses the
  holdout-selected artifact metadata `decision_threshold`, then
  `QUORABUST_DECISION_THRESHOLD`, then `0.5`.
- Wire ingress timeouts and autoscaling to your **latency** SLO using the histogram in `/metrics`. See [LOAD_TESTING.md](LOAD_TESTING.md) for k6 and [GRAFANA.md](GRAFANA.md) for a starter dashboard JSON.

## Scale and NLP

See [SCALING.md](SCALING.md) for chunked CSV I/O, optional **embedding** training (`pip install ".[nlp]"`, `quorabust-train --feature-backend embedding`), and pointers to distributed XGBoost.

## Registry and drift (lightweight)

- **`quorabust.registry`**: append JSONL rows with `--registry-dir` after training; swap for MLflow when you need a UI.
- **`quorabust.drift`**: compare live batch feature means to `meta["reference_feature_means"]`.
- **`quorabust-validate-report`**: validate JSON model-card reports in CI/release jobs;
  use `--require-holdout --require-calibration` before promoting a benchmarked artifact.

## Releases

Tag versions, update `CHANGELOG.md`, and align `[project].version` in `pyproject.toml`. Publish to an internal index or PyPI as appropriate.
