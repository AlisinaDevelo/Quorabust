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

Training writes `csv_sha256`, `git_revision`, `quorabust_version`, `feature_schema`, `reference_feature_means` (for drift checks), and metric fields into the pickle `meta` dict. Treat `.pkl` files as **trusted** (pickle); load only from controlled storage.

## Serving and SLOs

- **`quorabust-serve`**: FastAPI with `/health`, `/ready`, `/predict`, `/metrics` (Prometheus). Configure **`QUORABUST_MODEL_PATH`** and optional **`QUORABUST_MODEL_B`** for A/B; clients may send **`X-Quorabust-Variant: b`**.
- Wire ingress timeouts and autoscaling to your **latency** SLO using the histogram in `/metrics`.

## Scale and NLP

See [SCALING.md](SCALING.md) for chunked CSV I/O, optional **embedding** training (`pip install ".[nlp]"`, `quorabust-train --feature-backend embedding`), and pointers to distributed XGBoost.

## Registry and drift (lightweight)

- **`quorabust.registry`**: append JSONL rows with `--registry-dir` after training; swap for MLflow when you need a UI.
- **`quorabust.drift`**: compare live batch feature means to `meta["reference_feature_means"]`.

## Releases

Tag versions, update `CHANGELOG.md`, and align `[project].version` in `pyproject.toml`. Publish to an internal index or PyPI as appropriate.
