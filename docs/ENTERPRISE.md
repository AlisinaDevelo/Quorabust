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
when an eval split exists, and metric fields into the pickle `meta` dict. A calibrated
artifact additionally records its calibration method, calibration and threshold-data
hashes, base-artifact hash, calibration diagnostics, and threshold-selection policy. Treat `.pkl`
files as **trusted** (pickle); load only from controlled storage.

When source data includes complete `qid1`/`qid2` columns, the default holdout is split by
question connected components so a question cannot cross the train/eval boundary. The
chosen strategy is persisted in metadata for review.

Before training, run `quorabust-audit-data --csv ... --out ... --require-question-ids` to create a path-light
dataset preflight manifest. It records the source CSV hash, schema and binary-label
checks, empty or repeated pair warnings, and whether complete question IDs are available.
The command fails for missing required columns, an empty dataset, non-binary labels, or
missing/incomplete IDs when the strict benchmark flag is present. Without that flag, missing
IDs remain an explicit warning because the trainer supports a documented row-level fallback.

For benchmark runs, pass `--eval-out` to `quorabust-train` and retain the exported holdout
CSV with its metadata sidecar. The exact holdout hash is also copied into the lightweight
registry record for release review.
Use `--require-question-ids` on public benchmark runs to fail closed if leakage-safe
question IDs are missing or incomplete; customer-domain runs can explicitly use the
documented fallback when those IDs do not exist.

Use `quorabust-train --metadata-out models/quorabust.meta.json` to write the same
lineage and metric metadata as JSON. Reviewers and release tooling can inspect that
sidecar without loading executable pickle content. The sidecar also records the
post-save `artifact_sha256`, and `--registry-dir` records the same digest in
`models.jsonl`. The sidecar is not a replacement for the model artifact; it is a safer
inspection path.

For serving, set `QUORABUST_MODEL_SHA256` and optionally `QUORABUST_MODEL_B_SHA256` to
pin the exact artifact bytes. Quorabust verifies the digest before loading and fails startup
on a mismatch. This is an integrity check, not a cryptographic signature; keep the expected
digest in trusted deployment configuration. For the current TF-IDF/XGBoost control model,
`quorabust-export-safe` writes a non-pickle `.qmodel` bundle with explicit TF-IDF state and
the XGBoost native JSON model. The pickle path remains the compatibility path for optional
builders and calibrated wrappers; see [ARTIFACTS.md](ARTIFACTS.md) for the threat model and
per-backend policy.

## Serving and SLOs

- **`quorabust-serve`**: FastAPI with `/health`, `/ready`, `/predict`, `/metrics` (Prometheus). Configure **`QUORABUST_MODEL_PATH`** and optional **`QUORABUST_MODEL_B`** for A/B; clients may send **`X-Quorabust-Variant: b`**.
- **Decisioning**: `/predict` returns both `proba_duplicate` and thresholded
  `is_duplicate`. Clients can pass `?threshold=0.7`; otherwise serving uses the
  holdout-selected artifact metadata `decision_threshold`, then
  `QUORABUST_DECISION_THRESHOLD`, then `0.5`.
- **Optional access controls**: set `QUORABUST_API_KEY` to require
  `X-Quorabust-API-Key` on `/predict` and `/models`. Set
  `QUORABUST_MAX_BATCH_SIZE` to bound scoring work; the default is 256 pairs. Health,
  readiness, and metrics remain available for platform probes and should be network-scoped.
  Set `QUORABUST_MAX_TEXT_LENGTH` to bound each question string before feature construction;
  the default is 8192 characters and both sides of every pair are checked.
- **Artifact integrity**: pin `QUORABUST_MODEL_SHA256` and
  `QUORABUST_MODEL_B_SHA256` when promoting artifacts; mismatches fail before pickle
  loading.
- **Loaded artifact identity**: `GET /models` exposes the computed SHA-256 for each loaded
  variant, allowing deployment and support checks to verify the serving bytes without
  exposing local paths.
- **Request correlation**: every completed response includes a canonical `X-Request-ID`
  UUID. A valid client-provided UUID is reused; malformed values are replaced. The
  `quorabust.http` logger emits one-line JSON events with method, path, status, duration,
  and request ID only; question text, headers, tokens, and secrets are not logged.
- **HTTP RED metrics**: `quorabust_http_requests_total` records method, matched route,
  and status code; `quorabust_http_request_duration_seconds` records endpoint latency.
  Unknown paths use a fixed `<unmatched>` label to prevent metric-cardinality abuse.
- **Optional traces**: install the `observability` extra and configure a global OpenTelemetry
  provider to emit privacy-reviewed HTTP, model-load, feature, retrieval, and rerank spans.
  See [OBSERVABILITY.md](OBSERVABILITY.md); Prometheus behavior is unchanged by default.
- **Error contract**: client errors return `error.code`, a safe message, and the same
  request ID as the `X-Request-ID` header. Clients can branch on stable codes rather than
  framework-specific validation prose.
- **Local demo**: `compose.yaml` runs the API, Prometheus, and Grafana together. See
  [DEMO.md](DEMO.md).
- **Hosted demo**: expose only the smoke model and label it clearly as non-production.
  See [HOSTED_DEMO.md](HOSTED_DEMO.md).
- Wire ingress timeouts and autoscaling to your **latency** SLO using the histogram in `/metrics`. See [LOAD_TESTING.md](LOAD_TESTING.md) for k6 and [GRAFANA.md](GRAFANA.md) for a starter dashboard JSON.

## Scale and NLP

See [SCALING.md](SCALING.md) for chunked CSV I/O, optional **embedding** training
(`pip install ".[nlp]"`, `quorabust-train --feature-backend embedding`), optional
**cross-encoder** pair scoring (`quorabust-train --feature-backend cross-encoder`), and
pointers to distributed XGBoost. Use `quorabust-retrieve-profile` there for fresh-process
startup and artifact-size evidence before setting deployment budgets.

## Registry and drift (lightweight)

- **`quorabust.registry`**: append JSONL rows with `--registry-dir` after training; swap for MLflow when you need a UI.
- **`quorabust.drift`**: compare live batch feature means to `meta["reference_feature_means"]`.
- **`quorabust-validate-report`**: validate JSON model-card reports in CI/release jobs; use
  `--require-holdout --require-calibration --require-manifest` before promoting a
  benchmarked artifact. The manifest binds the report to model and holdout hashes,
  training lineage, runtime, and the exact evaluation invocation.
- **`quorabust-validate-retrieval`** applies the same fail-closed discipline to
  retrieval/profile evidence and caller-supplied recall/p95 policies.
- Put TLS termination, distributed rate limiting, quotas, and key rotation at the ingress
  or gateway. The application key and batch cap are defense-in-depth controls, not a
  replacement for a multi-worker gateway policy. Ship the JSON request events to the
  platform log system and propagate the request ID into distributed traces there.

## Releases

Tag versions, update `CHANGELOG.md`, and align `[project].version` in `pyproject.toml`. Publish to an internal index or PyPI as appropriate.
