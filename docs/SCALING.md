# Scale, NLP, serving, and MLOps

This document maps **ambitions** to what ships in-repo and what stays external.

## Big data and distributed training

- **In-repo**: `quorabust.dataio.iter_csv_batches` streams CSV chunks with pandas (`chunksize`) so you can build custom loops (sample, aggregate stats, incremental vocab experiments) without loading full files.
- **XGBoost**: use `xgb_params` (e.g. `n_jobs=-1`, `tree_method="hist"`, GPU `device="cuda"` when available). For cluster-scale training, use vendor docs (XGBoost on Spark/YARN, Dask, Ray Train); this library stays a **single-node default**.

## Modern NLP (embeddings)

- **Optional extra** `pip install ".[nlp]"`**: `PairEmbeddingBuilder` in `quorabust.embedding_features` uses `sentence-transformers` to encode pairs and feeds cosine / L2 / pooling stats into the same XGBoost head. Training can select `--feature-backend embedding` in `quorabust-train`.

## Online serving, SLOs, monitoring

- **`quorabust-serve`**: FastAPI app with `/health`, `/ready`, `/predict`, and **Prometheus** `/metrics` (latency histogram + request counter). Run behind your platform ingress and attach SLO dashboards to those metrics.
- **Readiness**: `/ready` is 503 until a model path is loaded successfully.
- **Thresholds**: `/predict` returns the raw probability plus an `is_duplicate`
  decision. Use `?threshold=...` for per-request policy tests, or set
  `QUORABUST_DECISION_THRESHOLD` / artifact metadata for service defaults.

## A/B testing

- Set `QUORABUST_MODEL_B` to a second artifact path. Clients send **`X-Quorabust-Variant: b`** (default **a**). Your edge proxy can split traffic; this repo only routes per request.

## MLOps (lightweight)

- **Registry**: `quorabust.registry.append_model_record` appends JSON lines under `registry_dir` (e.g. path, metrics, git SHA). Swap for MLflow/W&B when you need a full registry UI.
- **Drift**: `quorabust.drift.mean_shift_scores` compares current batch feature means to a **reference** dict (e.g. from training `meta`). No feature store server—persist reference JSON next to the model and refresh on retrains.
