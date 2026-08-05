# Scale, NLP, serving, and MLOps

This document maps **ambitions** to what ships in-repo and what stays external.

## Big data and distributed training

- **In-repo**: `quorabust.dataio.iter_csv_batches` streams CSV chunks with pandas (`chunksize`) so you can build custom loops (sample, aggregate stats, incremental vocab experiments) without loading full files.
- **XGBoost**: use `xgb_params` (e.g. `n_jobs=-1`, `tree_method="hist"`, GPU `device="cuda"` when available). For cluster-scale training, use vendor docs (XGBoost on Spark/YARN, Dask, Ray Train); this library stays a **single-node default**.

## Modern NLP (embeddings)

- **Optional extra** `pip install ".[nlp]"`**: `PairEmbeddingBuilder` in `quorabust.embedding_features` uses `sentence-transformers` to encode pairs and feeds cosine / L2 / pooling stats into the same XGBoost head. Training can select `--feature-backend embedding` in `quorabust-train`.
- **Cross-encoder pair scoring**: `PairCrossEncoderBuilder` in
  `quorabust.cross_encoder_features` uses a Sentence Transformers `CrossEncoder` to score
  each pair directly, then feeds that score plus length stats into the same XGBoost head.
  Select it with `--feature-backend cross-encoder`. This is the modern high-accuracy path
  for pair scoring, but it is slower than TF-IDF or bi-encoder embeddings because every
  pair must be passed through the transformer jointly.

## Catalog retrieval

The optional catalog path starts with a deterministic TF-IDF first-stage retriever:

```bash
quorabust-retrieve \
  --catalog-csv data/catalog/questions.csv \
  --query "How do I deploy an API?" \
  --k 10
```

The retrieval module also exposes a bounded batch reranker contract. With the optional NLP
extra, select `--retriever embedding` and pass `--reranker-model` to apply a cross-encoder
only to the candidate set. Measure candidate recall@k first, and do not interpret an
uncalibrated reranker score as `P(duplicate)`. See
[#15](https://github.com/AlisinaDevelo/Quorabust/issues/15) for the retrieve-and-rerank
product work.

### Retrieval evaluation and cost proxy

Use `quorabust-retrieve-benchmark` with a qrels CSV containing `query`, `question_id`, and
optional non-negative `relevance` values:

```bash
quorabust-retrieve-benchmark \
  --catalog-csv data/catalog/questions.csv \
  --qrels-csv data/processed/retrieval-qrels.csv \
  --ks 1,5,10 \
  --candidate-k 50 \
  --warmup-runs 1 \
  --repetitions 3 \
  --timeout-seconds 120 \
  --out reports/retrieval-benchmark.json
```

The report records first-stage and final recall/MRR/NDCG, retrieval/rerank/end-to-end
latency distributions, throughput, reranker pair count, source hashes, model names, runtime,
and the exact command. By default, one complete serial warm-up pass is discarded and three
complete serial passes contribute latency samples. Quality metrics are calculated once from
the first measured pass so query count is not overweighted. The optional timeout is a
cooperative wall-clock deadline checked between queries and stages; a process supervisor is
still required to interrupt a native model call that does not return. Reranker pair count is
a bounded-work cost proxy, not a cloud billing estimate. The checked-in example catalog and
qrels are smoke fixtures only. Real comparisons must use the frozen, permitted evaluation
protocol tracked in
[#13](https://github.com/AlisinaDevelo/Quorabust/issues/13) and the quality/cost gate in
[#18](https://github.com/AlisinaDevelo/Quorabust/issues/18).

## Online serving, SLOs, monitoring

- **`quorabust-serve`**: FastAPI app with `/health`, `/ready`, `/predict`, and **Prometheus** `/metrics` (prediction and endpoint RED metrics). Route-level request counters include status codes, latency histograms support p50/p95 SLOs, and unknown paths are grouped to keep label cardinality bounded. Run behind your platform ingress and attach SLO dashboards to those metrics.
- **Readiness**: `/ready` is 503 until a model path is loaded successfully.
- **Thresholds**: `/predict` returns the raw probability plus an `is_duplicate`
  decision. Use `?threshold=...` for per-request policy tests, or set
  `QUORABUST_DECISION_THRESHOLD` / artifact metadata for service defaults.

## A/B testing

- Set `QUORABUST_MODEL_B` to a second artifact path. Clients send **`X-Quorabust-Variant: b`** (default **a**). Your edge proxy can split traffic; this repo only routes per request.

## MLOps (lightweight)

- **Registry**: `quorabust.registry.append_model_record` appends JSON lines under `registry_dir` (e.g. path, metrics, git SHA). Swap for MLflow/W&B when you need a full registry UI.
- **Drift**: `quorabust.drift.mean_shift_scores` compares current batch feature means to a **reference** dict (e.g. from training `meta`). No feature store server—persist reference JSON next to the model and refresh on retrains.
