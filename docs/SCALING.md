# Scale, NLP, serving, and MLOps

This document maps **ambitions** to what ships in-repo and what stays external.

## Big data and distributed training

- **In-repo**: `quorabust.dataio.iter_csv_batches` streams CSV chunks with pandas (`chunksize`) so you can build custom loops (sample, aggregate stats, incremental vocab experiments) without loading full files.
- **XGBoost**: use `xgb_params` (e.g. `n_jobs=-1`, `tree_method="hist"`, GPU `device="cuda"` when available). For cluster-scale training, use vendor docs (XGBoost on Spark/YARN, Dask, Ray Train); this library stays a **single-node default**.

## Modern NLP (embeddings)

The pair builder encodes each distinct normalized question once per batch and keeps a
bounded in-memory LRU cache of 50,000 vectors by default. `batch_size` and `cache_size`
are explicit constructor controls; the cache is process-local and is not a persisted
feature store. This keeps repeated-question workloads predictable without allowing
unbounded text-derived memory growth.

- **Optional extra** `pip install ".[nlp]"`**: `PairEmbeddingBuilder` in `quorabust.embedding_features` uses `sentence-transformers` to encode pairs and feeds cosine / L2 / pooling stats into the same XGBoost head. Training can select `--feature-backend embedding` in `quorabust-train`.
- **Cross-encoder pair scoring**: `PairCrossEncoderBuilder` in
  `quorabust.cross_encoder_features` uses a Sentence Transformers `CrossEncoder` to score
  each pair directly, then feeds that score plus length stats into the same XGBoost head.
  Select it with `--feature-backend cross-encoder`. This is the modern high-accuracy path
  for pair scoring, but it is slower than TF-IDF or bi-encoder embeddings because every
  pair must be passed through the transformer jointly.

### Leakage-safe hard negatives

Use `quorabust-mine-hard-negatives` to generate lexical near-neighbours for training or
tuning experiments:

```bash
quorabust-mine-hard-negatives \
  --csv data/processed/train.csv \
  --out data/derived/train-hard-negatives.csv \
  --metadata-out data/derived/train-hard-negatives.meta.json \
  --candidate-k 50 \
  --negatives-per-positive 2 \
  --seed 42
```

The input must contain complete `qid1`/`qid2` values and binary labels. Only label-1 edges
build the known-positive graph; every candidate in the anchor's connected component is
excluded, including transitive positives. The output remains a normal pair CSV, and each
row records its source positive row, anchor side, lexical rank, and retrieval score. The
sidecar binds the generated file to the input hash, configuration, output hash, git
revision, and runtime.

TF-IDF is the dependency-free control. To test semantic candidate generation, install the
optional NLP extra and select the existing sentence-transformer catalog retriever:

```bash
quorabust-mine-hard-negatives \
  --csv data/processed/train.csv \
  --out data/derived/train-hard-negatives-dense.csv \
  --retriever embedding \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --candidate-k 50 \
  --negatives-per-positive 2
```

The embedding model is used only to generate bounded candidates; it is not fine-tuned by
this command, and its score is not a calibrated duplicate probability. Base CI does not
download model weights; optional NLP evaluation belongs in a separately permitted run.

Keep calibration and final-holdout rows out of the mining input. If the source contains
only the training role, the generated negatives may be appended to training data after
review; they must not be used to tune a decision threshold or to replace the untouched
final holdout. TF-IDF similarity is a candidate-generation signal, not a duplicate
probability, and this path makes no quality claim without a permitted real-data evaluation.
See [#16](https://github.com/AlisinaDevelo/Quorabust/issues/16) and
[#44](https://github.com/AlisinaDevelo/Quorabust/issues/44), and
[#45](https://github.com/AlisinaDevelo/Quorabust/issues/45).

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
latency distributions including p50/p95/p99, throughput, reranker pair count, source hashes,
model names, runtime, and the exact command. On Unix, runtime also records normalized process
peak RSS as `peak_rss_bytes`; this is `ru_maxrss` since process start, including model startup,
not a per-request allocation delta. Runtime also reports retriever and reranker initialization
times separately; these include model construction/download and catalog encoding for the
selected backend, but not Python interpreter import time. By default, one complete serial
warm-up pass is discarded and three complete serial passes contribute latency samples. Quality
metrics are calculated once from the first measured pass so query count is not overweighted.
The optional timeout is a
cooperative wall-clock deadline checked between queries and stages; a process supervisor is
still required to interrupt a native model call that does not return. Reranker pair count is
a bounded-work cost proxy, not a cloud billing estimate. The checked-in example catalog and
qrels are smoke fixtures only. Real comparisons must use the frozen, permitted evaluation
protocol tracked in
[#13](https://github.com/AlisinaDevelo/Quorabust/issues/13) and the quality/cost gate in
[#18](https://github.com/AlisinaDevelo/Quorabust/issues/18).

Reports also include `query_length_strata` for measured latency by whitespace-token query
length: `short` is 1-5 tokens, `medium` is 6-15, and `long` is 16 or more. Each populated
bucket reports its source-case count, measured sample count, token-count range, and retrieval,
rerank, and end-to-end p50/p95/p99 summaries. Warm-up passes are excluded; repetitions count
once per case. Empty buckets are omitted. These are performance slices only and do not establish
quality, language coverage, or production capacity without representative permitted data.

### Retrieval report release gate

Validate a benchmark or fresh-process profile before attaching it to a release decision:

```bash
quorabust-validate-retrieval \
  --report reports/retrieval-benchmark.json \
  --min-final-recall-at-k 10=0.95 \
  --max-end-to-end-p95-ms 100

quorabust-validate-retrieval \
  --report reports/retrieval-profile.json \
  --max-cold-start-p95-ms 5000 \
  --max-peak-rss-bytes 2147483648 \
  --max-total-artifact-bytes 1073741824
```

The validator checks source hashes and byte counts, runtime provenance, repetition and latency
sample invariants, and the query-length strata contract. The policy values belong to the target
deployment and must be chosen from representative measurements; the command does not claim
that any displayed recall, latency, memory, or size threshold is universally correct. Cold-start
and artifact-size policies require a fresh-process profile; artifact-size enforcement also
requires at least one repeated `--artifact` input when generating that profile. RSS policy works
with either report and fails closed where peak RSS is unavailable. The profile is accepted
through its embedded `warm_benchmark` report and still retains its timing-only evidence boundary.

### Fresh-process profile

Use `quorabust-retrieve-profile` when warm latency is not enough for a rollout decision:

```bash
quorabust-retrieve-profile \
  --catalog-csv data/catalog/questions.csv \
  --qrels-csv data/processed/retrieval-qrels.csv \
  --candidate-k 50 \
  --cold-start-repetitions 3 \
  --warmup-runs 1 \
  --repetitions 3 \
  --timeout-seconds 120 \
  --out reports/retrieval-profile.json
```

Each cold-start measurement runs the benchmark in a fresh subprocess and records
process-to-report wall time. The report also carries warm benchmark metrics, input source
hashes and byte sizes, optional local artifact hashes/sizes from repeated `--artifact` flags,
and a path-light command. This is timing and packaging evidence only; it makes no quality
claim and does not upload or commit model weights. The child timeout is a bounded process
supervisor timeout, while the nested benchmark timeout remains a cooperative stage deadline.

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
