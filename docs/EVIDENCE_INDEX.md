# Evidence Index

Quorabust is a production-minded semantic matching service and an ML release-evidence
artifact. This page is the shortest route through the project for a technical reviewer,
recruiter, or deployment owner.

## The Short Route

1. [Product positioning](PRODUCT.md) explains the buyer-facing matching workflow and the
   supported service boundary.
2. [Architecture](ARCHITECTURE.md) shows the offline training, artifact, registry, and
   FastAPI serving path.
3. [API snapshots](demo-assets/README.md) show representative requests, responses, and
   OpenAPI output without requiring a running service.
4. [Real benchmark result](REAL_BENCHMARK_RESULT.md) records the permitted frozen-role
   comparison, provenance, measured tradeoffs, and promotion decision.
5. [Artifact trust policy](ARTIFACTS.md) explains the safe `.qmodel` boundary and why
   transformer-backed artifacts are not promoted automatically.
6. [Enterprise operations](ENTERPRISE.md) and [scaling evidence](SCALING.md) document
   serving controls, retrieval profiles, and caller-owned release policies.

## Current Position

| Surface | What the repository can defend | Evidence and boundary |
| --- | --- | --- |
| Supported service | Thresholded pair scoring through FastAPI with health/readiness, request IDs, Prometheus metrics, A/B routing, drift helpers, and safe model metadata. | [Product](PRODUCT.md), [architecture](ARCHITECTURE.md), and [API snapshots](demo-assets/README.md). Runtime behavior is covered by the test suite and container gate. |
| Supported model control | TF-IDF + XGBoost + isotonic is the current supported control because it has a safe `.qmodel`, small artifact, and fast local warm scoring. | Corrected strict v2 reports ROC-AUC `0.7896`, PR-AUC `0.6243`, F1 `0.6711`, and a 354,745-byte artifact. These are one permitted holdout run and a local warm comparison, not universal SLOs. |
| Quality candidate | A direct Quora cross-encoder is materially stronger on the same holdout, but remains direct scoring evidence rather than a promoted Quorabust artifact. | The recorded run reports ROC-AUC `0.9731`, PR-AUC `0.9472`, F1 `0.8940`, approximately 1.6 GB peak RSS, and roughly 163-184 pairs/second on the local CPU profile. Safe packaging, startup, and deployment validation remain open. |
| Intermediate candidate | Sentence-transformer embeddings improve measured quality over the lexical control, but the current trusted pickle is not an untrusted-distribution boundary. | The recorded run reports ROC-AUC `0.8834`, PR-AUC `0.7799`, a 173,917,653-byte pickle, approximately 1.29 GB peak RSS, and 4,216 ms local load time. |
| Retrieval path | TF-IDF retrieval and optional bounded reranking have reproducible quality, latency, provenance, and release-policy contracts. | The 1,000-query sample reports recall@1/@5/@10 of `0.3066`/`0.5435`/`0.6549`, warm p95 `7.98 ms`, and 143.31 queries/second at concurrency 1. The full 13,046-query attempt hit the cooperative deadline; no capacity extrapolation is made. |

## Release Controls

`quorabust-validate-retrieval` can fail closed on caller-owned policies for:

- first-stage and final recall at selected cutoffs;
- end-to-end and fresh-process p95 latency;
- catalog size and first-stage candidate-k;
- reranker pairs per measured query as a bounded work proxy;
- peak RSS, declared artifact bytes, and declared model-cache bytes;
- source hashes, runtime provenance, measurement counts, and profile invariants.

These controls compare supplied evidence with a deployment policy. They do not choose a
universal target, turn the smoke fixture into quality evidence, or establish production
capacity. The pair-profile validator provides the analogous resource and optional quality
controls for pair-classifier artifacts.

## Open Gates

| Issue | What would close it | Current status |
| --- | --- | --- |
| [#15 Retrieve and rerank](https://github.com/AlisinaDevelo/Quorabust/issues/15) | Permitted real-data candidate recall/ranking evidence and a concrete deployment path. | Open. Retrieval implementation and evidence contracts are shipped. |
| [#18 Quality versus cost](https://github.com/AlisinaDevelo/Quorabust/issues/18) | A deployment-owned quality/cost decision using representative data, cold start, warm tails, memory, cache, timeout, and work budgets. | Open. Enforcement infrastructure is shipped; promotion is not claimed. |
| [#16 Hard-negative fine-tuning](https://github.com/AlisinaDevelo/Quorabust/issues/16) | A reproducible experiment with hard-negative provenance, including an acceptable no-go result. | Open. Do not fine-tune for novelty alone. |
| [#19 Domain and multilingual slices](https://github.com/AlisinaDevelo/Quorabust/issues/19) | Permitted representative labels and a frozen slice protocol. | Open. English Quora evidence does not support Persian, Italian, or customer-domain claims. |

## Accurate Portfolio Claim

> I built a deployable semantic matching service with reproducible training, model lineage,
> protocol-bound evaluation, thresholded API decisions, safe artifact handling, observability,
> retrieval profiling, and fail-closed release policies. The measured transformer quality/cost
> tradeoff is documented, while the lightweight TF-IDF/XGBoost control remains the supported
> artifact until safe packaging and deployment evidence justify a change.

## Do Not Claim

- Do not call any candidate state of the art.
- Do not present the 1,000-query retrieval sample as full-catalog capacity.
- Do not present Quora metrics as multilingual, customer-domain, or production guarantees.
- Do not imply that a direct cross-encoder score is a serialized, safe Quorabust artifact.
- Do not turn caller-owned policy examples into universal SLOs or cloud billing estimates.
