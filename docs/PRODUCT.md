# Enterprise Product Positioning

Quorabust is a compact semantic matching service for teams that need to score pairs of
short texts, apply a reviewed threshold, and operate the model behind a normal backend
contract.

The public dataset is Quora Question Pairs, but the useful enterprise pattern is broader:
deduplicate support questions, route knowledge-base suggestions, catch duplicate tickets,
flag near-identical marketplace listings, or review repeated customer feedback.

## Buyer-Facing Promise

Quorabust answers one operational question:

> Given two short text records, how likely are they to mean the same thing, and should the
> system treat them as a duplicate under the current policy?

It does this with:

- Reproducible offline training.
- A deployable FastAPI scoring service.
- Thresholded decisions, not only raw probabilities.
- Holdout-selected serving thresholds persisted in artifact metadata.
- Model-card reporting with threshold sweeps.
- Artifact metadata, JSON sidecars, and safe public model metadata.
- Prometheus metrics, A/B model routing, drift helpers, and load-test assets.

## Method Strategy

The TF-IDF + XGBoost backend is the baseline/control model. It is fast, cheap, inspectable,
and easy to serve. That makes it valuable for production comparisons even when a stronger
NLP model exists.

The modern path is:

1. **TF-IDF + XGBoost baseline** for speed, explainable feature values, and cheap serving.
2. **Sentence-transformer embeddings + XGBoost** for better semantic recall when wording
   differs.
3. **Cross-encoder reranker** for the highest-accuracy pair scoring when latency and
   compute budget allow it.

Do not claim state-of-the-art quality from the checked-in smoke model. Use a real held-out
dataset or customer-domain labels before making performance claims.

## API Contract

`POST /predict` accepts aligned `question1[]` and `question2[]` arrays. It returns:

- `proba_duplicate`: raw duplicate probability per pair.
- `is_duplicate`: thresholded decision per pair.
- `decision_threshold`: threshold used for the decision.
- `variant`: active model variant, useful during A/B rollout.
- `features`: optional input feature values when `?explain=true`.

Decision threshold precedence:

1. Request query parameter: `?threshold=0.7`.
2. Holdout-selected artifact metadata: `decision_threshold`.
3. Environment: `QUORABUST_DECISION_THRESHOLD`.
4. Default: `0.5`.

## Enterprise Readiness

What is already in-repo:

- Health and readiness probes.
- Prometheus metrics.
- OpenAPI docs.
- Docker build.
- k6 load test.
- Grafana dashboard starter.
- JSONL registry.
- Drift mean-shift helpers.
- Model-card report generation.
- Dependency audit workflow.
- Safe `/models` metadata endpoint.

What should be added before a serious production deployment:

- Real domain-specific benchmark and model card.
- Probability calibration and recommended threshold persistence.
- Non-pickle artifact format for untrusted distribution.
- Authentication and rate limiting at the gateway.
- A production model registry such as MLflow or an internal equivalent.
- Structured request IDs and centralized logs.

## Portfolio Story

Quorabust should be presented as an ML-backed backend/platform project, not as a Kaggle
notebook. The strongest story is operational:

> I built a deployable semantic matching service with reproducible training, model
> lineage, model-card reporting, thresholded API decisions, observability, A/B routing,
> and drift hooks.

That is more marketable than claiming a single algorithm is impressive.
