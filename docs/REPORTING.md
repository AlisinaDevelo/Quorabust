# Model Reporting

Use `quorabust-report` to turn a saved artifact into a Markdown model card. The report
is meant for artifact review: it shows lineage, feature schema, persisted metrics, and
optional holdout metrics with a threshold confusion matrix, threshold sweep, and
probability calibration summary.

## Smoke Workflow

This repository includes `examples/smoke_pairs.csv` so the full train-to-report path can
run in CI without the Kaggle dataset.

```bash
quorabust-train \
  --csv examples/smoke_pairs.csv \
  --out /tmp/quorabust-smoke.pkl \
  --metadata-out /tmp/quorabust-smoke.meta.json \
  --eval-fraction 0 \
  --seed 7

quorabust-report \
  --model /tmp/quorabust-smoke.pkl \
  --artifact-label quorabust-smoke.pkl \
  --eval-csv examples/smoke_pairs.csv \
  --thresholds 0.3,0.5,0.7 \
  --calibration-bins 5 \
  --out /tmp/quorabust-smoke-model-card.md

quorabust-report \
  --model /tmp/quorabust-smoke.pkl \
  --artifact-label quorabust-smoke.pkl \
  --eval-csv examples/smoke_pairs.csv \
  --format json \
  --out /tmp/quorabust-smoke-model-card.json \
  --manifest-out /tmp/quorabust-smoke-model-card.manifest.json
```

The smoke dataset proves the command path works. It is not a benchmark and should not be
used for public model-quality claims.

Reports include precision, recall, F1, accuracy, and predicted-positive rate at the
selected threshold plus a threshold sweep. Use `--thresholds` to compare operating
points before choosing one.

Reports also include calibration diagnostics: Brier score, expected calibration error,
mean predicted probability, mean observed rate, and probability-bin rows. Use
`--calibration-bins` to control how many bins are printed. Calibration helps decide
whether probabilities are fit for thresholding or only useful for ranking.

## Evaluation slices

When a permitted holdout includes caller-owned labels such as language, product area, or
traffic cohort, request bounded per-label diagnostics without changing the model or
benchmark protocol:

```bash
quorabust-report \
  --model models/quorabust.pkl \
  --eval-csv data/processed/holdout.csv \
  --slice-column language \
  --slice-column domain \
  --slice-manifest reports/holdout-slices.json \
  --max-slices 20 \
  --format json \
  --out reports/quorabust-slices.json
```

For a benchmark or release report, bind the labels to a path-light sidecar whose hash is the
exact evaluated CSV bytes:

```json
{
  "schema_version": 1,
  "source": {
    "reference": "dataset://permitted-holdout-v1",
    "sha256": "<64-hex-evaluated-csv-sha256>",
    "rows": 1200
  },
  "columns": {
    "language": {
      "labeling_method": "dataset-owner annotation, reviewed 2026-08-09"
    },
    "domain": {
      "labeling_method": "dataset-owner taxonomy v2"
    }
  }
}
```

`--slice-manifest` requires the exact requested `--slice-column` set, a current CSV hash, a
matching row count, and a non-empty labeling method for every column. The report records the
canonical sidecar plus observed per-label row counts in its reproducibility manifest. A tiny
synthetic fixture exercises this contract in CI; it is not multilingual or domain quality
evidence.

Each requested column is validated for existence, missing or blank labels, and bounded
cardinality. The JSON and Markdown report contains one row per label with its sample count,
positive rate, selected-threshold metrics, log loss, and calibration diagnostics. Labels are
sorted for deterministic output, and the evaluation manifest records the requested slice
columns. Quorabust does not infer language or domain membership; slice rows are descriptive
evidence only and do not establish model quality without a permitted, representative
dataset.

Each slice also carries a 95% Wilson binomial interval for positive rate, accuracy,
precision, and recall. The JSON object records the method and confidence level; Markdown
prints the intervals and the same caveat. Undefined denominators, such as recall for a
slice with no positive labels, are reported as unavailable rather than fabricated. Log
loss, F1, ROC-AUC, Brier score, and expected calibration error remain point estimates
unless a permitted resampling protocol is supplied.

## Real Evaluation

Start by freezing the source, split roles, decision policy, and runtime provenance in the
[benchmark protocol manifest](BENCHMARK_PROTOCOL.md). Validate it before publishing any
real-data comparison:

```bash
quorabust-validate-protocol \
  --protocol reports/quorabust-benchmark-protocol.json
```

The protocol validator requires a passing qid-aware audit, distinct train/tuning/calibration/
final-holdout artifacts, a question-component split, and an explicit barrier against using
the final holdout for selection. It is a reproducibility gate, not a quality result.

Then run the data preflight and keep its JSON output next to the training and evaluation
manifests:

```bash
quorabust-audit-data \
  --csv data/raw/train.csv \
  --out reports/train-data-audit.json \
  --require-question-ids
```

The audit records a SHA-256 for the source CSV, required-column and binary-label checks,
empty or repeated pair signals, and whether complete question IDs are available. It does
not replace the leakage-aware split performed by `quorabust-train`; it makes the input
contract and any warnings reviewable before training starts. `--require-question-ids` makes
missing or blank `qid1`/`qid2` a failed audit for the public benchmark protocol. Omit it only
for an explicitly documented exploratory or customer-domain run where the row-level fallback
is an accepted limitation.

For a role-based benchmark, train on the frozen `train` role and pass the independent
`tuning` role explicitly. This keeps the raw source archive, model-fit rows, threshold-selection
rows, calibration rows, and final evaluation rows distinct:

```bash
quorabust-train \
  --csv /external/quora/roles/train.csv \
  --eval-csv /external/quora/roles/tuning.csv \
  --out models/quorabust.pkl \
  --metadata-out models/quorabust.meta.json \
  --require-question-ids \
  --eval-fraction 0.1 \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --threshold-metric f1 \
  --seed 42

quorabust-report \
  --model models/quorabust.pkl \
  --artifact-label quorabust-tfidf-v1.pkl \
  --eval-csv /external/quora/roles/final_holdout.csv \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --calibration-bins 10 \
  --out reports/quorabust-tfidf-v1.md
```

When `--eval-csv` is supplied with complete Kaggle `qid1` and `qid2` columns,
`quorabust-train` verifies that no question component crosses the training/evaluation
boundary and records `eval_split_source: explicit_csv`. The artifact metadata records
`split_strategy: question_component_holdout` and both role hashes. Without `--eval-csv`,
the existing deterministic connected-component holdout remains available; datasets without
complete question IDs retain the documented shuffled-row fallback.

`--eval-out` exports the exact rows used for early stopping and threshold selection in the
single-source workflow. With `--eval-csv`, the supplied evaluation role is never rewritten;
its SHA-256 is recorded directly so the later report can be tied to the frozen role artifact.

Use `--require-question-ids` for public benchmark runs. It fails before training when
`qid1` or `qid2` is missing or incomplete, preventing an accidental row-level fallback.
Leave it off for customer datasets that do not provide IDs, but call out the fallback in
the evaluation manifest and model card.

Record the dataset source, split method, command, commit SHA, and date next to any
published result. Do not compare artifacts unless they use the same holdout split and
threshold.

Every report generated with `--eval-csv` includes an `evaluation_manifest` with SHA-256
hashes for the model and holdout CSV, row and label counts, the threshold policy, training
lineage, runtime details, the report commit, and the exact report invocation. Use
`--manifest-out` to write that object as a separate JSON sidecar. It is an audit record,
not a substitute for a declared dataset source or split policy; the holdout itself must
still be frozen and kept outside the training data.

Training metadata sidecars also include the post-save `artifact_sha256`. Use that digest
as `QUORABUST_MODEL_SHA256` (and `QUORABUST_MODEL_B_SHA256` for A/B) to make serving fail
closed if the promoted pickle bytes change.

Use `--format json` when you want CI or release tooling to compare metrics without
scraping Markdown.

Use `quorabust-validate-report` as a release gate for machine-readable reports:

```bash
quorabust-report \
  --model models/quorabust.pkl \
  --artifact-label quorabust-tfidf-v1.pkl \
  --eval-csv data/processed/holdout.csv \
  --format json \
  --out reports/quorabust-tfidf-v1.json \
  --manifest-out reports/quorabust-tfidf-v1.manifest.json

quorabust-validate-report \
  --report reports/quorabust-tfidf-v1.json \
  --require-holdout \
  --require-calibration \
  --require-manifest \
  --require-question-component-split
```

`--require-question-component-split` is an opt-in benchmark release policy. It fails closed
unless the evaluation manifest records `question_component_holdout`, `require_question_ids: true`,
both `qid1`/`qid2` metadata columns, and those columns in the evaluated CSV. Leave it off for
an explicitly documented exploratory or customer-domain report that accepts the row-level
fallback. This policy validates protocol metadata; it does not establish model quality.

Use repeated `--compare-model label=path` arguments when you need to compare trained
backends against the same holdout split:

```bash
quorabust-report \
  --model models/quorabust-tfidf.pkl \
  --artifact-label quorabust-tfidf.pkl \
  --compare-model tfidf=models/quorabust-tfidf.pkl \
  --compare-model embedding=models/quorabust-embedding.pkl \
  --compare-model cross=models/quorabust-cross.pkl \
  --eval-csv data/processed/holdout.csv \
  --format json \
  --out reports/backend-comparison.json
```

The comparison rows are sorted by F1. Treat the table as meaningful only when all
artifacts use the same holdout CSV, threshold, and metric code.

When `quorabust-train` has a holdout split, it stores a selected `decision_threshold` in
artifact metadata. The threshold is chosen from `--thresholds` by maximizing
`--threshold-metric` (default `f1`). The supported metrics include `expected_cost`, which
minimizes weighted false-positive and false-negative counts per holdout row. Supply both
`--false-positive-cost` and `--false-negative-cost`; these are relative policy units and
are persisted with the selected threshold. Serving uses that artifact threshold unless the
request overrides it with `?threshold=...`.

## Separate calibration and threshold selection

For a promoted artifact, fit probability calibration and choose the action threshold on
the independent `calibration` and `tuning` roles rather than reusing training or final
evaluation rows:

```bash
quorabust-calibrate \
  --model models/quorabust.pkl \
  --calibration-csv /external/quora/roles/calibration.csv \
  --threshold-csv /external/quora/roles/tuning.csv \
  --calibration-method sigmoid \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --threshold-metric f1 \
  --out models/quorabust-calibrated.qmodel \
  --metadata-out models/quorabust-calibrated.meta.json
```

To make the action policy cost-sensitive, replace `f1` and add
`--threshold-metric expected_cost --false-positive-cost 10 --false-negative-cost 1`.

Use `--calibration-method isotonic` when the calibration sample is large enough to support
its more flexible mapping. The command rejects reuse of the source training/evaluation CSV,
identical calibration and threshold files, and overlapping question IDs when both inputs
provide complete `qid1`/`qid2` columns. It records both data hashes, the base artifact
hash, raw and calibrated reliability diagnostics, and the threshold policy. Evaluate the
result once on a final holdout that was not used by either step.

## Sample model card

A checked-in example produced from the [`examples/smoke_pairs.csv`](../examples/smoke_pairs.csv)
fixture lives at [SAMPLE_MODEL_CARD.md](SAMPLE_MODEL_CARD.md). It shows the exact shape of a
generated card without depending on the Kaggle dataset; its metrics describe that toy data
only. The commands to regenerate it are in that file's Reproduce section.
