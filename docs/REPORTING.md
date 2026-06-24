# Model Reporting

Use `quorabust-report` to turn a saved artifact into a Markdown model card. The report
is meant for artifact review: it shows lineage, feature schema, persisted metrics, and
optional holdout metrics with a threshold confusion matrix.

## Smoke Workflow

This repository includes `examples/smoke_pairs.csv` so the full train-to-report path can
run in CI without the Kaggle dataset.

```bash
quorabust-train \
  --csv examples/smoke_pairs.csv \
  --out /tmp/quorabust-smoke.pkl \
  --eval-fraction 0 \
  --seed 7

quorabust-report \
  --model /tmp/quorabust-smoke.pkl \
  --artifact-label quorabust-smoke.pkl \
  --eval-csv examples/smoke_pairs.csv \
  --out /tmp/quorabust-smoke-model-card.md
```

The smoke dataset proves the command path works. It is not a benchmark and should not be
used for public model-quality claims.

## Real Evaluation

For comparable numbers, generate the report from a held-out CSV that was not used for
training:

```bash
quorabust-train \
  --csv data/raw/train.csv \
  --out models/quorabust.pkl \
  --eval-fraction 0.1 \
  --seed 42

quorabust-report \
  --model models/quorabust.pkl \
  --artifact-label quorabust-tfidf-v1.pkl \
  --eval-csv data/processed/holdout.csv \
  --out reports/quorabust-tfidf-v1.md
```

Record the dataset source, split method, command, commit SHA, and date next to any
published result. Do not compare artifacts unless they use the same holdout split and
threshold.
