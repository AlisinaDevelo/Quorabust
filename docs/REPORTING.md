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
  --metadata-out /tmp/quorabust-smoke.meta.json \
  --eval-fraction 0 \
  --seed 7

quorabust-report \
  --model /tmp/quorabust-smoke.pkl \
  --artifact-label quorabust-smoke.pkl \
  --eval-csv examples/smoke_pairs.csv \
  --thresholds 0.3,0.5,0.7 \
  --out /tmp/quorabust-smoke-model-card.md

quorabust-report \
  --model /tmp/quorabust-smoke.pkl \
  --artifact-label quorabust-smoke.pkl \
  --eval-csv examples/smoke_pairs.csv \
  --format json \
  --out /tmp/quorabust-smoke-model-card.json
```

The smoke dataset proves the command path works. It is not a benchmark and should not be
used for public model-quality claims.

Reports include precision, recall, F1, accuracy, and predicted-positive rate at the
selected threshold plus a threshold sweep. Use `--thresholds` to compare operating
points before choosing one.

## Real Evaluation

For comparable numbers, generate the report from a held-out CSV that was not used for
training:

```bash
quorabust-train \
  --csv data/raw/train.csv \
  --out models/quorabust.pkl \
  --metadata-out models/quorabust.meta.json \
  --eval-fraction 0.1 \
  --seed 42

quorabust-report \
  --model models/quorabust.pkl \
  --artifact-label quorabust-tfidf-v1.pkl \
  --eval-csv data/processed/holdout.csv \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --out reports/quorabust-tfidf-v1.md
```

Record the dataset source, split method, command, commit SHA, and date next to any
published result. Do not compare artifacts unless they use the same holdout split and
threshold.

Use `--format json` when you want CI or release tooling to compare metrics without
scraping Markdown.

## Sample model card

A checked-in example produced from the [`examples/smoke_pairs.csv`](../examples/smoke_pairs.csv)
fixture lives at [SAMPLE_MODEL_CARD.md](SAMPLE_MODEL_CARD.md). It shows the exact shape of a
generated card without depending on the Kaggle dataset; its metrics describe that toy data
only. The commands to regenerate it are in that file's Reproduce section.
