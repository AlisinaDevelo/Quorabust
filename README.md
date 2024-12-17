# Quorabust

Library for **Quora-style duplicate question detection**: text cleaning, pairwise TF–IDF and lexical features, and an **XGBoost** classifier on top.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras: `pip install -e ".[viz,notebooks]"` for Matplotlib, Seaborn, and Jupyter.

## Usage

```python
import pandas as pd
from quorabust import (
    clean_text,
    load_classifier,
    predict_proba_duplicate,
    train_duplicate_classifier,
)

df = pd.read_csv("data/raw/train.csv")  # Kaggle Quora Question Pairs
df.columns = [c.strip() for c in df.columns]
builder, clf = train_duplicate_classifier(df.head(5000))

proba = predict_proba_duplicate(
    builder,
    clf,
    ["How do I learn Python?"],
    ["What is the best way to learn Python?"],
)
print(proba[:, 1])  # P(duplicate)
```

### Train from the terminal

With `data/raw/train.csv` in place:

```bash
quorabust-train --csv data/raw/train.csv --out models/quorabust.pkl
python -m quorabust --csv data/raw/train.csv --out models/quorabust.pkl   # equivalent
```

Options: `--max-rows N`, `--eval-fraction 0.1` (default), `--eval-fraction 0` to train on all rows without a holdout, `--seed`.

### Load a saved model

```python
from quorabust import load_classifier, predict_proba_duplicate

builder, clf, meta = load_classifier("models/quorabust.pkl")
print(meta)  # n_train, eval_log_loss, paths, etc.
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/quorabust/` | Package: `preprocess`, `features`, `model`, `persist`, `cli` |
| `tests/` | Pytest suite |
| `data/raw/` | Original CSVs (not committed; see `data/README.md`) |
| `data/processed/` | Cleaned splits |
| `models/` | Saved `.pkl` artifacts (gitignored) |
| `notebooks/` | Exploratory work (optional) |

## Development

```bash
ruff check src tests
mypy src/quorabust
pytest -q --cov=quorabust --cov-fail-under=70
pre-commit run --all-files   # optional
```

Design notes: [docs/NOTES.md](docs/NOTES.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## Enterprise / operations

Governance (security policy, Dependabot, audits), containers, and release expectations are summarized in [docs/ENTERPRISE.md](docs/ENTERPRISE.md). Saved model pickles include `meta` (CSV checksum, git revision, package version, metrics); only load trusted artifacts.

## License

MIT — see [LICENSE](LICENSE).
