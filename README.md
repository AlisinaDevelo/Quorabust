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
from quorabust import clean_text, train_duplicate_classifier, predict_proba_duplicate

df = pd.read_csv("data/raw/quora_train.csv")  # place Kaggle Quora data here
df = df.rename(columns=lambda c: c.strip())
builder, clf = train_duplicate_classifier(df.head(5000))  # example subset

proba = predict_proba_duplicate(
    builder,
    clf,
    ["How do I learn Python?"],
    ["What is the best way to learn Python?"],
)
print(proba[:, 1])  # P(duplicate)
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/quorabust/` | Package: `preprocess`, `features`, `model` |
| `tests/` | Pytest suite |
| `data/raw/` | Original CSVs (not committed; see `data/README.md`) |
| `data/processed/` | Cleaned splits |
| `models/` | Saved `.pkl` artifacts (gitignored) |
| `notebooks/` | Exploratory work (optional) |

## Development

```bash
ruff check src tests
pytest -q --cov=quorabust
```

Design notes and data pointers: [docs/NOTES.md](docs/NOTES.md).

## License

MIT — see [LICENSE](LICENSE).
