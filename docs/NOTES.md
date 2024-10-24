# Notes

## Goal

Turn pairs of questions into a small numeric feature matrix (TF–IDF cosine similarity, word Jaccard, length stats), then train a gradient-boosted tree model to predict `is_duplicate`.

## Data

The public [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset is the natural training source. Download `train.csv`, place it under `data/raw/`, and load it with pandas. Expected columns include `question1`, `question2`, and `is_duplicate`.

## Design choices

- **TF–IDF** uses `stop_words=None` so short or stopword-heavy questions still produce a vocabulary (important for tests and for “what is …” style duplicates).
- **Early stopping** is enabled when you pass `eval_df` to `train_duplicate_classifier`; `early_stopping_rounds` defaults to 20 and can be overridden via `xgb_params`.
- **Artifacts**: use `save_classifier` / `load_classifier` from `quorabust.persist` (or `from quorabust import …`) so the `PairFeatureBuilder` (TF–IDF state) and `XGBClassifier` stay in sync.

## Training CLI

`quorabust-train` reads a CSV with `question1`, `question2`, and `is_duplicate`, shuffles, optionally holds out the first fraction for early stopping, fits the pipeline, prints log loss, and writes a pickle. See `quorabust.cli` or run `quorabust-train --help`.

## Notebooks

See [notebooks/10_workflow.ipynb](../notebooks/10_workflow.ipynb) for a short end-to-end checklist; add your own EDA notebooks alongside it.
