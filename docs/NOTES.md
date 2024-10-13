# Notes

## Goal

Turn pairs of questions into a small numeric feature matrix (TF–IDF cosine similarity, word Jaccard, length stats), then train a gradient-boosted tree model to predict `is_duplicate`.

## Data

The public [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset is the natural training source. Download `train.csv`, place it under `data/raw/`, and load it with pandas. Expected columns include `question1`, `question2`, and `is_duplicate`.

## Design choices

- **TF–IDF** uses `stop_words=None` so short or stopword-heavy questions still produce a vocabulary (important for tests and for “what is …” style duplicates).
- **Early stopping** is enabled when you pass `eval_df` to `train_duplicate_classifier`; `early_stopping_rounds` defaults to 20 and can be overridden via `xgb_params`.
- **Artifacts**: serialize `(PairFeatureBuilder, XGBClassifier)` together if you save models; the vectorizer vocabulary must match training.

## Notebooks

The `notebooks/` folder is reserved for ad-hoc EDA and experiments. Templates were removed in favor of a small, tested library; recreate notebooks as needed for your workflow.
