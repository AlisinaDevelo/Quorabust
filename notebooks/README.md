# Notebooks

Use this folder for exploratory analysis (EDA, plotting, error analysis). The shipped library and tests live under `src/quorabust` and `tests/`.

Suggested flow:

1. Load `data/raw/train.csv` (Quora Question Pairs).
2. Call `quorabust.preprocess.clean_text` on samples.
3. Fit `PairFeatureBuilder` on a slice and inspect feature distributions.
4. Train with `train_duplicate_classifier` and evaluate with `eval_log_loss`.
