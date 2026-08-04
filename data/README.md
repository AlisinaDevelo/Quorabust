# Data directory

Datasets for Quorabust. See [docs/NOTES.md](../docs/NOTES.md) for download links and column expectations.

## Subdirectories

- `raw/`: Contains the original, unprocessed Quora dataset.
- `processed/`: Contains the cleaned and processed datasets ready for modeling.
- `external/`: Contains any supplementary or external datasets used in the project.

## Raw Data

Place the original Quora Question Pairs `train.csv` under `raw/` locally. Raw datasets
are intentionally not committed. Run `quorabust-audit-data` before training and keep the
generated manifest outside version control with the dataset.

## Processed Data

The `processed/` directory includes:
- `train.csv`: Processed training dataset after cleaning and feature engineering.
- `test.csv`: Processed testing dataset after cleaning and feature engineering.

## External Data

The `external/` directory includes any additional data sources used for feature engineering or other purposes.
