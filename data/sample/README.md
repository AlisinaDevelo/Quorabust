# Sample data (synthetic)

`synthetic_pairs.csv` is **fully synthetic** data produced by
[`scripts/make_sample_data.py`](../../scripts/make_sample_data.py). It is not the
Kaggle Quora Question Pairs dataset and contains no real questions or labels.

Its only purpose is to let the train → report → serve workflow run end-to-end —
and to back the checked-in [sample model card](../../docs/SAMPLE_MODEL_CARD.md) —
without downloading or redistributing third-party data. The classes are trivially
separable, so any metrics derived from it describe the toy data, not real model
quality.

Regenerate deterministically:

```bash
python scripts/make_sample_data.py --out data/sample/synthetic_pairs.csv --rows 2000 --seed 42
```
