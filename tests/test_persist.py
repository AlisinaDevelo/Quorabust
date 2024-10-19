import numpy as np
import pandas as pd

from quorabust.model import predict_proba_duplicate, train_duplicate_classifier
from quorabust.persist import load_classifier, save_classifier


def _df():
    return pd.DataFrame(
        {
            "question1": [f"topic {i} explain" for i in range(24)],
            "question2": [f"explain topic {i % 6}" for i in range(24)],
            "is_duplicate": [i % 2 for i in range(24)],
        }
    )


def test_save_load_roundtrip(tmp_path):
    df = _df()
    builder, clf = train_duplicate_classifier(
        df,
        xgb_params={"n_estimators": 16, "max_depth": 3},
    )
    path = tmp_path / "model.pkl"
    save_classifier(path, builder, clf, meta={"run": "test"})
    b2, c2, meta = load_classifier(path)
    assert meta.get("run") == "test"
    q1, q2 = ["what is python"], ["how to learn python"]
    p1 = predict_proba_duplicate(builder, clf, q1, q2)
    p2 = predict_proba_duplicate(b2, c2, q1, q2)
    assert np.allclose(p1, p2)
