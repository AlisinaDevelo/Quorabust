import numpy as np
import pandas as pd

from quorabust.model import train_duplicate_classifier


class _ZeroBuilder:
    def fit_from_frame(self, df, col_q1="question1", col_q2="question2"):
        return self

    def transform_frame(self, df, col_q1="question1", col_q2="question2"):
        return np.zeros((len(df), 5), dtype=np.float64)

    def transform_pairs(self, q1, q2):
        return np.zeros((len(q1), 5), dtype=np.float64)


def test_train_accepts_custom_feature_builder():
    df = pd.DataFrame(
        {
            "question1": ["a", "b", "c"],
            "question2": ["d", "e", "f"],
            "is_duplicate": [0, 1, 0],
        }
    )
    b, clf = train_duplicate_classifier(
        df,
        feature_builder=_ZeroBuilder(),
        xgb_params={"n_estimators": 5, "max_depth": 2},
    )
    assert clf.predict_proba(b.transform_frame(df)).shape == (3, 2)
