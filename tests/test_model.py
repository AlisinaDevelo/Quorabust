import numpy as np
import pandas as pd

from quorabust.model import (
    eval_log_loss,
    predict_proba_duplicate,
    train_duplicate_classifier,
)


def _tiny_df():
    return pd.DataFrame(
        {
            "question1": [
                "how do i learn python",
                "best pizza in town",
                "what is machine learning",
                "capital of france",
                "install numpy pip",
                "weather today",
                "define recursion",
                "sort a list python",
            ],
            "question2": [
                "python learning resources",
                "where to get pizza",
                "explain ml basics",
                "paris is the capital",
                "pip install numpy",
                "forecast tomorrow",
                "recursion meaning",
                "python list sort",
            ],
            "is_duplicate": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def test_train_and_predict():
    df = _tiny_df()
    builder, clf = train_duplicate_classifier(
        df,
        xgb_params={"n_estimators": 24, "max_depth": 3, "learning_rate": 0.2},
    )
    proba = predict_proba_duplicate(
        builder,
        clf,
        ["how to learn python"],
        ["learning python tips"],
    )
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_train_with_eval_df():
    df = _tiny_df()
    eval_df = df.iloc[:3].copy()
    train_df = df.iloc[3:].copy()
    builder, clf = train_duplicate_classifier(
        train_df,
        eval_df=eval_df,
        xgb_params={"n_estimators": 40, "max_depth": 3},
    )
    ll = eval_log_loss(builder, clf, df)
    assert 0.0 < ll < 2.0
