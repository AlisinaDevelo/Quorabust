import pandas as pd

from quorabust.model import eval_classification_metrics, train_duplicate_classifier


def test_eval_classification_metrics_has_auc():
    df = pd.DataFrame(
        {
            "question1": [f"q1 {i}" for i in range(20)],
            "question2": [f"q2 {i % 4}" for i in range(20)],
            "is_duplicate": [i % 2 for i in range(20)],
        }
    )
    b, clf = train_duplicate_classifier(df, xgb_params={"n_estimators": 20, "max_depth": 3})
    m = eval_classification_metrics(b, clf, df)
    assert "accuracy" in m and "log_loss" in m and "roc_auc" in m
    assert 0.0 <= m["roc_auc"] <= 1.0
