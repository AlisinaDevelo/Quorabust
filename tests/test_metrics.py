import pandas as pd

from quorabust.model import (
    eval_classification_metrics,
    select_decision_threshold,
    train_duplicate_classifier,
)


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


def test_eval_classification_metrics_handles_one_class_eval():
    df = pd.DataFrame(
        {
            "question1": [f"q1 {i}" for i in range(20)],
            "question2": [f"q2 {i % 4}" for i in range(20)],
            "is_duplicate": [i % 2 for i in range(20)],
        }
    )
    b, clf = train_duplicate_classifier(df, xgb_params={"n_estimators": 20, "max_depth": 3})
    one_class = df[df["is_duplicate"] == 1].head(5).copy()
    m = eval_classification_metrics(b, clf, one_class)
    assert "accuracy" in m and "log_loss" in m
    assert "roc_auc" not in m


def test_select_decision_threshold_optimizes_requested_metric():
    y = pd.Series([0, 0, 1, 1]).to_numpy()
    proba = pd.Series([0.1, 0.4, 0.6, 0.9]).to_numpy()
    selected = select_decision_threshold(
        y,
        proba,
        thresholds=[0.3, 0.5, 0.7],
        optimize_for="f1",
    )
    assert selected["threshold"] == 0.5
    assert selected["f1"] == 1.0
