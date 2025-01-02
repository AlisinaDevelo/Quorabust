from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from quorabust.features import PairFeatureBuilder


def train_duplicate_classifier(
    train_df: pd.DataFrame,
    label_col: str = "is_duplicate",
    col_q1: str = "question1",
    col_q2: str = "question2",
    *,
    feature_builder: Any | None = None,
    eval_df: pd.DataFrame | None = None,
    random_state: int = 42,
    xgb_params: dict[str, Any] | None = None,
) -> tuple[Any, XGBClassifier]:
    """
    Fit feature builder on training questions, build pair features, train XGBoost.
    Pass ``feature_builder`` for non-TF–IDF backends (e.g. embeddings).
    If eval_df is provided, uses early stopping on logloss.
    """
    for col in (col_q1, col_q2, label_col):
        if col not in train_df.columns:
            raise KeyError(f"Missing column: {col}")

    builder: Any = feature_builder if feature_builder is not None else PairFeatureBuilder()
    builder.fit_from_frame(train_df, col_q1=col_q1, col_q2=col_q2)
    X_tr = builder.transform_frame(train_df, col_q1=col_q1, col_q2=col_q2)
    y_tr = train_df[label_col].astype(int).to_numpy()

    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if eval_df is not None:
        params.setdefault("early_stopping_rounds", 20)
    if xgb_params:
        params.update(xgb_params)

    clf = XGBClassifier(**params)

    if eval_df is not None:
        X_ev = builder.transform_frame(eval_df, col_q1=col_q1, col_q2=col_q2)
        y_ev = eval_df[label_col].astype(int).to_numpy()
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_ev, y_ev)],
            verbose=False,
        )
    else:
        clf.fit(X_tr, y_tr)

    return builder, clf


def predict_proba_duplicate(
    builder: Any,
    clf: XGBClassifier,
    q1: list[str],
    q2: list[str],
) -> np.ndarray:
    """Probability of duplicate for each pair (shape (n, 2) for sklearn convention)."""
    X = builder.transform_pairs(q1, q2)
    return clf.predict_proba(X)


def eval_log_loss(
    builder: Any,
    clf: XGBClassifier,
    df: pd.DataFrame,
    label_col: str = "is_duplicate",
    col_q1: str = "question1",
    col_q2: str = "question2",
) -> float:
    X = builder.transform_frame(df, col_q1=col_q1, col_q2=col_q2)
    y = df[label_col].astype(int).to_numpy()
    proba = clf.predict_proba(X)[:, 1]
    return float(log_loss(y, proba))


def eval_classification_metrics(
    builder: Any,
    clf: XGBClassifier,
    df: pd.DataFrame,
    label_col: str = "is_duplicate",
    col_q1: str = "question1",
    col_q2: str = "question2",
) -> dict[str, float]:
    """Accuracy, log loss, and ROC-AUC when both classes are present."""
    X = builder.transform_frame(df, col_q1=col_q1, col_q2=col_q2)
    y = df[label_col].astype(int).to_numpy()
    proba = clf.predict_proba(X)[:, 1]
    y_hat = (proba >= 0.5).astype(int)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y, y_hat)),
        "log_loss": float(log_loss(y, proba)),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    return out
