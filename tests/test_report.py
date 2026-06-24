import json

import pandas as pd

from quorabust.model import train_duplicate_classifier
from quorabust.persist import save_classifier
from quorabust.report import (
    build_report_payload,
    evaluate_holdout,
    main,
    render_model_card,
    threshold_sweep,
)


def _df():
    return pd.DataFrame(
        {
            "question1": [f"how to solve task {i}" for i in range(30)],
            "question2": [f"task {i % 5} solution steps" for i in range(30)],
            "is_duplicate": [i % 2 for i in range(30)],
        }
    )


def _artifact(tmp_path):
    df = _df()
    builder, clf = train_duplicate_classifier(
        df,
        xgb_params={"n_estimators": 12, "max_depth": 2},
    )
    model = tmp_path / "model.pkl"
    save_classifier(
        model,
        builder,
        clf,
        meta={
            "feature_backend": "tfidf",
            "feature_schema": ["cos", "jaccard", "len_ratio", "abs_len_diff", "len_sum"],
            "n_train": len(df),
            "eval_accuracy": 0.75,
            "git_revision": "abc123",
        },
    )
    return model, builder, clf


def test_render_model_card_includes_metadata_and_persisted_metrics():
    card = render_model_card(
        artifact="/tmp/model.pkl",
        meta={"feature_backend": "tfidf", "n_train": 20, "eval_log_loss": 0.61},
    )
    assert "# Quorabust Model Card" in card
    assert "| feature_backend | tfidf |" in card
    assert "| log_loss | 0.6100 |" in card


def test_build_report_payload_is_machine_readable():
    payload = build_report_payload(
        artifact="model.pkl",
        meta={"feature_backend": "tfidf", "eval_accuracy": 0.75, "csv": "/private/train.csv"},
        holdout_metrics={
            "n": 10,
            "threshold": 0.5,
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "tn": 4,
            "fp": 1,
            "fn": 1,
            "tp": 4,
        },
        sweep_metrics=[
            {
                "threshold": 0.5,
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "tn": 4,
                "fp": 1,
                "fn": 1,
                "tp": 4,
                "predicted_positive_rate": 0.5,
            }
        ],
    )
    assert payload["artifact"] == "model.pkl"
    assert payload["training_metadata"] == {"feature_backend": "tfidf"}
    assert payload["persisted_evaluation"]["accuracy"] == 0.75
    assert payload["confusion_matrix"]["actual_1"]["predicted_1"] == 4
    assert payload["threshold_sweep"][0]["f1"] == 0.8
    assert "csv" not in payload["training_metadata"]


def test_evaluate_holdout_returns_confusion_counts(tmp_path):
    _, builder, clf = _artifact(tmp_path)
    metrics = evaluate_holdout(builder, clf, _df(), threshold=0.5)
    assert metrics["n"] == 30
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == 30
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_threshold_sweep_returns_tradeoff_rows(tmp_path):
    _, builder, clf = _artifact(tmp_path)
    rows = threshold_sweep(builder, clf, _df(), thresholds=[0.3, 0.5, 0.7])
    assert [row["threshold"] for row in rows] == [0.3, 0.5, 0.7]
    assert all(0.0 <= row["precision"] <= 1.0 for row in rows)
    assert all(0.0 <= row["recall"] <= 1.0 for row in rows)


def test_report_cli_writes_model_card(tmp_path):
    model, _, _ = _artifact(tmp_path)
    eval_csv = tmp_path / "eval.csv"
    _df().to_csv(eval_csv, index=False)
    out = tmp_path / "MODEL_CARD.md"

    assert (
        main(
            [
                "--model",
                str(model),
                "--artifact-label",
                "smoke-model.pkl",
                "--eval-csv",
                str(eval_csv),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    card = out.read_text(encoding="utf-8")
    assert "| artifact | smoke-model.pkl |" in card
    assert "## Holdout Evaluation" in card
    assert "## Confusion Matrix" in card
    assert "## Threshold Sweep" in card


def test_report_cli_writes_json_payload(tmp_path):
    model, _, _ = _artifact(tmp_path)
    eval_csv = tmp_path / "eval.csv"
    _df().to_csv(eval_csv, index=False)
    out = tmp_path / "MODEL_CARD.json"

    assert (
        main(
            [
                "--model",
                str(model),
                "--artifact-label",
                "smoke-model.pkl",
                "--eval-csv",
                str(eval_csv),
                "--format",
                "json",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact"] == "smoke-model.pkl"
    assert payload["holdout_evaluation"]["n"] == 30
    assert payload["confusion_matrix"]["labels"] == ["not_duplicate", "duplicate"]
    assert len(payload["threshold_sweep"]) == 3


def test_report_cli_rejects_bad_threshold(tmp_path):
    model, _, _ = _artifact(tmp_path)
    assert main(["--model", str(model), "--threshold", "1.5"]) == 1


def test_report_cli_rejects_bad_threshold_grid(tmp_path):
    model, _, _ = _artifact(tmp_path)
    assert main(["--model", str(model), "--thresholds", "0.2,nope"]) == 1
