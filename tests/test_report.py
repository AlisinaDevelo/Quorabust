import pandas as pd

from quorabust.model import train_duplicate_classifier
from quorabust.persist import save_classifier
from quorabust.report import evaluate_holdout, main, render_model_card


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


def test_evaluate_holdout_returns_confusion_counts(tmp_path):
    _, builder, clf = _artifact(tmp_path)
    metrics = evaluate_holdout(builder, clf, _df(), threshold=0.5)
    assert metrics["n"] == 30
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == 30
    assert 0.0 <= metrics["accuracy"] <= 1.0


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


def test_report_cli_rejects_bad_threshold(tmp_path):
    model, _, _ = _artifact(tmp_path)
    assert main(["--model", str(model), "--threshold", "1.5"]) == 1
