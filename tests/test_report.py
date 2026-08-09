import json

import pandas as pd
import pytest

from quorabust.lineage import sha256_file
from quorabust.model import train_duplicate_classifier
from quorabust.persist import save_classifier
from quorabust.report import (
    build_evaluation_manifest,
    build_report_payload,
    calibration_summary,
    evaluate_holdout,
    evaluate_slices,
    main,
    render_comparison_report,
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


def _slice_df():
    return _df().assign(language=["en" if i < 15 else "it" for i in range(30)])


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
        meta={
            "feature_backend": "tfidf",
            "n_train": 20,
            "eval_log_loss": 0.61,
            "calibration_metrics": {
                "raw": {"n": 20, "brier_score": 0.2, "expected_calibration_error": 0.1},
                "calibrated": {
                    "n": 20,
                    "brier_score": 0.18,
                    "expected_calibration_error": 0.05,
                },
            },
        },
    )
    assert "# Quorabust Model Card" in card
    assert "| feature_backend | tfidf |" in card
    assert "| log_loss | 0.6100 |" in card
    assert "## Calibration Fit Comparison" in card


def test_report_excludes_lineage_fields_from_persisted_metrics():
    payload = build_report_payload(
        artifact="model.pkl",
        meta={
            "eval_accuracy": 0.75,
            "eval_fraction": 0.1,
            "eval_csv_sha256": "a" * 64,
        },
    )

    assert payload["persisted_evaluation"] == {"accuracy": 0.75}


def test_build_report_payload_is_machine_readable():
    payload = build_report_payload(
        artifact="model.pkl",
        meta={
            "feature_backend": "tfidf",
            "question_id_columns": ["qid1", "qid2"],
            "eval_accuracy": 0.75,
            "csv": "/private/train.csv",
            "training_command": "quorabust-train --csv /private/train.csv",
            "calibration_metrics": {
                "raw": {"n": 20, "brier_score": 0.2},
                "calibrated": {"n": 20, "brier_score": 0.18},
            },
        },
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
            "calibration": {
                "n_bins": 2,
                "brier_score": 0.12,
                "expected_calibration_error": 0.05,
                "mean_predicted_probability": 0.48,
                "mean_observed_rate": 0.5,
                "bins": [
                    {
                        "lower": 0.0,
                        "upper": 0.5,
                        "count": 5,
                        "mean_predicted_probability": 0.2,
                        "observed_positive_rate": 0.2,
                        "absolute_error": 0.0,
                    }
                ],
            },
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
    assert payload["training_metadata"] == {
        "feature_backend": "tfidf",
        "question_id_columns": ["qid1", "qid2"],
    }
    assert payload["persisted_evaluation"]["accuracy"] == 0.75
    assert payload["confusion_matrix"]["actual_1"]["predicted_1"] == 4
    assert payload["calibration"]["expected_calibration_error"] == 0.05
    assert payload["serving_contract"]["output"]["raw_proba_duplicate"] == "list[float]"
    assert payload["serving_contract"]["output"]["calibrated_proba_duplicate"] == (
        "list[float] | null"
    )
    assert payload["serving_contract"]["output"]["probability_source"] == "raw|calibrated"
    assert payload["threshold_sweep"][0]["f1"] == 0.8
    assert payload["calibration_comparison"]["calibrated"]["brier_score"] == 0.18
    assert "csv" not in payload["training_metadata"]
    assert "training_command" not in payload["training_metadata"]


def test_build_evaluation_manifest_captures_reproducibility_context(tmp_path):
    model, _, _ = _artifact(tmp_path)
    eval_csv = tmp_path / "holdout.csv"
    _df().to_csv(eval_csv, index=False)

    manifest = build_evaluation_manifest(
        artifact_path=model,
        artifact_label="tfidf-v1.pkl",
        eval_path=eval_csv,
        eval_df=_df(),
        meta={
            "git_revision": "train-commit",
            "quorabust_version": "0.3.2",
            "csv_sha256": "train-dataset-hash",
            "feature_backend": "tfidf",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_revision": "embedding-commit",
            "question_id_columns": ["qid1", "qid2"],
            "n_train": 100,
            "seed": 42,
            "split_strategy": "shuffled_prefix_holdout",
            "training_command": "quorabust-train --csv data/raw/train.csv",
        },
        threshold=0.5,
        thresholds=[0.3, 0.5, 0.7],
        calibration_bins=10,
        command="quorabust-report --eval-csv holdout.csv",
    )

    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == {
        "label": "tfidf-v1.pkl",
        "sha256": sha256_file(model),
    }
    assert manifest["evaluation_dataset"]["sha256"] == sha256_file(eval_csv)
    assert manifest["evaluation_dataset"]["rows"] == 30
    assert manifest["evaluation_dataset"]["positive_count"] == 15
    assert manifest["evaluation_dataset"]["positive_rate"] == 0.5
    assert manifest["evaluation_policy"] == {
        "threshold": 0.5,
        "thresholds": [0.3, 0.5, 0.7],
        "calibration_bins": 10,
    }
    assert manifest["training_lineage"]["git_revision"] == "train-commit"
    assert manifest["training_lineage"]["embedding_model_revision"] == "embedding-commit"
    assert manifest["training_lineage"]["question_id_columns"] == ["qid1", "qid2"]
    assert manifest["command"] == "quorabust-report --eval-csv holdout.csv"
    assert manifest["runtime"]["report_git_revision"]
    assert manifest["generated_at_utc"].endswith("Z")


def test_evaluate_holdout_returns_confusion_counts(tmp_path):
    _, builder, clf = _artifact(tmp_path)
    metrics = evaluate_holdout(builder, clf, _df(), threshold=0.5)
    assert metrics["n"] == 30
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == 30
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["calibration"]["brier_score"] <= 1.0
    assert "expected_calibration_error" in metrics["calibration"]


def test_calibration_summary_returns_bins():
    summary = calibration_summary(
        y=pd.Series([0, 0, 1, 1]).to_numpy(),
        proba=pd.Series([0.1, 0.3, 0.7, 0.9]).to_numpy(),
        n_bins=2,
    )
    assert summary["n_bins"] == 2
    assert len(summary["bins"]) == 2
    assert 0.0 <= summary["expected_calibration_error"] <= 1.0


def test_threshold_sweep_returns_tradeoff_rows(tmp_path):
    _, builder, clf = _artifact(tmp_path)
    rows = threshold_sweep(builder, clf, _df(), thresholds=[0.3, 0.5, 0.7])
    assert [row["threshold"] for row in rows] == [0.3, 0.5, 0.7]
    assert all(0.0 <= row["precision"] <= 1.0 for row in rows)
    assert all(0.0 <= row["recall"] <= 1.0 for row in rows)


def test_evaluate_slices_returns_sorted_metrics(tmp_path):
    _, builder, clf = _artifact(tmp_path)

    slices = evaluate_slices(
        builder,
        clf,
        _slice_df(),
        slice_columns=["language"],
        threshold=0.5,
        calibration_bins=5,
    )

    assert list(slices) == ["language"]
    assert [row["value"] for row in slices["language"]] == ["en", "it"]
    assert [row["n"] for row in slices["language"]] == [15, 15]
    assert all(0.0 <= row["positive_rate"] <= 1.0 for row in slices["language"])
    assert all(row["threshold"] == 0.5 for row in slices["language"])
    assert all("log_loss" in row for row in slices["language"])
    assert all("brier_score" in row["calibration"] for row in slices["language"])
    assert all(
        "expected_calibration_error" in row["calibration"] for row in slices["language"]
    )
    for row in slices["language"]:
        uncertainty = row["uncertainty"]
        assert uncertainty["confidence_level"] == 0.95
        assert uncertainty["method"] == "wilson_binomial_interval"
        assert uncertainty["confidence_intervals"]["positive_rate"] is not None
        assert uncertainty["confidence_intervals"]["accuracy"] is not None
        for metric in ("positive_rate", "accuracy", "precision", "recall"):
            interval = uncertainty["confidence_intervals"][metric]
            if interval is None:
                assert metric in {"precision", "recall"}
                continue
            assert interval[0] <= row[metric] <= interval[1]
            assert 0.0 <= interval[0] <= interval[1] <= 1.0
        assert "remain point estimates" in uncertainty["caveat"]


def test_evaluate_slices_marks_degenerate_rate_intervals_unavailable(tmp_path):
    _, builder, clf = _artifact(tmp_path)
    df = _df().assign(language=["negative" if i % 2 == 0 else "positive" for i in range(30)])

    slices = evaluate_slices(
        builder,
        clf,
        df,
        slice_columns=["language"],
        threshold=0.5,
        calibration_bins=5,
    )

    negative = slices["language"][0]
    assert negative["value"] == "negative"
    assert negative["uncertainty"]["confidence_intervals"]["positive_rate"][0] == 0.0
    assert negative["uncertainty"]["confidence_intervals"]["positive_rate"][1] > 0.0
    assert negative["uncertainty"]["confidence_intervals"]["recall"] is None


def test_evaluate_slices_rejects_unsafe_labels_and_cardinality(tmp_path):
    _, builder, clf = _artifact(tmp_path)
    common = {
        "threshold": 0.5,
        "calibration_bins": 5,
    }

    with pytest.raises(ValueError, match="Unknown slice column"):
        evaluate_slices(builder, clf, _df(), slice_columns=["language"], **common)
    with pytest.raises(ValueError, match="missing labels"):
        evaluate_slices(
            builder,
            clf,
            _slice_df().assign(language=[None] + ["en"] * 14 + ["it"] * 15),
            slice_columns=["language"],
            **common,
        )
    with pytest.raises(ValueError, match="blank labels"):
        evaluate_slices(
            builder,
            clf,
            _slice_df().assign(language=[" "] + ["en"] * 14 + ["it"] * 15),
            slice_columns=["language"],
            **common,
        )
    with pytest.raises(ValueError, match="maximum is 1"):
        evaluate_slices(
            builder,
            clf,
            _slice_df(),
            slice_columns=["language"],
            max_slices=1,
            **common,
        )
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_slices(
            builder,
            clf,
            _slice_df(),
            slice_columns=["language", " language "],
            **common,
        )


def test_render_comparison_report_sorts_by_f1():
    report = render_comparison_report(
        [
            {
                "artifact": "tfidf",
                "feature_backend": "tfidf",
                "threshold": 0.5,
                "accuracy": 0.7,
                "precision": 0.7,
                "recall": 0.7,
                "f1": 0.7,
                "roc_auc": 0.75,
                "log_loss": 0.5,
            },
            {
                "artifact": "cross",
                "feature_backend": "cross-encoder",
                "threshold": 0.5,
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1": 0.9,
                "roc_auc": 0.95,
                "log_loss": 0.2,
            },
        ]
    )
    assert report.index("| cross |") < report.index("| tfidf |")
    assert "## Backend Comparison" in report


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
    assert "## Calibration Summary" in card
    assert "## Calibration Bins" in card
    assert "## Threshold Sweep" in card


def test_report_cli_writes_json_payload(tmp_path):
    model, _, _ = _artifact(tmp_path)
    eval_csv = tmp_path / "eval.csv"
    _df().to_csv(eval_csv, index=False)
    out = tmp_path / "MODEL_CARD.json"
    manifest_out = tmp_path / "MODEL_CARD.manifest.json"

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
                "--manifest-out",
                str(manifest_out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact"] == "smoke-model.pkl"
    assert payload["holdout_evaluation"]["n"] == 30
    assert "calibration" in payload
    assert payload["confusion_matrix"]["labels"] == ["not_duplicate", "duplicate"]
    assert len(payload["threshold_sweep"]) == 3
    assert payload["evaluation_manifest"]["evaluation_dataset"]["rows"] == 30
    assert json.loads(manifest_out.read_text(encoding="utf-8")) == payload["evaluation_manifest"]


def test_report_cli_writes_evaluation_slices_to_json_and_markdown(tmp_path):
    model, _, _ = _artifact(tmp_path)
    eval_csv = tmp_path / "eval.csv"
    _slice_df().to_csv(eval_csv, index=False)
    json_out = tmp_path / "MODEL_CARD.json"
    markdown_out = tmp_path / "MODEL_CARD.md"

    assert (
        main(
            [
                "--model",
                str(model),
                "--eval-csv",
                str(eval_csv),
                "--slice-column",
                "language",
                "--max-slices",
                "2",
                "--format",
                "json",
                "--out",
                str(json_out),
            ]
        )
        == 0
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert [row["value"] for row in payload["evaluation_slices"]["language"]] == ["en", "it"]
    assert sum(row["n"] for row in payload["evaluation_slices"]["language"]) == 30
    assert payload["evaluation_slices"]["language"][0]["calibration"]["brier_score"] >= 0
    assert payload["evaluation_manifest"]["evaluation_policy"]["slice_columns"] == [
        "language"
    ]

    assert (
        main(
            [
                "--model",
                str(model),
                "--eval-csv",
                str(eval_csv),
                "--slice-column",
                "language",
                "--out",
                str(markdown_out),
            ]
        )
        == 0
    )
    card = markdown_out.read_text(encoding="utf-8")
    assert "## Evaluation Slices" in card
    assert "Wilson 95% confidence intervals" in card
    assert "| language | en |" in card


def test_report_cli_binds_slice_provenance_manifest(tmp_path):
    model, _, _ = _artifact(tmp_path)
    eval_csv = tmp_path / "eval.csv"
    _slice_df().to_csv(eval_csv, index=False)
    slice_manifest = tmp_path / "slice-manifest.json"
    slice_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "reference": "synthetic://evaluation.csv",
                    "sha256": sha256_file(eval_csv),
                    "rows": 30,
                },
                "columns": {
                    "language": {
                        "labeling_method": "synthetic fixture owner labels",
                        "description": "Controlled CI-only language labels.",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "MODEL_CARD.json"

    assert (
        main(
            [
                "--model",
                str(model),
                "--eval-csv",
                str(eval_csv),
                "--slice-column",
                "language",
                "--slice-manifest",
                str(slice_manifest),
                "--format",
                "json",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    provenance = payload["evaluation_manifest"]["slice_provenance"]
    assert provenance["manifest"]["source"]["sha256"] == sha256_file(eval_csv)
    assert provenance["observed_row_counts"] == {
        "language": {"rows": 30, "labels": {"en": 15, "it": 15}}
    }


def test_report_cli_writes_comparison_json(tmp_path):
    model, _, _ = _artifact(tmp_path)
    eval_csv = tmp_path / "eval.csv"
    _df().to_csv(eval_csv, index=False)
    out = tmp_path / "COMPARE.json"

    assert (
        main(
            [
                "--model",
                str(model),
                "--compare-model",
                f"tfidf={model}",
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
    assert payload["artifact"] == str(model.resolve())
    assert payload["comparison"][0]["artifact"] == "tfidf"
    assert payload["comparison"][0]["feature_backend"] == "tfidf"


def test_report_cli_rejects_comparison_without_eval_csv(tmp_path):
    model, _, _ = _artifact(tmp_path)
    assert main(["--model", str(model), "--compare-model", f"tfidf={model}"]) == 1


def test_report_cli_rejects_bad_threshold(tmp_path):
    model, _, _ = _artifact(tmp_path)
    assert main(["--model", str(model), "--threshold", "1.5"]) == 1


def test_report_cli_rejects_bad_threshold_grid(tmp_path):
    model, _, _ = _artifact(tmp_path)
    assert main(["--model", str(model), "--thresholds", "0.2,nope"]) == 1
