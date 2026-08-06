import json

from quorabust.report_validation import main, validate_report_payload


def _payload():
    return {
        "artifact": "model.pkl",
        "generated_by": "Quorabust 0.3.2",
        "intended_use": "score pairs",
        "training_metadata": {
            "feature_backend": "tfidf",
            "feature_schema": ["cos", "jaccard"],
        },
        "persisted_evaluation": {"accuracy": 0.8},
        "serving_contract": {
            "predict": "POST /predict",
            "metrics": "GET /metrics",
            "input": {"question1": "list[str]", "question2": "list[str]"},
            "output": {
                "proba_duplicate": "list[float]",
                "is_duplicate": "list[bool]",
                "decision_threshold": "float",
            },
        },
        "caveats": ["Use a current holdout."],
        "holdout_evaluation": {
            "n": 10,
            "threshold": 0.5,
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "log_loss": 0.4,
            "positive_rate": 0.5,
            "predicted_positive_rate": 0.5,
        },
        "calibration": {
            "n_bins": 5,
            "brier_score": 0.2,
            "expected_calibration_error": 0.1,
            "mean_predicted_probability": 0.5,
            "mean_observed_rate": 0.5,
            "bins": [
                {
                    "lower": 0.0,
                    "upper": 0.2,
                    "count": 2,
                    "mean_predicted_probability": 0.1,
                    "observed_positive_rate": 0.0,
                    "absolute_error": 0.1,
                }
            ],
        },
        "evaluation_manifest": {
            "schema_version": 1,
            "artifact": {"label": "model.pkl", "sha256": "a" * 64},
            "evaluation_dataset": {
                "sha256": "b" * 64,
                "rows": 10,
            "columns": ["question1", "question2", "is_duplicate", "qid1", "qid2"],
                "positive_count": 5,
                "positive_rate": 0.5,
            },
            "evaluation_policy": {
                "threshold": 0.5,
                "thresholds": [0.3, 0.5, 0.7],
                "calibration_bins": 5,
            },
            "training_lineage": {
                "git_revision": "abc123",
                "split_strategy": "question_component_holdout",
                "question_id_columns": ["qid1", "qid2"],
                "require_question_ids": True,
            },
            "runtime": {
                "python_version": "3.12.0",
                "system": "Darwin",
                "machine": "arm64",
                "report_git_revision": "def456",
            },
            "command": "quorabust-report --eval-csv holdout.csv",
            "generated_at_utc": "2026-08-02T12:00:00Z",
        },
    }


def test_validate_report_payload_accepts_release_report():
    assert validate_report_payload(
        _payload(),
        require_holdout=True,
        require_calibration=True,
        require_manifest=True,
        require_question_component_split=True,
    ) == []


def test_validate_report_payload_reports_missing_fields():
    payload = _payload()
    del payload["serving_contract"]["output"]["is_duplicate"]
    del payload["calibration"]["bins"]

    errors = validate_report_payload(payload, require_calibration=True)

    assert "missing serving_contract.output field: is_duplicate" in errors
    assert "missing calibration field: bins" in errors


def test_validate_report_cli_passes(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_payload()), encoding="utf-8")
    assert (
        main(
            [
                "--report",
                str(report),
                "--require-holdout",
                "--require-calibration",
                "--require-manifest",
                "--require-question-component-split",
            ]
        )
        == 0
    )


def test_validate_report_cli_fails_for_missing_holdout(tmp_path):
    payload = _payload()
    del payload["holdout_evaluation"]
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--report", str(report), "--require-holdout"]) == 1


def test_validate_report_payload_reports_missing_manifest():
    payload = _payload()
    del payload["evaluation_manifest"]

    assert "missing evaluation_manifest" in validate_report_payload(
        payload,
        require_manifest=True,
    )


def test_validate_report_payload_rejects_row_level_fallback_when_required():
    payload = _payload()
    lineage = payload["evaluation_manifest"]["training_lineage"]
    lineage["split_strategy"] = "shuffled_prefix_holdout"
    lineage["require_question_ids"] = False
    payload["evaluation_manifest"]["evaluation_dataset"]["columns"] = [
        "question1",
        "question2",
        "is_duplicate",
    ]

    errors = validate_report_payload(
        payload,
        require_question_component_split=True,
    )

    assert (
        "evaluation_manifest.training_lineage.split_strategy must be "
        "question_component_holdout"
    ) in errors
    assert "evaluation_manifest.training_lineage.require_question_ids must be true" in errors
    assert (
        "evaluation_manifest.evaluation_dataset.columns is missing: qid1, qid2"
    ) in errors


def test_validate_report_payload_requires_manifest_for_component_policy():
    payload = _payload()
    del payload["evaluation_manifest"]

    assert validate_report_payload(
        payload,
        require_question_component_split=True,
    ) == ["missing evaluation_manifest"]
