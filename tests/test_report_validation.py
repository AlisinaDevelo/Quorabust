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
    }


def test_validate_report_payload_accepts_release_report():
    assert validate_report_payload(
        _payload(),
        require_holdout=True,
        require_calibration=True,
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
    assert main(["--report", str(report), "--require-holdout", "--require-calibration"]) == 0


def test_validate_report_cli_fails_for_missing_holdout(tmp_path):
    payload = _payload()
    del payload["holdout_evaluation"]
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--report", str(report), "--require-holdout"]) == 1
