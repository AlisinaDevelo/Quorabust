import hashlib
import json

from quorabust.report_validation import main, validate_report_payload


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


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
                "csv_sha256": _digest("train"),
                "eval_csv_sha256": _digest("tuning"),
                "eval_split_source": "explicit_csv",
                "split_strategy": "question_component_holdout",
                "question_id_columns": ["qid1", "qid2"],
                "require_question_ids": True,
                "require_question_text": True,
                "seed": 42,
                "eval_fraction": 0.1,
                "threshold_metric": "f1",
                "calibration_method": "sigmoid",
                "calibration_csv_sha256": _digest("calibration"),
                "threshold_csv_sha256": _digest("tuning"),
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


def _protocol_payload():
    return {
        "schema_version": 1,
        "protocol_name": "quorabust-synthetic-report-binding-v1",
        "evidence_scope": "protocol_only_no_quality_claim",
        "dataset": {
            "name": "synthetic report fixture",
            "source_reference": "synthetic://source.csv",
            "sha256": "c" * 64,
            "license": "synthetic fixture",
            "terms": "no external benchmark claim",
            "raw_data_policy": "external_not_committed",
            "audit": {
                "reference": "synthetic://audit.json",
                "sha256": _digest("audit"),
                "status": "pass",
                "source_sha256": "c" * 64,
                "require_question_ids": True,
                "require_question_text": True,
            },
        },
        "roles": {
            "train": {
                "purpose": "fit model parameters",
                "allowed_activities": ["fit"],
                "artifact": {"reference": "synthetic://train.csv", "sha256": _digest("train")},
            },
            "tuning": {
                "purpose": "select model and threshold policy",
                "allowed_activities": ["model_selection", "threshold_selection"],
                "artifact": {"reference": "synthetic://tuning.csv", "sha256": _digest("tuning")},
            },
            "calibration": {
                "purpose": "fit probability calibration",
                "allowed_activities": ["probability_calibration"],
                "artifact": {
                    "reference": "synthetic://calibration.csv",
                    "sha256": _digest("calibration"),
                },
            },
            "final_holdout": {
                "purpose": "evaluate the frozen release once",
                "allowed_activities": ["final_evaluation"],
                "artifact": {"reference": "synthetic://holdout.csv", "sha256": "b" * 64},
            },
        },
        "split": {
            "strategy": "question_component_holdout",
            "question_id_columns": ["qid1", "qid2"],
            "seed": 42,
            "eval_fraction": 0.1,
            "manifest": {"reference": "synthetic://split.json", "sha256": _digest("split")},
        },
        "decision_policy": {
            "threshold_metric": "f1",
            "threshold_candidates": [0.3, 0.5, 0.7],
            "threshold_source_role": "tuning",
            "calibration_method": "sigmoid",
            "calibration_source_role": "calibration",
            "final_holdout_role": "final_holdout",
        },
        "provenance": {
            "git_revision": "d" * 40,
            "python_version": "3.12.0",
            "dependency_lock": {
                "reference": "requirements.txt",
                "sha256": _digest("requirements"),
            },
            "command": "synthetic report binding",
            "machine": "ci/linux-x86_64",
        },
        "safeguards": {
            "roles_are_disjoint": True,
            "final_holdout_used_for_tuning": False,
            "final_holdout_used_for_calibration": False,
            "final_holdout_used_for_model_selection": False,
            "raw_data_committed": False,
            "public_quality_claims_allowed": False,
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


def test_validate_report_payload_rejects_incomplete_slice_provenance():
    payload = _payload()
    payload["evaluation_manifest"]["slice_provenance"] = {
        "manifest": {
            "schema_version": 1,
            "source": {
                "reference": "synthetic://evaluation.csv",
                "sha256": "a" * 64,
                "rows": 3,
            },
            "columns": {"language": {"labeling_method": "owner labels"}},
        },
        "observed_row_counts": {"language": {"rows": 3, "labels": {"en": 1}}},
    }

    errors = validate_report_payload(payload, require_manifest=True)

    assert (
        "evaluation_manifest.slice_provenance.observed_row_counts.language.labels must sum to rows"
        in errors
    )


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
    lineage["require_question_text"] = False
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
    assert "evaluation_manifest.training_lineage.require_question_text must be true" in errors
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


def test_validate_report_payload_binds_to_protocol():
    assert validate_report_payload(
        _payload(),
        protocol_payload=_protocol_payload(),
    ) == []


def test_validate_report_payload_binds_expected_cost_policy():
    report = _payload()
    protocol = _protocol_payload()
    report["evaluation_manifest"]["training_lineage"].update(
        {
            "threshold_metric": "expected_cost",
            "decision_threshold_costs": {
                "false_positive_cost": 10.0,
                "false_negative_cost": 1.0,
            },
        }
    )
    protocol["decision_policy"].update(
        {
            "threshold_metric": "expected_cost",
            "false_positive_cost": 10.0,
            "false_negative_cost": 1.0,
        }
    )

    assert validate_report_payload(report, protocol_payload=protocol) == []


def test_protocol_binding_requires_holdout_and_calibration():
    report = _payload()
    del report["holdout_evaluation"]
    del report["calibration"]

    errors = validate_report_payload(report, protocol_payload=_protocol_payload())

    assert "missing holdout_evaluation" in errors
    assert "missing calibration" in errors


def test_validate_report_payload_rejects_protocol_mismatches():
    report = _payload()
    protocol = _protocol_payload()
    report["evaluation_manifest"]["evaluation_dataset"]["sha256"] = "e" * 64
    report["evaluation_manifest"]["training_lineage"]["seed"] = 7
    report["evaluation_manifest"]["evaluation_policy"]["thresholds"] = [0.2, 0.5, 0.8]

    errors = validate_report_payload(report, protocol_payload=protocol)

    assert (
        "evaluation_manifest.evaluation_dataset.sha256 must match "
        "protocol.roles.final_holdout.artifact.sha256"
    ) in errors
    assert "evaluation_manifest.training_lineage.seed must match protocol.split.seed" in errors
    assert (
        "evaluation_manifest.evaluation_policy.thresholds must match "
        "protocol.decision_policy.threshold_candidates"
    ) in errors


def test_validate_report_cli_binds_to_protocol(tmp_path, capsys):
    report = tmp_path / "report.json"
    protocol = tmp_path / "protocol.json"
    report.write_text(json.dumps(_payload()), encoding="utf-8")
    protocol.write_text(json.dumps(_protocol_payload()), encoding="utf-8")

    assert main(["--report", str(report), "--protocol", str(protocol)]) == 0
    assert "validated" in capsys.readouterr().out


def test_validate_report_cli_rejects_invalid_protocol_json(tmp_path, capsys):
    report = tmp_path / "report.json"
    protocol = tmp_path / "protocol.json"
    report.write_text(json.dumps(_payload()), encoding="utf-8")
    protocol.write_text("{", encoding="utf-8")

    assert main(["--report", str(report), "--protocol", str(protocol)]) == 1
    assert "Invalid protocol JSON" in capsys.readouterr().err
