import copy
import hashlib
import json

from quorabust.benchmark_protocol import main, validate_protocol_payload


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _artifact(name: str) -> dict[str, str]:
    return {"reference": f"external/{name}.csv", "sha256": _digest(name)}


def _payload() -> dict:
    source_sha256 = _digest("synthetic-source")
    return {
        "schema_version": 1,
        "protocol_name": "quorabust-synthetic-smoke-v1",
        "evidence_scope": "protocol_only_no_quality_claim",
        "dataset": {
            "name": "synthetic smoke fixture",
            "source_reference": "examples/smoke_pairs.csv",
            "sha256": source_sha256,
            "license": "repository test fixture",
            "terms": "synthetic data; no external benchmark claim",
            "raw_data_policy": "external_not_committed",
            "audit": {
                "reference": "external/smoke-audit.json",
                "sha256": _digest("smoke-audit"),
                "status": "pass",
                "source_sha256": source_sha256,
                "require_question_ids": True,
                "require_question_text": True,
            },
        },
        "roles": {
            "train": {
                "purpose": "fit model parameters",
                "allowed_activities": ["fit"],
                "artifact": _artifact("train"),
            },
            "tuning": {
                "purpose": "select model and threshold policy",
                "allowed_activities": ["model_selection", "threshold_selection"],
                "artifact": _artifact("tuning"),
            },
            "calibration": {
                "purpose": "fit the probability calibration mapping",
                "allowed_activities": ["probability_calibration"],
                "artifact": _artifact("calibration"),
            },
            "final_holdout": {
                "purpose": "evaluate the frozen release once",
                "allowed_activities": ["final_evaluation"],
                "artifact": _artifact("final-holdout"),
            },
        },
        "split": {
            "strategy": "question_component_holdout",
            "question_id_columns": ["qid1", "qid2"],
            "seed": 42,
            "eval_fraction": 0.1,
            "manifest": {
                "reference": "external/smoke-split.json",
                "sha256": _digest("smoke-split"),
            },
        },
        "decision_policy": {
            "threshold_metric": "f1",
            "threshold_candidates": [0.2, 0.5, 0.8],
            "threshold_source_role": "tuning",
            "calibration_method": "sigmoid",
            "calibration_source_role": "calibration",
            "final_holdout_role": "final_holdout",
        },
        "provenance": {
            "git_revision": "a" * 40,
            "python_version": "3.12.0",
            "dependency_lock": {
                "reference": "requirements.txt",
                "sha256": _digest("requirements"),
            },
            "command": "quorabust-train --csv external/train.csv --seed 42",
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


def test_validate_protocol_payload_accepts_complete_contract():
    assert validate_protocol_payload(_payload()) == []


def test_validate_protocol_payload_accepts_expected_cost_policy():
    payload = _payload()
    payload["decision_policy"].update(
        {
            "threshold_metric": "expected_cost",
            "false_positive_cost": 10.0,
            "false_negative_cost": 1.0,
        }
    )

    assert validate_protocol_payload(payload) == []


def test_validate_protocol_payload_rejects_incomplete_expected_cost_policy():
    payload = _payload()
    payload["decision_policy"]["threshold_metric"] = "expected_cost"
    payload["decision_policy"]["false_positive_cost"] = 10.0

    errors = validate_protocol_payload(payload)

    assert (
        "decision_policy expected_cost requires: false_negative_cost" in errors
    )

    payload["decision_policy"]["false_negative_cost"] = 0.0
    payload["decision_policy"]["false_positive_cost"] = 0.0
    errors = validate_protocol_payload(payload)
    assert "decision_policy requires at least one positive threshold cost" in errors


def test_validate_protocol_payload_rejects_leakage_and_provenance_breaks():
    payload = _payload()
    payload["dataset"]["audit"]["require_question_ids"] = False
    payload["roles"]["final_holdout"]["artifact"]["sha256"] = payload["roles"]["train"][
        "artifact"
    ]["sha256"]
    payload["split"]["strategy"] = "shuffled_row_holdout"
    payload["decision_policy"]["threshold_source_role"] = "final_holdout"
    payload["safeguards"]["final_holdout_used_for_model_selection"] = True
    payload["provenance"]["git_revision"] = "working-tree"

    errors = validate_protocol_payload(payload)

    assert "dataset.audit.require_question_ids must be true" in errors
    assert "split.strategy must be question_component_holdout" in errors
    assert "decision_policy.threshold_source_role must be tuning" in errors
    assert "safeguards.final_holdout_used_for_model_selection must be false" in errors
    assert "provenance.git_revision must be a 40-character hexadecimal commit SHA" in errors
    assert any("shared by: final_holdout, train" in error for error in errors)


def test_validate_protocol_payload_rejects_missing_roles_and_duplicate_thresholds():
    payload = _payload()
    del payload["roles"]["calibration"]
    payload["decision_policy"]["threshold_candidates"] = [0.5, 0.5]
    payload["split"]["question_id_columns"] = ["qid1", "qid1"]

    errors = validate_protocol_payload(payload)

    assert "missing roles field: calibration" in errors
    assert "decision_policy.threshold_candidates must not contain duplicates" in errors
    assert "split.question_id_columns is missing: qid2" in errors


def test_validate_protocol_cli_passes(tmp_path, capsys):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_payload()), encoding="utf-8")

    assert main(["--protocol", str(protocol)]) == 0
    assert "validated" in capsys.readouterr().out


def test_validate_protocol_cli_fails_for_invalid_json(tmp_path, capsys):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{", encoding="utf-8")

    assert main(["--protocol", str(protocol)]) == 1
    assert "Invalid protocol JSON" in capsys.readouterr().err


def test_validate_protocol_cli_fails_for_policy_error(tmp_path, capsys):
    payload = copy.deepcopy(_payload())
    payload["dataset"]["sha256"] = "invalid"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--protocol", str(protocol)]) == 1
    assert "dataset.sha256 must be a 64-character hexadecimal SHA-256 digest" in (
        capsys.readouterr().err
    )
