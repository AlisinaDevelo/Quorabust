import json

import pandas as pd
from starlette.testclient import TestClient

from quorabust.calibration_cli import main as calibrate_main
from quorabust.cli import main as train_main
from quorabust.lineage import sha256_file
from quorabust.persist import load_classifier
from quorabust.serve import create_app


def _write_pairs(path, start: int, count: int) -> None:
    pd.DataFrame(
        {
            "question1": [f"how to do task {i}" for i in range(start, start + count)],
            "question2": [f"task {i} instructions" for i in range(start, start + count)],
            "is_duplicate": [i % 2 for i in range(start, start + count)],
        }
    ).to_csv(path, index=False)


def _train_base(tmp_path):
    train_csv = tmp_path / "train.csv"
    model = tmp_path / "base.pkl"
    _write_pairs(train_csv, 0, 80)
    assert (
        train_main(
            [
                "--csv",
                str(train_csv),
                "--out",
                str(model),
                "--eval-fraction",
                "0",
                "--metadata-out",
                str(tmp_path / "base.meta.json"),
            ]
        )
        == 0
    )
    return train_csv, model


def test_calibrate_cli_writes_artifact_metadata_and_registry(tmp_path):
    train_csv, model = _train_base(tmp_path)
    calibration_csv = tmp_path / "calibration.csv"
    threshold_csv = tmp_path / "threshold.csv"
    calibrated = tmp_path / "calibrated.pkl"
    metadata = tmp_path / "calibrated.meta.json"
    registry = tmp_path / "registry"
    _write_pairs(calibration_csv, 100, 40)
    _write_pairs(threshold_csv, 200, 40)

    assert (
        calibrate_main(
            [
                "--model",
                str(model),
                "--calibration-csv",
                str(calibration_csv),
                "--threshold-csv",
                str(threshold_csv),
                "--out",
                str(calibrated),
                "--metadata-out",
                str(metadata),
                "--registry-dir",
                str(registry),
                "--calibration-method",
                "isotonic",
                "--thresholds",
                "0.3,0.5,0.7",
            ]
        )
        == 0
    )

    assert train_csv.is_file()
    assert calibrated.is_file()
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["calibration_method"] == "isotonic"
    assert payload["calibration_csv_sha256"] == sha256_file(calibration_csv)
    assert payload["threshold_csv_sha256"] == sha256_file(threshold_csv)
    assert payload["calibrated_from_artifact_sha256"] == sha256_file(model)
    assert payload["artifact_sha256"] == sha256_file(calibrated)
    assert payload["decision_threshold_source"] == "calibration_threshold_csv"
    assert set(payload["calibration_metrics"]) == {"raw", "calibrated"}

    _, classifier, meta = load_classifier(calibrated)
    assert classifier.calibrator.method == "isotonic"
    assert meta["calibration_method"] == "isotonic"
    record = json.loads((registry / "models.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["artifact_sha256"] == sha256_file(calibrated)


def test_calibrate_cli_persists_expected_cost_policy(tmp_path):
    _train_base(tmp_path)
    calibration_csv = tmp_path / "calibration.csv"
    threshold_csv = tmp_path / "threshold.csv"
    calibrated = tmp_path / "calibrated.pkl"
    metadata = tmp_path / "calibrated.meta.json"
    registry = tmp_path / "registry"
    _write_pairs(calibration_csv, 100, 40)
    _write_pairs(threshold_csv, 200, 40)

    assert (
        calibrate_main(
            [
                "--model",
                str(tmp_path / "base.pkl"),
                "--calibration-csv",
                str(calibration_csv),
                "--threshold-csv",
                str(threshold_csv),
                "--out",
                str(calibrated),
                "--metadata-out",
                str(metadata),
                "--registry-dir",
                str(registry),
                "--threshold-metric",
                "expected_cost",
                "--false-positive-cost",
                "10",
                "--false-negative-cost",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["decision_threshold_metric"] == "expected_cost"
    assert payload["decision_threshold_costs"] == {
        "false_positive_cost": 10.0,
        "false_negative_cost": 1.0,
    }
    record = json.loads((registry / "models.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["decision_threshold_costs"] == payload["decision_threshold_costs"]


def test_calibrated_metadata_is_safe_and_served(tmp_path):
    _train_base(tmp_path)
    calibration_csv = tmp_path / "calibration.csv"
    threshold_csv = tmp_path / "threshold.csv"
    calibrated = tmp_path / "calibrated.pkl"
    _write_pairs(calibration_csv, 100, 40)
    _write_pairs(threshold_csv, 200, 40)
    assert (
        calibrate_main(
            [
                "--model",
                str(tmp_path / "base.pkl"),
                "--calibration-csv",
                str(calibration_csv),
                "--threshold-csv",
                str(threshold_csv),
                "--out",
                str(calibrated),
            ]
        )
        == 0
    )

    with TestClient(create_app(model_path_a=str(calibrated))) as client:
        response = client.get("/models")

    assert response.status_code == 200
    public = response.json()["variants"]["a"]
    assert public["calibration_method"] == "sigmoid"
    assert public["n_calibration"] == 40
    assert public["n_threshold"] == 40
    assert "calibration_command" not in public
    assert "csv" not in public


def test_calibrate_cli_rejects_training_csv_reuse(tmp_path, capsys):
    train_csv, model = _train_base(tmp_path)
    threshold_csv = tmp_path / "threshold.csv"
    _write_pairs(threshold_csv, 200, 40)

    code = calibrate_main(
        [
            "--model",
            str(model),
            "--calibration-csv",
            str(train_csv),
            "--threshold-csv",
            str(threshold_csv),
            "--out",
            str(tmp_path / "calibrated.pkl"),
        ]
    )

    assert code == 1
    assert "must not reuse" in capsys.readouterr().err
