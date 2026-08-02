import json

import pandas as pd

import quorabust.cross_encoder_features as cef
from quorabust.cli import main


class _FakeCrossEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs, show_progress_bar: bool = False):
        return [0.9 if idx % 2 else 0.1 for idx, _pair in enumerate(pairs)]


def _write_synthetic_csv(path, n: int = 30) -> None:
    df = pd.DataFrame(
        {
            "question1": [f"how to do task {i}" for i in range(n)],
            "question2": [f"task {i % 7} instructions" for i in range(n)],
            "is_duplicate": [i % 2 for i in range(n)],
        }
    )
    df.to_csv(path, index=False)


def test_cli_trains_and_writes_pkl(tmp_path):
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv)
    out = tmp_path / "model.pkl"
    assert main(["--csv", str(csv), "--out", str(out), "--seed", "3"]) == 0
    assert out.is_file() and out.stat().st_size > 100


def test_cli_writes_metadata_sidecar(tmp_path):
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv)
    out = tmp_path / "model.pkl"
    meta = tmp_path / "model.meta.json"

    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(out),
                "--metadata-out",
                str(meta),
                "--eval-fraction",
                "0",
            ]
        )
        == 0
    )

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["n_train"] > 0
    assert payload["feature_backend"] == "tfidf"
    assert "eval_accuracy" in payload
    assert payload["eval_fraction"] == 0
    assert payload["max_rows"] is None
    assert payload["threshold_candidates"] == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert payload["threshold_metric"] == "f1"
    assert payload["training_command"].startswith("quorabust-train ")


def test_cli_persists_holdout_decision_threshold(tmp_path):
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv, n=80)
    out = tmp_path / "model.pkl"
    meta = tmp_path / "model.meta.json"

    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(out),
                "--metadata-out",
                str(meta),
                "--thresholds",
                "0.3,0.5,0.7",
                "--threshold-metric",
                "f1",
            ]
        )
        == 0
    )

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["decision_threshold"] in {0.3, 0.5, 0.7}
    assert payload["decision_threshold_source"] == "eval_holdout"
    assert payload["decision_threshold_metric"] == "f1"
    assert "f1" in payload["decision_threshold_metrics"]
    assert payload["eval_fraction"] == 0.1
    assert payload["threshold_candidates"] == [0.3, 0.5, 0.7]


def test_cli_rejects_bad_threshold_grid(tmp_path):
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv)
    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(tmp_path / "model.pkl"),
                "--thresholds",
                "0.2,nope",
            ]
        )
        == 1
    )


def test_cli_trains_with_cross_encoder_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(cef, "CrossEncoder", _FakeCrossEncoder)
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv)
    out = tmp_path / "model.pkl"
    meta = tmp_path / "model.meta.json"

    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(out),
                "--feature-backend",
                "cross-encoder",
                "--cross-encoder-model",
                "fake-cross-encoder",
                "--metadata-out",
                str(meta),
                "--eval-fraction",
                "0",
            ]
        )
        == 0
    )

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["feature_backend"] == "cross-encoder"
    assert payload["feature_schema"] == ["cross_score", "len_ratio", "abs_len_diff", "len_sum"]


def test_cli_rejects_bad_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    assert main(["--csv", str(csv), "--out", str(tmp_path / "m.pkl")]) == 1
