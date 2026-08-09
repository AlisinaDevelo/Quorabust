import json

import pandas as pd

import quorabust.cross_encoder_features as cef
from quorabust.cli import main
from quorabust.lineage import sha256_file
from quorabust.registry import load_model_records
from quorabust.split import split_train_eval


class _FakeCrossEncoder:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def predict(self, pairs, batch_size: int = 32, show_progress_bar: bool = False):
        assert batch_size == 8
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
    registry = tmp_path / "registry"

    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(out),
                "--metadata-out",
                str(meta),
                "--registry-dir",
                str(registry),
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
    assert payload["artifact_sha256"] == sha256_file(out)
    assert load_model_records(registry)[0]["artifact_sha256"] == sha256_file(out)


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


def test_cli_persists_expected_cost_policy(tmp_path):
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv, n=80)
    out = tmp_path / "model.pkl"
    meta = tmp_path / "model.meta.json"
    registry = tmp_path / "registry"

    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(out),
                "--metadata-out",
                str(meta),
                "--registry-dir",
                str(registry),
                "--thresholds",
                "0.3,0.5,0.7",
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

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["decision_threshold_metric"] == "expected_cost"
    assert payload["decision_threshold_costs"] == {
        "false_positive_cost": 10.0,
        "false_negative_cost": 1.0,
    }
    assert {"expected_cost", "false_positives", "false_negatives"}.issubset(
        payload["decision_threshold_metrics"]
    )
    record = load_model_records(registry)[0]
    assert record["decision_threshold_costs"] == payload["decision_threshold_costs"]


def test_cli_exports_the_exact_holdout_and_records_its_hash(tmp_path):
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv, n=80)
    out = tmp_path / "model.pkl"
    eval_out = tmp_path / "holdout.csv"
    meta = tmp_path / "model.meta.json"
    registry = tmp_path / "registry"

    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(out),
                "--eval-out",
                str(eval_out),
                "--metadata-out",
                str(meta),
                "--registry-dir",
                str(registry),
                "--seed",
                "19",
            ]
        )
        == 0
    )

    holdout = pd.read_csv(eval_out)
    _, expected, _ = split_train_eval(pd.read_csv(csv), eval_fraction=0.1, seed=19)
    assert expected is not None
    pd.testing.assert_frame_equal(holdout, expected)
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["eval_csv_sha256"] == sha256_file(eval_out)
    assert payload["n_eval"] == len(holdout)
    record = load_model_records(registry)[0]
    assert record["eval_csv_sha256"] == sha256_file(eval_out)
    assert "eval_csv_sha256" not in record["eval_metrics"]


def test_cli_uses_question_component_holdout_when_ids_are_available(tmp_path):
    n = 40
    csv = tmp_path / "train-with-qids.csv"
    pd.DataFrame(
        {
            "qid1": list(range(1, n * 2, 2)),
            "qid2": list(range(2, n * 2 + 1, 2)),
            "question1": [f"how to do task {i}" for i in range(n)],
            "question2": [f"task {i} instructions" for i in range(n)],
            "is_duplicate": [i % 2 for i in range(n)],
        }
    ).to_csv(csv, index=False)
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
                "--require-question-ids",
                "--require-question-text",
            ]
        )
        == 0
    )

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["split_strategy"] == "question_component_holdout"
    assert payload["require_question_ids"] is True
    assert payload["require_question_text"] is True
    assert payload["n_eval"] > 0


def test_cli_accepts_explicit_disjoint_evaluation_csv(tmp_path):
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "tuning.csv"
    out = tmp_path / "model.pkl"
    meta = tmp_path / "model.meta.json"

    def write_role(path, start, count):
        pd.DataFrame(
            {
                "qid1": [start + index * 2 for index in range(count)],
                "qid2": [start + index * 2 + 1 for index in range(count)],
                "question1": [f"question one {index}" for index in range(count)],
                "question2": [f"question two {index}" for index in range(count)],
                "is_duplicate": [index % 2 for index in range(count)],
            }
        ).to_csv(path, index=False)

    write_role(train_csv, 1, 40)
    write_role(eval_csv, 1001, 20)

    assert (
        main(
            [
                "--csv",
                str(train_csv),
                "--eval-csv",
                str(eval_csv),
                "--eval-fraction",
                "0.1",
                "--out",
                str(out),
                "--metadata-out",
                str(meta),
                "--require-question-ids",
                "--require-question-text",
            ]
        )
        == 0
    )

    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["n_train"] == 40
    assert payload["n_eval"] == 20
    assert payload["eval_csv_sha256"] == sha256_file(eval_csv)
    assert payload["eval_split_source"] == "explicit_csv"
    assert payload["split_strategy"] == "question_component_holdout"


def test_cli_rejects_overlapping_explicit_evaluation_csv(tmp_path, capsys):
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "tuning.csv"
    frame = pd.DataFrame(
        {
            "qid1": ["q1", "q3"],
            "qid2": ["q2", "q4"],
            "question1": ["a", "b"],
            "question2": ["c", "d"],
            "is_duplicate": [0, 1],
        }
    )
    frame.to_csv(train_csv, index=False)
    frame.to_csv(eval_csv, index=False)

    assert (
        main(
            [
                "--csv",
                str(train_csv),
                "--eval-csv",
                str(eval_csv),
                "--out",
                str(tmp_path / "model.pkl"),
                "--require-question-ids",
            ]
        )
        == 1
    )
    assert "question IDs overlap" in capsys.readouterr().err


def test_cli_can_require_question_ids_for_benchmark_runs(tmp_path):
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv)

    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(tmp_path / "model.pkl"),
                "--require-question-ids",
            ]
        )
        == 1
    )


def test_cli_can_require_question_text_for_benchmark_runs(tmp_path, capsys):
    csv = tmp_path / "train.csv"
    _write_synthetic_csv(csv)
    frame = pd.read_csv(csv)
    frame.loc[0, "question2"] = " "
    frame.to_csv(csv, index=False)

    assert (
        main(
            [
                "--csv",
                str(csv),
                "--out",
                str(tmp_path / "model.pkl"),
                "--require-question-text",
            ]
        )
        == 1
    )
    assert "--require-question-text" in capsys.readouterr().err


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
                "--cross-encoder-model-revision",
                "abc123",
                "--cross-encoder-batch-size",
                "8",
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
    assert payload["cross_encoder_model_revision"] == "abc123"
    assert payload["cross_encoder_batch_size"] == 8


def test_cli_rejects_bad_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    assert main(["--csv", str(csv), "--out", str(tmp_path / "m.pkl")]) == 1
