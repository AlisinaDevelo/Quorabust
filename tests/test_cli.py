import json

import pandas as pd

from quorabust.cli import main


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


def test_cli_rejects_bad_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    assert main(["--csv", str(csv), "--out", str(tmp_path / "m.pkl")]) == 1
