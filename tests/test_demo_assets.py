import json
from pathlib import Path

from quorabust.demo_assets import build_demo_assets, main


def test_build_demo_assets_writes_api_snapshots(tmp_path):
    out = tmp_path / "assets"
    written = build_demo_assets(
        csv_path=Path(__file__).resolve().parents[1] / "examples" / "smoke_pairs.csv",
        out_dir=out,
    )

    names = {path.name for path in written}
    assert {
        "README.md",
        "models-response.json",
        "openapi-excerpt.json",
        "predict-request.json",
        "predict-response.json",
    } <= names
    predict = json.loads((out / "predict-response.json").read_text(encoding="utf-8"))
    assert predict["variant"] == "a"
    assert "decision_threshold" in predict
    assert predict["features"]

    models = json.loads((out / "models-response.json").read_text(encoding="utf-8"))
    assert models["variants"]["a"]["feature_backend"] == "tfidf"


def test_demo_assets_cli_fails_for_missing_csv(tmp_path, capsys):
    code = main(["--csv", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "assets")])

    assert code == 1
    assert "demo CSV not found" in capsys.readouterr().err
