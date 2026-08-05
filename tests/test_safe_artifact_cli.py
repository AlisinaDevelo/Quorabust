import json

import pandas as pd

from quorabust.model import train_duplicate_classifier
from quorabust.persist import save_classifier
from quorabust.safe_artifact_cli import main


def test_safe_export_cli_writes_qmodel_and_path_light_sidecar(tmp_path):
    frame = pd.DataFrame(
        {
            "question1": ["hello world", "foo bar", "what is python", "cache api"],
            "question2": ["hello there", "baz qux", "python language", "api cache"],
            "is_duplicate": [1, 0, 1, 1],
        }
    )
    builder, classifier = train_duplicate_classifier(
        frame,
        xgb_params={"n_estimators": 8, "max_depth": 2},
    )
    source = tmp_path / "source.pkl"
    output = tmp_path / "exported.qmodel"
    sidecar = tmp_path / "exported.meta.json"
    save_classifier(source, builder, classifier, meta={"csv": "/private/train.csv"})

    assert (
        main(
            [
                "--model",
                str(source),
                "--out",
                str(output),
                "--metadata-out",
                str(sidecar),
            ]
        )
        == 0
    )

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["artifact_format"] == "quorabust.safe.tfidf_xgboost"
    assert payload["artifact_sha256"]
    assert payload["source_artifact_sha256"]
    assert "csv" not in payload
