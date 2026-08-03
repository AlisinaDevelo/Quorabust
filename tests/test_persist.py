import json

import numpy as np
import pandas as pd
import pytest

from quorabust.lineage import sha256_file
from quorabust.model import predict_proba_duplicate, train_duplicate_classifier
from quorabust.persist import load_classifier, save_classifier, save_metadata_sidecar


def _df():
    return pd.DataFrame(
        {
            "question1": [f"topic {i} explain" for i in range(24)],
            "question2": [f"explain topic {i % 6}" for i in range(24)],
            "is_duplicate": [i % 2 for i in range(24)],
        }
    )


def test_save_load_roundtrip(tmp_path):
    df = _df()
    builder, clf = train_duplicate_classifier(
        df,
        xgb_params={"n_estimators": 16, "max_depth": 3},
    )
    path = tmp_path / "model.pkl"
    save_classifier(path, builder, clf, meta={"run": "test"})
    b2, c2, meta = load_classifier(path)
    assert meta.get("run") == "test"
    q1, q2 = ["what is python"], ["how to learn python"]
    p1 = predict_proba_duplicate(builder, clf, q1, q2)
    p2 = predict_proba_duplicate(b2, c2, q1, q2)
    assert np.allclose(p1, p2)


def test_load_classifier_can_pin_artifact_sha256(tmp_path):
    path = tmp_path / "model.pkl"
    save_classifier(path, builder={"builder": True}, clf={"model": True}, meta={})

    digest = sha256_file(path)
    assert load_classifier(path, expected_sha256=digest)[2] == {}
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_classifier(path, expected_sha256="0" * 64)


def test_load_classifier_rejects_malformed_expected_sha256(tmp_path):
    path = tmp_path / "model.pkl"
    save_classifier(path, builder={}, clf={}, meta={})

    with pytest.raises(ValueError, match="64 hexadecimal"):
        load_classifier(path, expected_sha256="not-a-digest")


def test_save_metadata_sidecar_writes_json(tmp_path):
    path = tmp_path / "model.meta.json"
    written = save_metadata_sidecar(path, {"run": "test", "eval_accuracy": 0.9})
    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "run": "test",
        "eval_accuracy": 0.9,
    }
