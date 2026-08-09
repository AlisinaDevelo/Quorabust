import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from quorabust.calibration import calibrate_classifier
from quorabust.lineage import sha256_file
from quorabust.model import predict_proba_duplicate, train_duplicate_classifier
from quorabust.persist import load_classifier, save_classifier
from quorabust.serve import create_app


def _trained_model():
    frame = pd.DataFrame(
        {
            "question1": ["hello world", "foo bar", "what is python", "cache api"],
            "question2": ["hello there", "baz qux", "python language", "api cache"],
            "is_duplicate": [1, 0, 1, 1],
        }
    )
    return frame, train_duplicate_classifier(
        frame,
        xgb_params={"n_estimators": 8, "max_depth": 2},
    )


def test_qmodel_round_trip_preserves_predictions_and_strips_local_paths(tmp_path):
    frame, (builder, classifier) = _trained_model()
    source = tmp_path / "source.pkl"
    safe = tmp_path / "model.qmodel"
    save_classifier(
        source,
        builder,
        classifier,
        meta={"csv": "/private/train.csv", "decision_threshold": 0.7},
    )
    source_builder, source_classifier, _ = load_classifier(source)
    save_classifier(
        safe,
        source_builder,
        source_classifier,
        meta={"csv": "/private/train.csv", "decision_threshold": 0.7},
    )

    loaded_builder, loaded_classifier, meta = load_classifier(safe)
    source_proba = predict_proba_duplicate(
        source_builder,
        source_classifier,
        frame["question1"].tolist(),
        frame["question2"].tolist(),
    )
    safe_proba = predict_proba_duplicate(
        loaded_builder,
        loaded_classifier,
        frame["question1"].tolist(),
        frame["question2"].tolist(),
    )

    assert np.allclose(source_proba, safe_proba, atol=1e-7)
    assert meta["artifact_format"] == "quorabust.safe.tfidf_xgboost"
    assert meta["decision_threshold"] == 0.7
    assert "csv" not in meta
    assert b"pickle" not in safe.read_bytes().lower()


def test_qmodel_serves_with_digest_and_safe_model_identity(tmp_path):
    _frame, (builder, classifier) = _trained_model()
    safe = tmp_path / "model.qmodel"
    save_classifier(safe, builder, classifier, meta={"feature_backend": "tfidf"})
    app = create_app(model_path_a=str(safe), model_sha256=sha256_file(safe))

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        model = client.get("/models").json()["variants"]["a"]

    assert model["artifact_sha256"] == sha256_file(safe)
    assert model["artifact_format"] == "quorabust.safe.tfidf_xgboost"


def test_qmodel_members_are_a_small_explicit_bundle(tmp_path):
    _frame, (builder, classifier) = _trained_model()
    safe = tmp_path / "model.qmodel"
    save_classifier(safe, builder, classifier, meta={})

    import zipfile

    with zipfile.ZipFile(safe) as archive:
        assert set(archive.namelist()) == {"manifest.json", "builder.json", "booster.json"}
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "quorabust.safe.tfidf_xgboost"


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrated_qmodel_round_trip_preserves_predictions_without_pickle(tmp_path, method):
    frame, (builder, classifier) = _trained_model()
    calibrated = calibrate_classifier(
        classifier,
        builder.transform_frame(frame),
        frame["is_duplicate"].to_numpy(),
        method=method,
    )
    safe = tmp_path / f"calibrated-{method}.qmodel"
    save_classifier(
        safe,
        builder,
        calibrated,
        meta={"calibration_method": method, "decision_threshold": 0.6},
    )

    loaded_builder, loaded_classifier, meta = load_classifier(safe)
    source_proba = predict_proba_duplicate(
        builder,
        calibrated,
        frame["question1"].tolist(),
        frame["question2"].tolist(),
    )
    safe_proba = predict_proba_duplicate(
        loaded_builder,
        loaded_classifier,
        frame["question1"].tolist(),
        frame["question2"].tolist(),
    )

    assert np.allclose(source_proba, safe_proba, atol=1e-7)
    assert meta["calibration_method"] == method
    assert b"pickle" not in safe.read_bytes().lower()
    with zipfile.ZipFile(safe) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "builder.json",
            "booster.json",
            "calibrator.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema_version"] == 2
        assert manifest["calibrator_member"] == "calibrator.json"


def test_calibrated_qmodel_requires_matching_metadata(tmp_path):
    frame, (builder, classifier) = _trained_model()
    calibrated = calibrate_classifier(
        classifier,
        builder.transform_frame(frame),
        frame["is_duplicate"].to_numpy(),
    )

    with pytest.raises(ValueError, match="matching calibration_method"):
        save_classifier(tmp_path / "calibrated.qmodel", builder, calibrated, meta={})


def test_calibrated_qmodel_rejects_malformed_calibrator_payload(tmp_path):
    frame, (builder, classifier) = _trained_model()
    calibrated = calibrate_classifier(
        classifier,
        builder.transform_frame(frame),
        frame["is_duplicate"].to_numpy(),
        method="isotonic",
    )
    source = tmp_path / "calibrated.qmodel"
    broken = tmp_path / "broken.qmodel"
    save_classifier(
        source,
        builder,
        calibrated,
        meta={"calibration_method": "isotonic"},
    )

    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(broken, "w") as output:
        for name in archive.namelist():
            payload = archive.read(name)
            if name == "calibrator.json":
                payload = json.dumps(
                    {
                        "schema_version": 1,
                        "method": "isotonic",
                        "x_thresholds": [0.8, 0.2],
                        "y_thresholds": [0.0, 1.0],
                    }
                ).encode("utf-8")
            output.writestr(name, payload)

    with pytest.raises(ValueError, match="isotonic calibration thresholds are invalid"):
        load_classifier(broken)
