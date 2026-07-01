import pandas as pd
from starlette.testclient import TestClient

from quorabust.model import train_duplicate_classifier
from quorabust.persist import save_classifier
from quorabust.serve import create_app


def _tiny_pkl(path):
    df = pd.DataFrame(
        {
            "question1": ["hello world", "foo bar", "what is python"],
            "question2": ["hello there", "baz qux", "python language"],
            "is_duplicate": [1, 0, 1],
        }
    )
    b, clf = train_duplicate_classifier(df, xgb_params={"n_estimators": 12, "max_depth": 3})
    save_classifier(
        path,
        b,
        clf,
        meta={
            "id": "a",
            "csv": "/private/path/train.csv",
            "feature_backend": "tfidf",
            "feature_schema": ["cos", "jaccard", "len_ratio", "abs_len_diff", "len_sum"],
            "n_train": len(df),
            "eval_accuracy": 0.8,
        },
    )


def _tiny_pkl_with_threshold(path, threshold: float) -> None:
    df = pd.DataFrame(
        {
            "question1": ["hello world", "foo bar", "what is python"],
            "question2": ["hello there", "baz qux", "python language"],
            "is_duplicate": [1, 0, 1],
        }
    )
    b, clf = train_duplicate_classifier(df, xgb_params={"n_estimators": 12, "max_depth": 3})
    save_classifier(
        path,
        b,
        clf,
        meta={
            "feature_backend": "tfidf",
            "feature_schema": ["cos", "jaccard", "len_ratio", "abs_len_diff", "len_sum"],
            "decision_threshold": threshold,
        },
    )


def test_serve_health_ready_predict(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/ready").status_code == 200
        r = client.post(
            "/predict",
            json={"question1": ["hello"], "question2": ["hello there"]},
        )
        assert r.status_code == 200
        body = r.json()
        assert "proba_duplicate" in body
        assert "is_duplicate" in body
        assert body["decision_threshold"] == 0.5
        assert len(body["is_duplicate"]) == len(body["proba_duplicate"])
        assert body["variant"] == "a"
        assert body["features"] is None
        m = client.get("/metrics")
        assert m.status_code == 200
        assert b"quorabust_predictions_total" in m.content


def test_ready_without_model_file():
    app = create_app(model_path_a="/nonexistent/quorabust_missing.pkl")
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503


def test_openapi_includes_predict_examples(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        spec = client.get("/openapi.json").json()
    post = spec["paths"]["/predict"]["post"]
    assert post.get("summary")
    assert "scoring" in post.get("tags", [])
    body = spec["components"]["schemas"]["PredictBody"]
    examples = body.get("examples") or []
    assert examples and "question1" in examples[0]


def test_models_returns_safe_public_metadata(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        r = client.get("/models")
    assert r.status_code == 200
    model = r.json()["variants"]["a"]
    assert model["feature_backend"] == "tfidf"
    assert model["eval_metrics"]["accuracy"] == 0.8
    assert "csv" not in model
    assert "id" not in model


def test_predict_can_return_feature_explanations(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        r = client.post(
            "/predict?explain=true",
            json={"question1": ["hello"], "question2": ["hello there"]},
        )
    assert r.status_code == 200
    features = r.json()["features"]
    assert features and set(features[0]) == {
        "cos",
        "jaccard",
        "len_ratio",
        "abs_len_diff",
        "len_sum",
    }


def test_predict_uses_artifact_decision_threshold(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl_with_threshold(p, 0.91)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        r = client.post(
            "/predict",
            json={"question1": ["hello"], "question2": ["hello there"]},
        )
    assert r.status_code == 200
    assert r.json()["decision_threshold"] == 0.91


def test_models_exposes_public_decision_threshold_metadata(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl_with_threshold(p, 0.91)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        r = client.get("/models")
    assert r.status_code == 200
    model = r.json()["variants"]["a"]
    assert model["decision_threshold"] == 0.91


def test_predict_threshold_query_overrides_artifact_threshold(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl_with_threshold(p, 0.91)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        r = client.post(
            "/predict?threshold=0.2",
            json={"question1": ["hello"], "question2": ["hello there"]},
        )
    assert r.status_code == 200
    assert r.json()["decision_threshold"] == 0.2


def test_predict_rejects_invalid_threshold(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        r = client.post(
            "/predict?threshold=1.2",
            json={"question1": ["hello"], "question2": ["hello there"]},
        )
    assert r.status_code == 422


def test_models_without_loaded_artifacts_is_unavailable():
    app = create_app(model_path_a="/nonexistent/quorabust_missing.pkl")
    with TestClient(app) as client:
        assert client.get("/models").status_code == 503


def test_serve_ab_variant_header(tmp_path):
    pa = tmp_path / "a.pkl"
    pb = tmp_path / "b.pkl"
    _tiny_pkl(pa)
    _tiny_pkl(pb)
    app = create_app(model_path_a=str(pa), model_path_b=str(pb))
    with TestClient(app) as client:
        r = client.post(
            "/predict",
            json={"question1": ["x"], "question2": ["y"]},
            headers={"X-Quorabust-Variant": "b"},
        )
        assert r.status_code == 200
        assert r.json()["variant"] == "b"
