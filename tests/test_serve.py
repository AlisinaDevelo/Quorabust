import json
import logging
import uuid

import pandas as pd
import pytest
from starlette.testclient import TestClient

from quorabust.lineage import sha256_file
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


def test_http_metrics_record_success_and_error_routes():
    app = create_app(api_key="secret")

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/models").status_code == 401
        metrics = client.get("/metrics").text

    assert (
        'quorabust_http_requests_total{method="GET",path="/health",status_code="200"} 1.0'
        in metrics
    )
    assert (
        'quorabust_http_requests_total{method="GET",path="/models",status_code="401"} 1.0'
        in metrics
    )
    assert (
        'quorabust_http_request_duration_seconds_count{method="GET",path="/health"} 1.0'
        in metrics
    )


def test_http_observability_bounds_unknown_path_labels():
    app = create_app()
    unknown_path = "/questions/this-is-user-content"

    with TestClient(app) as client:
        response = client.get(unknown_path)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
        metrics = client.get("/metrics").text

    assert 'path="<unmatched>"' in metrics
    assert unknown_path not in metrics


def test_ready_without_model_file():
    app = create_app(model_path_a="/nonexistent/quorabust_missing.pkl")
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503


def test_serve_rejects_model_checksum_mismatch(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p), model_sha256="0" * 64)

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        with TestClient(app):
            pass


def test_serve_accepts_pinned_model_checksum(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p), model_sha256=sha256_file(p))

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200


def test_request_id_is_generated_and_logged_without_request_content(caplog):
    app = create_app()

    with caplog.at_level(logging.INFO, logger="quorabust.http"):
        with TestClient(app) as client:
            response = client.get("/health")

    request_id = response.headers["X-Request-ID"]
    assert str(uuid.UUID(request_id)) == request_id
    events = [record for record in caplog.records if record.name == "quorabust.http"]
    assert events
    event = json.loads(events[-1].getMessage())
    assert set(event) == {
        "duration_ms",
        "event",
        "method",
        "path",
        "request_id",
        "status_code",
    }
    assert event["event"] == "http.request"
    assert event["method"] == "GET"
    assert event["path"] == "/health"
    assert event["request_id"] == request_id
    assert event["status_code"] == 200
    assert isinstance(event["duration_ms"], (int, float))


def test_request_id_reuses_valid_header_and_replaces_invalid_header():
    app = create_app()
    valid_request_id = "123e4567-e89b-42d3-a456-426614174000"

    with TestClient(app) as client:
        reused = client.get("/health", headers={"X-Request-ID": valid_request_id})
        replaced = client.get("/health", headers={"X-Request-ID": "not-a-uuid"})

    assert reused.headers["X-Request-ID"] == valid_request_id
    replaced_request_id = replaced.headers["X-Request-ID"]
    assert replaced_request_id != "not-a-uuid"
    assert str(uuid.UUID(replaced_request_id)) == replaced_request_id


def test_error_responses_include_request_id(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p), api_key="secret")

    with TestClient(app) as client:
        response = client.get("/models")

    assert response.status_code == 401
    request_id = response.headers["X-Request-ID"]
    assert str(uuid.UUID(request_id)) == request_id
    error = response.json()["error"]
    assert set(error) == {"code", "message", "request_id"}
    assert error["code"] == "unauthorized"
    assert error["message"] == "invalid API key"
    assert error["request_id"] == request_id


def test_openapi_includes_predict_examples(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p))
    with TestClient(app) as client:
        spec = client.get("/openapi.json").json()
    post = spec["paths"]["/predict"]["post"]
    assert post.get("summary")
    assert "scoring" in post.get("tags", [])
    assert "X-Request-ID" in post["responses"]["200"]["headers"]
    body = spec["components"]["schemas"]["PredictBody"]
    examples = body.get("examples") or []
    assert examples and "question1" in examples[0]
    metrics_content = spec["paths"]["/metrics"]["get"]["responses"]["200"]["content"]
    assert "text/plain" in metrics_content


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
    assert r.json()["error"]["code"] == "validation_error"


def test_predict_requires_configured_api_key(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p), api_key="secret")
    payload = {"question1": ["hello"], "question2": ["hello there"]}

    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 401
        wrong = client.post(
            "/predict",
            json=payload,
            headers={"X-Quorabust-API-Key": "wrong"},
        )
        assert wrong.status_code == 401
        assert wrong.headers["www-authenticate"] == "ApiKey"
        response = client.post(
            "/predict",
            json=payload,
            headers={"X-Quorabust-API-Key": "secret"},
        )

    assert response.status_code == 200


def test_models_requires_configured_api_key(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p), api_key="secret")

    with TestClient(app) as client:
        assert client.get("/models").status_code == 401
        response = client.get("/models", headers={"X-Quorabust-API-Key": "secret"})

    assert response.status_code == 200


def test_predict_rejects_batches_over_configured_limit(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p), max_batch_size=1)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "question1": ["hello", "foo"],
                "question2": ["hello there", "bar"],
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "batch_too_large"
    assert "maximum" in response.json()["error"]["message"]


def test_predict_returns_stable_error_for_mismatched_lists(tmp_path):
    p = tmp_path / "m.pkl"
    _tiny_pkl(p)
    app = create_app(model_path_a=str(p))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"question1": ["hello"], "question2": ["hello", "world"]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["message"] == "question1 and question2 length mismatch"


def test_models_without_loaded_artifacts_is_unavailable():
    app = create_app(model_path_a="/nonexistent/quorabust_missing.pkl")
    with TestClient(app) as client:
        response = client.get("/models")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


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
