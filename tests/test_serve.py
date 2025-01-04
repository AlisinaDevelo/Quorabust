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
    save_classifier(path, b, clf, meta={"id": "a"})


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
        assert body["variant"] == "a"
        m = client.get("/metrics")
        assert m.status_code == 200
        assert b"quorabust_predictions_total" in m.content


def test_ready_without_model_file():
    app = create_app(model_path_a="/nonexistent/quorabust_missing.pkl")
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503


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
