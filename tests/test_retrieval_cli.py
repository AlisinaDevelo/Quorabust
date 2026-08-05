import json

import pandas as pd

import quorabust.retrieval_cli as retrieval_cli
from quorabust.retrieval import CatalogHit
from quorabust.retrieval_cli import main


def _write_catalog(path):
    pd.DataFrame(
        {
            "question_id": ["q1", "q2", "q3"],
            "question": [
                "How do I learn Python?",
                "Where can I buy train tickets?",
                "How should I cache API responses?",
            ],
        }
    ).to_csv(path, index=False)


def test_retrieval_cli_writes_json_for_multiple_queries(tmp_path):
    catalog = tmp_path / "catalog.csv"
    output = tmp_path / "results.json"
    _write_catalog(catalog)

    assert (
        main(
            [
                "--catalog-csv",
                str(catalog),
                "--query",
                "best way to learn Python",
                "--query",
                "where to buy train tickets",
                "--k",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["catalog_size"] == 3
    assert payload["retriever"] == "tfidf"
    assert [query["hits"][0]["question_id"] for query in payload["queries"]] == ["q1", "q2"]
    assert "retrieval_score" in payload["queries"][0]["hits"][0]


def test_retrieval_cli_prints_json_and_rejects_bad_catalog(tmp_path, capsys):
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)

    assert main(["--catalog-csv", str(catalog), "--query", "python", "--k", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["queries"][0]["query"] == "python"

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": ["value"]}).to_csv(bad, index=False)
    assert main(["--catalog-csv", str(bad), "--query", "python"]) == 1
    assert "Missing catalog column" in capsys.readouterr().err


def test_retrieval_cli_can_use_optional_dense_and_reranker_stages(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.csv"
    output = tmp_path / "results.json"
    _write_catalog(catalog)

    class FakeDenseRetriever:
        def __init__(self, model_name):
            assert model_name == "fake-embedding"
            self.size = 2

        def fit_frame(self, frame, *, id_col, text_col):
            assert list(frame.columns) == ["question_id", "question"]
            assert (id_col, text_col) == ("question_id", "question")
            return self

        def search(self, _query, *, k):
            return [
                CatalogHit("q1", "python", retrieval_score=0.8),
                CatalogHit("q2", "tickets", retrieval_score=0.7),
            ][:k]

    class FakeReranker:
        def __init__(self, model_name):
            assert model_name == "fake-cross"

        def score_batch(self, question1, question2):
            assert question1 == ["python", "python"]
            assert question2 == ["python", "tickets"]
            return [0.9, 0.1]

    monkeypatch.setattr(retrieval_cli, "SentenceTransformerCatalogRetriever", FakeDenseRetriever)
    monkeypatch.setattr(retrieval_cli, "SentenceTransformerCrossEncoderReranker", FakeReranker)

    assert (
        main(
            [
                "--catalog-csv",
                str(catalog),
                "--query",
                "python",
                "--retriever",
                "embedding",
                "--embedding-model",
                "fake-embedding",
                "--reranker-model",
                "fake-cross",
                "--candidate-k",
                "2",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["retriever"] == "embedding"
    assert payload["reranker"] == "fake-cross"
    assert payload["queries"][0]["hits"][0]["rerank_score"] == 0.9
