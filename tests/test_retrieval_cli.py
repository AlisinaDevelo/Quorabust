import json

import pandas as pd

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
