import json

import pandas as pd

from quorabust.retrieval_benchmark_cli import main


def _write_inputs(tmp_path):
    catalog = tmp_path / "catalog.csv"
    qrels = tmp_path / "qrels.csv"
    pd.DataFrame(
        {
            "question_id": ["q1", "q2", "q3"],
            "question": [
                "How do I learn Python?",
                "Where can I buy train tickets?",
                "How should I cache API responses?",
            ],
        }
    ).to_csv(catalog, index=False)
    pd.DataFrame(
        {
            "query": ["best way to learn Python", "where to buy train tickets"],
            "question_id": ["q1", "q2"],
            "relevance": [2, 1],
        }
    ).to_csv(qrels, index=False)
    return catalog, qrels


def test_benchmark_cli_writes_provenance_and_stage_metrics(tmp_path):
    catalog, qrels = _write_inputs(tmp_path)
    output = tmp_path / "benchmark.json"

    assert (
        main(
            [
                "--catalog-csv",
                str(catalog),
                "--qrels-csv",
                str(qrels),
                "--ks",
                "1,2",
                "--candidate-k",
                "2",
                "--warmup-runs",
                "0",
                "--repetitions",
                "2",
                "--timeout-seconds",
                "10",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "quorabust-retrieve-benchmark"
    assert payload["configuration"] == {
        "embedding_model": None,
        "ks": [1, 2],
        "reranker_model": None,
        "retriever": "tfidf",
        "warmup_runs": 0,
        "repetitions": 2,
        "timeout_seconds": 10.0,
    }
    assert payload["sources"]["catalog"]["sha256"]
    assert payload["first_stage"]["recall_at_k"]["1"] == 1.0
    assert payload["final"]["ndcg_at_k"]["2"] == 1.0
    assert payload["work"]["reranker_enabled"] is False
    assert payload["measured_query_count"] == 4
    assert payload["latency_ms"]["end_to_end"]["count"] == 4
    assert payload["measurement_policy"]["timeout_seconds"] == 10.0
    assert payload["runtime"]["retriever_initialization_ms"] >= 0.0
    assert payload["runtime"]["reranker_initialization_ms"] == 0.0
    assert payload["runtime"]["startup_measurement"] == (
        "retriever_and_reranker_initialization_only"
    )


def test_benchmark_cli_rejects_qrels_outside_catalog(tmp_path, capsys):
    catalog, qrels = _write_inputs(tmp_path)
    frame = pd.read_csv(qrels)
    frame.loc[0, "question_id"] = "missing"
    frame.to_csv(qrels, index=False)

    assert main(["--catalog-csv", str(catalog), "--qrels-csv", str(qrels)]) == 1
    assert "unknown catalog" in capsys.readouterr().err


def test_benchmark_cli_rejects_candidate_k_below_requested_cutoff(tmp_path, capsys):
    catalog, qrels = _write_inputs(tmp_path)

    assert (
        main(
            [
                "--catalog-csv",
                str(catalog),
                "--qrels-csv",
                str(qrels),
                "--ks",
                "1,5",
                "--candidate-k",
                "2",
            ]
        )
        == 1
    )
    assert "largest k" in capsys.readouterr().err
