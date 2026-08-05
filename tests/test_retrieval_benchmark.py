import json

import pandas as pd
import pytest

from quorabust.retrieval import CatalogQuestion, TfidfCatalogRetriever
from quorabust.retrieval_benchmark import (
    RetrievalCase,
    benchmark_retrieval,
    evaluate_rankings,
    load_retrieval_qrels,
    summarize_latencies_ms,
)


def _retriever() -> TfidfCatalogRetriever:
    return TfidfCatalogRetriever().fit(
        [
            CatalogQuestion("q1", "How do I learn Python?"),
            CatalogQuestion("q2", "Where can I buy train tickets?"),
            CatalogQuestion("q3", "How should I cache API responses?"),
        ]
    )


def test_evaluate_rankings_reports_recall_mrr_and_ndcg():
    cases = [
        RetrievalCase("python", {"q1": 2.0}),
        RetrievalCase("tickets", {"q2": 1.0, "q3": 1.0}),
    ]

    metrics = evaluate_rankings(
        [["q1", "q3"], ["q1", "q2", "q3"]],
        cases,
        ks=[1, 2],
    )

    assert metrics["recall_at_k"]["1"] == 0.5
    assert metrics["recall_at_k"]["2"] == 0.75
    assert metrics["mrr_at_k"]["1"] == 0.5
    assert metrics["mrr_at_k"]["2"] == 0.75
    assert 0.0 < metrics["ndcg_at_k"]["1"] <= 1.0
    assert 0.0 < metrics["ndcg_at_k"]["2"] <= 1.0


def test_qrels_loader_groups_queries_and_rejects_unknown_ids(tmp_path):
    qrels = tmp_path / "qrels.csv"
    pd.DataFrame(
        {
            "query": ["python", "tickets", "tickets"],
            "question_id": ["q1", "q2", "q3"],
            "relevance": [2, 1, 0],
        }
    ).to_csv(qrels, index=False)

    cases = load_retrieval_qrels(qrels, catalog_ids={"q1", "q2", "q3"})

    assert cases == [
        RetrievalCase("python", {"q1": 2.0}),
        RetrievalCase("tickets", {"q2": 1.0, "q3": 0.0}),
    ]

    with pytest.raises(ValueError, match="unknown catalog"):
        load_retrieval_qrels(qrels, catalog_ids={"q1", "q2"})


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        ([], "at least one retrieval case"),
        ([RetrievalCase("", {"q1": 1})], "queries must not be empty"),
        ([RetrievalCase("q", {})], "relevant catalog ID"),
        ([RetrievalCase("q", {"q1": 0})], "positive relevance"),
    ],
)
def test_evaluation_rejects_invalid_cases(cases, message):
    with pytest.raises(ValueError, match=message):
        evaluate_rankings([["q1"]], cases, ks=[1])


def test_benchmark_separates_stage_metrics_latency_and_work():
    cases = [
        RetrievalCase("python", {"q1": 1}),
        RetrievalCase("python tickets", {"q3": 1}),
    ]
    seen = {}

    def score_batch(question1, question2):
        seen["count"] = len(question1)
        return [
            0.9
            if ("python" in query and "python" in text)
            or ("tickets" in query and "cache" in text)
            else 0.1
            for query, text in zip(question1, question2, strict=True)
        ]

    result = benchmark_retrieval(
        _retriever(),
        cases,
        ks=[1, 2],
        candidate_k=3,
        score_batch=score_batch,
        warmup_runs=1,
        repetitions=2,
    )

    assert result["query_count"] == 2
    assert result["measured_query_count"] == 4
    assert result["first_stage"]["recall_at_k"]["1"] == 0.5
    assert result["final"]["recall_at_k"]["1"] == 1.0
    assert result["work"]["reranker_enabled"] is True
    assert result["work"]["reranker_pairs"] == 12
    assert seen["count"] == 3
    assert result["latency_ms"]["end_to_end"]["count"] == 4
    assert result["measurement_policy"] == {
        "warmup_runs": 1,
        "repetitions": 2,
        "quality_passes": 1,
        "latency_samples_per_stage": 4,
        "timeout_seconds": None,
        "concurrency": 1,
        "execution": "serial",
        "timeout_behavior": "cooperative_deadline_between_queries_and_stages",
    }
    assert result["runtime"]["python_version"]
    assert result["runtime"]["rss_measurement"] == "process_maxrss_since_process_start"
    assert (
        result["runtime"]["peak_rss_bytes"] is None
        or result["runtime"]["peak_rss_bytes"] > 0
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup_runs": -1}, "warmup_runs"),
        ({"repetitions": 0}, "repetitions"),
        ({"timeout_seconds": 0.0}, "timeout_seconds"),
    ],
)
def test_benchmark_rejects_invalid_measurement_policy(kwargs, message):
    with pytest.raises(ValueError, match=message):
        benchmark_retrieval(
            _retriever(),
            [RetrievalCase("python", {"q1": 1})],
            ks=[1],
            candidate_k=1,
            **kwargs,
        )


def test_latency_summary_is_json_serializable_and_rejects_bad_values():
    summary = summarize_latencies_ms([1.0, 2.0, 3.0])
    assert summary["p50"] == 2.0
    assert summary["p99"] == pytest.approx(2.98)
    json.dumps(summary)

    with pytest.raises(ValueError, match="at least one"):
        summarize_latencies_ms([])
    with pytest.raises(ValueError, match="finite"):
        summarize_latencies_ms([float("nan")])
