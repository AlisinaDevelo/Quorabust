import numpy as np
import pandas as pd
import pytest

from quorabust.retrieval import (
    CatalogHit,
    CatalogQuestion,
    SentenceTransformerCatalogRetriever,
    SentenceTransformerCrossEncoderReranker,
    TfidfCatalogRetriever,
    candidate_recall_at_k,
    rerank_candidates,
    search_and_rerank,
)


class _FakeEmbeddingModel:
    def encode(self, texts, **_kwargs):
        return np.asarray(
            [[1.0, 0.0] if "python" in text else [0.0, 1.0] for text in texts],
            dtype=float,
        )


class _FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        return [0.9 if "python" in right else 0.1 for _left, right in pairs]


def _retriever() -> TfidfCatalogRetriever:
    return TfidfCatalogRetriever().fit(
        [
            CatalogQuestion("q1", "How do I learn Python?"),
            CatalogQuestion("q2", "Where can I buy train tickets?"),
            CatalogQuestion("q3", "How should I cache API responses?"),
        ]
    )


def test_tfidf_retriever_returns_deterministic_top_k_hits():
    retriever = _retriever()

    hits = retriever.search("What is the best way to learn Python?", k=2)

    assert retriever.size == 3
    assert [hit.question_id for hit in hits] == ["q1", "q2"]
    assert hits[0].retrieval_score > hits[1].retrieval_score
    assert hits[0].rerank_score is None
    assert hits[0].as_dict() == {
        "question_id": "q1",
        "text": "How do I learn Python?",
        "retrieval_score": hits[0].retrieval_score,
    }


def test_retriever_can_fit_a_catalog_frame():
    frame = pd.DataFrame(
        {
            "question_id": ["a", "b"],
            "question": ["How do I deploy an API?", "How do I tune a database?"],
        }
    )

    retriever = TfidfCatalogRetriever().fit_frame(frame)

    assert retriever.search("deploy service", k=1)[0].question_id == "a"


def test_sentence_transformer_retriever_uses_the_shared_catalog_contract():
    retriever = SentenceTransformerCatalogRetriever(
        "fake-embedding",
        model=_FakeEmbeddingModel(),
        revision="  embedding-commit  ",
    ).fit(
        [
            CatalogQuestion("q1", "python question"),
            CatalogQuestion("q2", "train question"),
        ]
    )

    assert retriever.search("python help", k=1)[0].question_id == "q1"
    assert retriever.model_revision == "embedding-commit"


def test_cross_encoder_adapter_returns_raw_batch_scores():
    reranker = SentenceTransformerCrossEncoderReranker(
        "fake-cross",
        model=_FakeCrossEncoder(),
        revision="cross-commit",
    )

    scores = reranker.score_batch(["query", "query"], ["python", "tickets"])

    assert scores == [0.9, 0.1]
    assert reranker.model_revision == "cross-commit"


def test_optional_retrieval_model_revisions_reject_blank_values():
    with pytest.raises(ValueError, match="revision"):
        SentenceTransformerCatalogRetriever(
            "fake-embedding", model=_FakeEmbeddingModel(), revision=" "
        )
    with pytest.raises(ValueError, match="revision"):
        SentenceTransformerCrossEncoderReranker(
            "fake-cross", model=_FakeCrossEncoder(), revision=" "
        )


def test_reranker_reorders_candidates_and_preserves_ties_deterministically():
    candidates = [
        CatalogHit("b", "second", retrieval_score=0.8),
        CatalogHit("a", "first", retrieval_score=0.7),
    ]

    reranked = rerank_candidates("query", candidates, lambda _q1, _q2: [0.5, 0.9])

    assert [hit.question_id for hit in reranked] == ["a", "b"]
    assert reranked[0].rerank_score == 0.9
    assert reranked[0].as_dict()["rerank_score"] == 0.9


def test_search_and_rerank_bounds_the_expensive_stage():
    seen = {}

    def score_batch(q1, q2):
        seen["count"] = len(q1)
        return [float(index) for index in range(len(q1))]

    results = search_and_rerank(
        _retriever(),
        "python learning",
        k=1,
        candidate_k=2,
        score_batch=score_batch,
    )

    assert seen["count"] == 2
    assert len(results) == 1
    assert results[0].rerank_score == 1.0


def test_candidate_recall_at_k():
    score = candidate_recall_at_k(
        _retriever(),
        [("python learning", {"q1"}), ("train tickets", {"q2"})],
        k=2,
    )

    assert score == 1.0


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda retriever: retriever.search("query", k=0), "k must be at least 1"),
        (lambda retriever: retriever.search("query"), r"fit\(\) or fit_frame\(\)"),
    ],
)
def test_retriever_rejects_invalid_usage(operation, message):
    retriever = TfidfCatalogRetriever()
    if "fit" in message:
        target = retriever
    else:
        target = _retriever()
    with pytest.raises((RuntimeError, ValueError), match=message):
        operation(target)


def test_retriever_rejects_duplicate_ids_and_invalid_reranker_output():
    with pytest.raises(ValueError, match="unique"):
        TfidfCatalogRetriever().fit(
            [CatalogQuestion("q1", "one"), CatalogQuestion("q1", "two")]
        )

    candidates = [CatalogHit("q1", "one", retrieval_score=1.0)]
    with pytest.raises(ValueError, match="one score"):
        rerank_candidates("query", candidates, lambda _q1, _q2: [])
    with pytest.raises(ValueError, match="finite"):
        rerank_candidates("query", candidates, lambda _q1, _q2: [np.nan])
