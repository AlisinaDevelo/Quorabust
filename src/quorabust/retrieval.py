from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from quorabust.preprocess import clean_text


@dataclass(frozen=True)
class CatalogQuestion:
    question_id: str
    text: str


@dataclass(frozen=True)
class CatalogHit:
    question_id: str
    text: str
    retrieval_score: float
    rerank_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question_id": self.question_id,
            "text": self.text,
            "retrieval_score": self.retrieval_score,
        }
        if self.rerank_score is not None:
            payload["rerank_score"] = self.rerank_score
        return payload


class TfidfCatalogRetriever:
    """Deterministic lexical first-stage retrieval for a question catalog."""

    def __init__(
        self,
        max_features: int = 4096,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
            stop_words=None,
        )
        self._questions: list[CatalogQuestion] = []
        self._matrix: Any | None = None

    @property
    def size(self) -> int:
        return len(self._questions)

    def fit(self, questions: Sequence[CatalogQuestion]) -> TfidfCatalogRetriever:
        normalized = [
            CatalogQuestion(str(question.question_id).strip(), str(question.text).strip())
            for question in questions
        ]
        if not normalized:
            raise ValueError("catalog must contain at least one question")
        if any(not question.question_id for question in normalized):
            raise ValueError("catalog question IDs must not be empty")
        if any(not question.text for question in normalized):
            raise ValueError("catalog question text must not be empty")
        ids = [question.question_id for question in normalized]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog question IDs must be unique")

        corpus = [clean_text(question.text) or "empty" for question in normalized]
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._questions = normalized
        return self

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        id_col: str = "question_id",
        text_col: str = "question",
    ) -> TfidfCatalogRetriever:
        for column in (id_col, text_col):
            if column not in frame.columns:
                raise KeyError(f"Missing catalog column: {column}")
        return self.fit(
            [
                CatalogQuestion(str(question_id), str(text))
                for question_id, text in zip(
                    frame[id_col].tolist(),
                    frame[text_col].tolist(),
                    strict=True,
                )
            ]
        )

    def search(self, query: str, *, k: int = 10) -> list[CatalogHit]:
        if self._matrix is None:
            raise RuntimeError("fit() or fit_frame() must be called before search()")
        if k < 1:
            raise ValueError("k must be at least 1")

        vector = self._vectorizer.transform([clean_text(query)])
        scores = np.asarray((self._matrix @ vector.T).toarray()).reshape(-1)
        ranked_indices = sorted(
            range(self.size),
            key=lambda index: (-float(scores[index]), self._questions[index].question_id),
        )[:k]
        return [
            CatalogHit(
                question_id=self._questions[index].question_id,
                text=self._questions[index].text,
                retrieval_score=float(scores[index]),
            )
            for index in ranked_indices
        ]


ScoreBatch = Callable[[list[str], list[str]], Sequence[float]]


def rerank_candidates(
    query: str,
    candidates: Sequence[CatalogHit],
    score_batch: ScoreBatch,
) -> list[CatalogHit]:
    """Apply a batch pair scorer and return candidates ordered by rerank score."""
    if not candidates:
        return []
    scores = np.asarray(
        score_batch(
            [query] * len(candidates),
            [candidate.text for candidate in candidates],
        ),
        dtype=np.float64,
    ).reshape(-1)
    if len(scores) != len(candidates):
        raise ValueError("reranker must return one score per candidate")
    if not np.isfinite(scores).all():
        raise ValueError("reranker scores must be finite")

    reranked = [
        replace(candidate, rerank_score=float(score))
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    return sorted(
        reranked,
        key=lambda candidate: (
            -float(
                candidate.rerank_score
                if candidate.rerank_score is not None
                else -np.inf
            ),
            -candidate.retrieval_score,
            candidate.question_id,
        ),
    )


def search_and_rerank(
    retriever: TfidfCatalogRetriever,
    query: str,
    *,
    k: int = 10,
    candidate_k: int = 50,
    score_batch: ScoreBatch,
) -> list[CatalogHit]:
    """Retrieve a bounded candidate set, rerank it, and return the top results."""
    if candidate_k < k:
        raise ValueError("candidate_k must be greater than or equal to k")
    candidates = retriever.search(query, k=candidate_k)
    return rerank_candidates(query, candidates, score_batch)[:k]


def candidate_recall_at_k(
    retriever: TfidfCatalogRetriever,
    cases: Sequence[tuple[str, Collection[str]]],
    *,
    k: int,
) -> float:
    """Measure whether at least one expected catalog ID is retrieved in the top k."""
    if not cases:
        raise ValueError("at least one recall case is required")
    if k < 1:
        raise ValueError("k must be at least 1")
    hits = 0
    for query, expected_ids in cases:
        expected = set(expected_ids)
        if not expected:
            raise ValueError("each recall case must contain an expected catalog ID")
        returned = {hit.question_id for hit in retriever.search(query, k=k)}
        hits += int(bool(returned & expected))
    return hits / len(cases)
