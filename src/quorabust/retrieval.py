from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Protocol, Sequence

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


def _normalize_questions(questions: Sequence[CatalogQuestion]) -> list[CatalogQuestion]:
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
    return normalized


def _rank_hits(
    questions: Sequence[CatalogQuestion],
    scores: np.ndarray,
    k: int,
) -> list[CatalogHit]:
    ranked_indices = sorted(
        range(len(questions)),
        key=lambda index: (-float(scores[index]), questions[index].question_id),
    )[:k]
    return [
        CatalogHit(
            question_id=questions[index].question_id,
            text=questions[index].text,
            retrieval_score=float(scores[index]),
        )
        for index in ranked_indices
    ]


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
        normalized = _normalize_questions(questions)
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
        return _rank_hits(self._questions, scores, k)


def _load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('Missing dependency: install with pip install "Quorabust[nlp]"') from exc
    return SentenceTransformer(model_name)


def _normalize_embeddings(embeddings: Any) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("embedding model must return a non-empty two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(norms, 1e-12))
    return np.asarray(normalized, dtype=np.float64)


class SentenceTransformerCatalogRetriever:
    """Optional dense first-stage retriever using normalized sentence embeddings."""

    def __init__(self, model_name: str, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model if model is not None else _load_sentence_transformer(model_name)
        self._questions: list[CatalogQuestion] = []
        self._matrix: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._questions)

    def _encode(self, texts: list[str]) -> np.ndarray:
        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return _normalize_embeddings(embeddings)

    def fit(self, questions: Sequence[CatalogQuestion]) -> SentenceTransformerCatalogRetriever:
        normalized = _normalize_questions(questions)
        self._matrix = self._encode([clean_text(question.text) for question in normalized])
        self._questions = normalized
        return self

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        id_col: str = "question_id",
        text_col: str = "question",
    ) -> SentenceTransformerCatalogRetriever:
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
        query_embedding = self._encode([clean_text(query)])[0]
        scores = np.asarray(self._matrix @ query_embedding, dtype=np.float64).reshape(-1)
        return _rank_hits(self._questions, scores, k)


def _load_cross_encoder(model_name: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError('Missing dependency: install with pip install "Quorabust[nlp]"') from exc
    return CrossEncoder(model_name)


class SentenceTransformerCrossEncoderReranker:
    """Optional raw-score adapter for the bounded second retrieval stage."""

    def __init__(self, model_name: str, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model if model is not None else _load_cross_encoder(model_name)

    def score_batch(self, question1: list[str], question2: list[str]) -> list[float]:
        if len(question1) != len(question2):
            raise ValueError("question1 and question2 must have the same length")
        scores = self._model.predict(
            list(zip(question1, question2, strict=True)),
            show_progress_bar=False,
        )
        return [float(score) for score in np.asarray(scores, dtype=np.float64).reshape(-1)]


class CatalogRetriever(Protocol):
    @property
    def size(self) -> int: ...

    def search(self, query: str, *, k: int = 10) -> list[CatalogHit]: ...


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
    retriever: CatalogRetriever,
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
    retriever: CatalogRetriever,
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
