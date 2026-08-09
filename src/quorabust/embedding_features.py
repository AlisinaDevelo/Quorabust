from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from quorabust.preprocess import clean_text

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[misc, assignment]


class PairEmbeddingBuilder:
    """Sentence-embedding features for question pairs (optional ``nlp`` extra)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        batch_size: int = 64,
        cache_size: int = 50_000,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                'Missing dependency: install with pip install "Quorabust[nlp]"',
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if cache_size < 0:
            raise ValueError("cache_size must be zero or greater")
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._cache_size = cache_size
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._fitted = True

    def fit(self, corpus: list[str] | None = None) -> PairEmbeddingBuilder:
        return self

    def feature_names(self) -> list[str]:
        return ["cos", "l2", "mad", "len_ratio", "len_sum"]

    def fit_from_frame(
        self,
        df: pd.DataFrame,
        col_q1: str = "question1",
        col_q2: str = "question2",
    ) -> PairEmbeddingBuilder:
        return self

    def transform_pairs(
        self,
        q1: list[str],
        q2: list[str],
    ) -> np.ndarray:
        if len(q1) != len(q2):
            raise ValueError("q1 and q2 must have the same length.")
        if not q1:
            return np.empty((0, len(self.feature_names())), dtype=np.float64)
        t1 = [clean_text(x) for x in q1]
        t2 = [clean_text(x) for x in q2]
        embeddings = self._encode_texts(t1 + t2)
        e1 = embeddings[: len(q1)].astype(np.float64, copy=False)
        e2 = embeddings[len(q1) :].astype(np.float64, copy=False)
        differences = e1 - e2
        norms = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        cosines = np.divide(
            np.einsum("ij,ij->i", e1, e2),
            norms,
            out=np.zeros(len(q1), dtype=np.float64),
            where=norms != 0,
        )
        l2 = np.linalg.norm(differences, axis=1)
        mad = np.mean(np.abs(differences), axis=1)
        lengths_q1 = np.fromiter(
            (len(value.split()) for value in t1), dtype=np.int64, count=len(t1)
        )
        lengths_q2 = np.fromiter(
            (len(value.split()) for value in t2), dtype=np.int64, count=len(t2)
        )
        max_lengths = np.maximum(np.maximum(lengths_q1, lengths_q2), 1)
        return np.column_stack(
            (
                cosines,
                l2,
                mad,
                np.minimum(lengths_q1, lengths_q2) / max_lengths,
                lengths_q1 + lengths_q2,
            )
        ).astype(np.float64, copy=False)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        resolved: dict[str, np.ndarray] = {}
        missing: list[str] = []
        missing_set: set[str] = set()
        for text in texts:
            cached = self._embedding_cache.get(text)
            if cached is not None:
                self._embedding_cache.move_to_end(text)
                resolved[text] = cached
            elif text not in missing_set:
                missing.append(text)
                missing_set.add(text)

        if missing:
            encoded = np.asarray(
                self._model.encode(
                    missing,
                    batch_size=self._batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if encoded.ndim != 2 or encoded.shape[0] != len(missing):
                raise ValueError("embedding model returned an invalid feature matrix")
            for text, vector in zip(missing, encoded, strict=True):
                resolved[text] = vector
                if self._cache_size:
                    self._embedding_cache[text] = vector
                    self._embedding_cache.move_to_end(text)
                    while len(self._embedding_cache) > self._cache_size:
                        self._embedding_cache.popitem(last=False)

        return np.asarray([resolved[text] for text in texts], dtype=np.float32)

    def transform_frame(
        self,
        df: pd.DataFrame,
        col_q1: str = "question1",
        col_q2: str = "question2",
    ) -> np.ndarray:
        return self.transform_pairs(
            df[col_q1].astype(str).tolist(),
            df[col_q2].astype(str).tolist(),
        )
