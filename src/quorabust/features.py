from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from quorabust.preprocess import clean_text, tokenize


def word_jaccard(q1: str, q2: str) -> float:
    """Jaccard similarity of word sets (0..1)."""
    a, b = set(tokenize(q1)), set(tokenize(q2))
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class PairFeatureBuilder:
    """TF–IDF cosine plus simple lexical stats for question pairs."""

    def __init__(
        self,
        max_features: int = 4096,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self._vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
            stop_words=None,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> PairFeatureBuilder:
        cleaned = [clean_text(t) for t in corpus if clean_text(t)]
        if not cleaned:
            cleaned = ["empty"]
        self._vec.fit(cleaned)
        self._fitted = True
        return self

    def feature_names(self) -> list[str]:
        return ["cos", "jaccard", "len_ratio", "abs_len_diff", "len_sum"]

    def fit_from_frame(
        self,
        df: pd.DataFrame,
        col_q1: str = "question1",
        col_q2: str = "question2",
    ) -> PairFeatureBuilder:
        parts: list[str] = []
        for c in (col_q1, col_q2):
            if c in df.columns:
                parts.extend(df[c].astype(str).tolist())
        return self.fit(parts)

    def transform_pairs(
        self,
        q1: list[str],
        q2: list[str],
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_from_frame() first.")
        if len(q1) != len(q2):
            raise ValueError("q1 and q2 must have the same length.")
        if not q1:
            return np.empty((0, len(self.feature_names())), dtype=np.float64)

        cleaned_q1 = [clean_text(value) for value in q1]
        cleaned_q2 = [clean_text(value) for value in q2]
        vectors_q1 = self._vec.transform(cleaned_q1)
        vectors_q2 = self._vec.transform(cleaned_q2)
        numerators = np.asarray(vectors_q1.multiply(vectors_q2).sum(axis=1)).ravel()
        norms_q1 = np.sqrt(np.asarray(vectors_q1.multiply(vectors_q1).sum(axis=1)).ravel())
        norms_q2 = np.sqrt(np.asarray(vectors_q2.multiply(vectors_q2).sum(axis=1)).ravel())
        denominators = norms_q1 * norms_q2
        cosines = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators, dtype=np.float64),
            where=denominators != 0,
        )
        jaccards = np.fromiter(
            (word_jaccard(left, right) for left, right in zip(cleaned_q1, cleaned_q2)),
            dtype=np.float64,
            count=len(cleaned_q1),
        )
        lengths_q1 = np.fromiter(
            (len(value.split()) for value in cleaned_q1),
            dtype=np.int64,
            count=len(cleaned_q1),
        )
        lengths_q2 = np.fromiter(
            (len(value.split()) for value in cleaned_q2),
            dtype=np.int64,
            count=len(cleaned_q2),
        )
        max_lengths = np.maximum(np.maximum(lengths_q1, lengths_q2), 1)
        min_lengths = np.minimum(lengths_q1, lengths_q2)
        return np.column_stack(
            (
                cosines,
                jaccards,
                min_lengths / max_lengths,
                np.abs(lengths_q1 - lengths_q2),
                lengths_q1 + lengths_q2,
            )
        ).astype(np.float64, copy=False)

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
