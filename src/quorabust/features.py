from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> PairFeatureBuilder:
        cleaned = [clean_text(t) for t in corpus if clean_text(t)]
        if not cleaned:
            cleaned = ["empty"]
        self._vec.fit(cleaned)
        self._fitted = True
        return self

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
        rows = []
        for a, b in zip(q1, q2, strict=True):
            ca, cb = clean_text(a), clean_text(b)
            va = self._vec.transform([ca])
            vb = self._vec.transform([cb])
            cos = float(cosine_similarity(va, vb)[0, 0])
            jac = word_jaccard(ca, cb)
            la, lb = len(ca.split()), len(cb.split())
            max_len = max(la, lb, 1)
            min_len = min(la, lb)
            len_ratio = min_len / max_len
            abs_diff = abs(la - lb)
            rows.append([cos, jac, len_ratio, float(abs_diff), float(la + lb)])
        return np.asarray(rows, dtype=np.float64)

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
