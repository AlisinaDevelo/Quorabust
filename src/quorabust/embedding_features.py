from __future__ import annotations

import numpy as np
import pandas as pd

from quorabust.preprocess import clean_text

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[misc, assignment]


class PairEmbeddingBuilder:
    """Sentence-embedding features for question pairs (optional ``nlp`` extra)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                'Missing dependency: install with pip install "Quorabust[nlp]"',
            )
        self._model = SentenceTransformer(model_name)
        self._fitted = True

    def fit(self, corpus: list[str] | None = None) -> PairEmbeddingBuilder:
        return self

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
        t1 = [clean_text(x) for x in q1]
        t2 = [clean_text(x) for x in q2]
        e1 = self._model.encode(
            t1,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        e2 = self._model.encode(
            t2,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        rows: list[list[float]] = []
        for i in range(len(q1)):
            a = e1[i].astype(np.float64, copy=False)
            b = e2[i].astype(np.float64, copy=False)
            na = float(np.linalg.norm(a)) + 1e-12
            nb = float(np.linalg.norm(b)) + 1e-12
            cos = float(np.dot(a, b) / (na * nb))
            l2 = float(np.linalg.norm(a - b))
            mad = float(np.mean(np.abs(a - b)))
            la, lb = len(t1[i].split()), len(t2[i].split())
            max_len = max(la, lb, 1)
            len_ratio = min(la, lb) / max_len
            rows.append([cos, l2, mad, len_ratio, float(la + lb)])
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
