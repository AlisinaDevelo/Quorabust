from __future__ import annotations

import numpy as np
import pandas as pd

from quorabust.preprocess import clean_text

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore[misc, assignment]


class PairCrossEncoderBuilder:
    """Cross-encoder pair scores plus simple length features (optional ``nlp`` extra)."""

    def __init__(self, model_name: str = "cross-encoder/quora-distilroberta-base") -> None:
        if CrossEncoder is None:
            raise RuntimeError(
                'Missing dependency: install with pip install "Quorabust[nlp]"',
            )
        self.model_name = model_name
        self._model = CrossEncoder(model_name)
        self._fitted = True

    def fit(self, corpus: list[str] | None = None) -> PairCrossEncoderBuilder:
        return self

    def feature_names(self) -> list[str]:
        return ["cross_score", "len_ratio", "abs_len_diff", "len_sum"]

    def fit_from_frame(
        self,
        df: pd.DataFrame,
        col_q1: str = "question1",
        col_q2: str = "question2",
    ) -> PairCrossEncoderBuilder:
        return self

    def transform_pairs(
        self,
        q1: list[str],
        q2: list[str],
    ) -> np.ndarray:
        if len(q1) != len(q2):
            raise ValueError("q1 and q2 must have the same length.")
        t1 = [clean_text(x) for x in q1]
        t2 = [clean_text(x) for x in q2]
        scores = np.asarray(
            self._model.predict(list(zip(t1, t2, strict=True)), show_progress_bar=False),
            dtype=np.float64,
        ).reshape(-1)

        rows: list[list[float]] = []
        for idx, (a, b) in enumerate(zip(t1, t2, strict=True)):
            la, lb = len(a.split()), len(b.split())
            max_len = max(la, lb, 1)
            len_ratio = min(la, lb) / max_len
            rows.append([float(scores[idx]), len_ratio, float(abs(la - lb)), float(la + lb)])
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
