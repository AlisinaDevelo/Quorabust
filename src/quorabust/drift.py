from __future__ import annotations

from typing import Any


def mean_shift_scores(
    reference: dict[str, float],
    current: dict[str, float],
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
    """
    Per-key normalized shift: |current - reference| / max(|reference|, eps).

    Use with batch feature means vs training reference stored in model meta.
    """
    out: dict[str, float] = {}
    for k, ref_v in reference.items():
        if k not in current:
            continue
        cur_v = current[k]
        denom = max(abs(ref_v), eps)
        out[k] = abs(cur_v - ref_v) / denom
    return out


def max_mean_shift(
    reference: dict[str, float],
    current: dict[str, float],
    *,
    eps: float = 1e-9,
) -> float:
    scores = mean_shift_scores(reference, current, eps=eps)
    return max(scores.values(), default=0.0)


def feature_means_from_matrix(
    feature_names: list[str],
    X: Any,
) -> dict[str, float]:
    """Column means for a 2D array-like (numpy)."""
    import numpy as np

    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if arr.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match X columns")
    means = arr.mean(axis=0)
    return {name: float(m) for name, m in zip(feature_names, means, strict=True)}
