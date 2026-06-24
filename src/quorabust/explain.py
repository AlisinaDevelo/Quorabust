from __future__ import annotations

from typing import Any


def feature_names_for_builder(
    builder: Any,
    feature_schema: list[str] | None = None,
) -> list[str]:
    """Return stable feature names for a loaded builder."""
    if feature_schema:
        return feature_schema
    if hasattr(builder, "feature_names"):
        names = builder.feature_names()
        if isinstance(names, list) and all(isinstance(x, str) for x in names):
            return names
    return []


def explain_pair_features(
    builder: Any,
    q1: list[str],
    q2: list[str],
    *,
    feature_schema: list[str] | None = None,
) -> list[dict[str, float]]:
    """Expose model input feature values for each question pair."""
    values = builder.transform_pairs(q1, q2)
    names = feature_names_for_builder(builder, feature_schema)
    if not names:
        names = [f"feature_{i}" for i in range(values.shape[1])]
    return [
        {name: float(value) for name, value in zip(names, row, strict=False)}
        for row in values
    ]
