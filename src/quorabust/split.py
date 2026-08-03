from __future__ import annotations

import numpy as np
import pandas as pd

_QUESTION_ID_COLUMNS = ("qid1", "qid2")


def _has_question_ids(df: pd.DataFrame) -> bool:
    if not set(_QUESTION_ID_COLUMNS).issubset(df.columns):
        return False
    return bool(df[list(_QUESTION_ID_COLUMNS)].notna().all().all())


def _question_components(df: pd.DataFrame) -> list[list[int]]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in zip(df["qid1"], df["qid2"], strict=True):
        union(str(left), str(right))

    components: dict[str, list[int]] = {}
    for index, (left, _right) in enumerate(zip(df["qid1"], df["qid2"], strict=True)):
        root = find(str(left))
        components.setdefault(root, []).append(index)
    return list(components.values())


def _component_holdout_indices(
    df: pd.DataFrame,
    *,
    n_eval: int,
    seed: int,
) -> set[int]:
    components = _question_components(df)
    if len(components) < 2:
        raise ValueError(
            "question IDs form one connected component; provide a separate holdout "
            "or use a dataset with at least two question components"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(components)).tolist()
    selected: set[int] = set()
    for component_index in order:
        component = components[component_index]
        if len(selected) + len(component) >= len(df):
            continue
        selected.update(component)
        if len(selected) >= n_eval:
            break

    if not selected:
        raise ValueError("unable to select a non-empty question-component holdout")
    return selected


def split_train_eval(
    df: pd.DataFrame,
    *,
    eval_fraction: float,
    seed: int,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Shuffle and split a labeled frame, avoiding question overlap when IDs exist."""
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be at least 1")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_rows is not None:
        shuffled = shuffled.head(max_rows).copy()

    n = len(shuffled)
    if eval_fraction == 0 or n < 20:
        return shuffled, None, "none"

    n_eval = max(1, min(int(n * eval_fraction), n - 1))
    if _has_question_ids(shuffled):
        eval_indices = _component_holdout_indices(shuffled, n_eval=n_eval, seed=seed)
        eval_mask = shuffled.index.isin(eval_indices)
        return (
            shuffled.loc[~eval_mask].reset_index(drop=True),
            shuffled.loc[eval_mask].reset_index(drop=True),
            "question_component_holdout",
        )

    return (
        shuffled.iloc[n_eval:].copy(),
        shuffled.iloc[:n_eval].copy(),
        "shuffled_prefix_holdout",
    )
