from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

_QUESTION_ID_COLUMNS = ("qid1", "qid2")
_ROLE_NAMES = ("train", "tuning", "calibration", "final_holdout")
_NON_TRAIN_ROLES = ("final_holdout", "calibration", "tuning")


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


def _require_positive_fraction(value: float, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError(f"{label} must be finite and strictly between 0 and 1")
    return float(value)


def split_question_component_roles(
    df: pd.DataFrame,
    *,
    tuning_fraction: float = 0.1,
    calibration_fraction: float = 0.1,
    final_holdout_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Materialize four disjoint roles from complete question components."""
    required_columns = {"question1", "question2", "is_duplicate", *_QUESTION_ID_COLUMNS}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError("benchmark split requires columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("benchmark split requires at least one row")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    fractions = {
        "tuning": _require_positive_fraction(tuning_fraction, "tuning_fraction"),
        "calibration": _require_positive_fraction(
            calibration_fraction,
            "calibration_fraction",
        ),
        "final_holdout": _require_positive_fraction(
            final_holdout_fraction,
            "final_holdout_fraction",
        ),
    }
    if sum(fractions.values()) >= 1.0:
        raise ValueError("non-training role fractions must sum to less than 1")

    complete_ids = df[list(_QUESTION_ID_COLUMNS)].notna().all(axis=1)
    for column in _QUESTION_ID_COLUMNS:
        complete_ids &= df[column].astype("string").str.strip().ne("").fillna(False)
    if not bool(complete_ids.all()):
        raise ValueError("benchmark split requires complete non-blank qid1/qid2 values")
    if not df["is_duplicate"].isin([0, 1]).all():
        raise ValueError("benchmark split requires binary is_duplicate labels")

    frame = df.reset_index(drop=True)
    components = _question_components(frame)
    if len(components) < len(_ROLE_NAMES):
        raise ValueError(
            "benchmark split requires at least four question components for the four roles"
        )

    targets = {role: max(1, int(len(frame) * fraction)) for role, fraction in fractions.items()}
    rng = np.random.default_rng(seed)
    remaining = [components[index] for index in rng.permutation(len(components)).tolist()]
    assigned: dict[str, list[int]] = {role: [] for role in _ROLE_NAMES}
    component_counts = {role: 0 for role in _ROLE_NAMES}

    for position, role in enumerate(_NON_TRAIN_ROLES):
        roles_after = len(_NON_TRAIN_ROLES) - position - 1 + 1
        while len(assigned[role]) < targets[role]:
            if len(remaining) <= roles_after:
                raise ValueError("unable to reserve non-empty components for every role")
            component = remaining.pop(0)
            assigned[role].extend(component)
            component_counts[role] += 1
    for component in remaining:
        assigned["train"].extend(component)
        component_counts["train"] += 1

    roles: dict[str, pd.DataFrame] = {}
    for role in _ROLE_NAMES:
        indices = sorted(assigned[role])
        if not indices:
            raise ValueError(f"unable to materialize a non-empty {role} role")
        roles[role] = frame.iloc[indices].reset_index(drop=True)

    seen_question_ids: set[str] = set()
    for role in _ROLE_NAMES:
        role_question_ids = set(
            roles[role][list(_QUESTION_ID_COLUMNS)].astype(str).to_numpy().reshape(-1)
        )
        if seen_question_ids.intersection(role_question_ids):
            raise ValueError("question components overlap across benchmark roles")
        seen_question_ids.update(role_question_ids)

    requested_fractions = {
        "train": 1.0 - sum(fractions.values()),
        **fractions,
    }
    stats: dict[str, Any] = {
        "strategy": "question_component_holdout",
        "question_id_columns": list(_QUESTION_ID_COLUMNS),
        "seed": seed,
        "component_count": len(components),
        "requested_fractions": requested_fractions,
        "roles_are_disjoint": True,
        "roles": {
            role: {
                "rows": len(roles[role]),
                "components": component_counts[role],
                "observed_fraction": len(roles[role]) / len(frame),
            }
            for role in _ROLE_NAMES
        },
    }
    return roles, stats


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
