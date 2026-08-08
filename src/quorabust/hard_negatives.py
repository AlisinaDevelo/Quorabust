"""Deterministic hard-negative mining for pair-model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quorabust.preprocess import clean_text
from quorabust.retrieval import (
    CatalogQuestion,
    CatalogRetriever,
    SentenceTransformerCatalogRetriever,
    TfidfCatalogRetriever,
)

HARD_NEGATIVE_SCHEMA_VERSION = 1
_REQUIRED_COLUMNS = ("question1", "question2", "is_duplicate", "qid1", "qid2")
_OUTPUT_COLUMNS = (
    "question1",
    "question2",
    "is_duplicate",
    "qid1",
    "qid2",
    "source_positive_row",
    "source_positive_qid1",
    "source_positive_qid2",
    "anchor_side",
    "retrieval_rank",
    "retrieval_score",
)


@dataclass(frozen=True)
class HardNegativeMiningResult:
    """Generated negatives and path-light mining metadata."""

    pairs: pd.DataFrame
    metadata: dict[str, Any]


def _require_positive_integer(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_seed(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _normalized_text(value: Any, label: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{label} must not be empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _normalized_qid(value: Any, label: str) -> str:
    return _normalized_text(value, label)


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError("hard-negative mining requires columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("hard-negative mining requires at least one row")
    if not df["is_duplicate"].isin([0, 1]).all():
        raise ValueError("hard-negative mining requires binary is_duplicate labels")

    frame = df.reset_index(drop=True).copy()
    for column in ("question1", "question2", "qid1", "qid2"):
        frame[column] = [
            _normalized_text(value, f"{column}[{index}]")
            for index, value in enumerate(frame[column].tolist())
        ]
    frame["is_duplicate"] = frame["is_duplicate"].astype(int)
    return frame


def _build_catalog(frame: pd.DataFrame) -> list[CatalogQuestion]:
    question_text_by_id: dict[str, str] = {}
    for row in frame[["question1", "question2", "qid1", "qid2"]].itertuples(
        index=False,
        name=None,
    ):
        question1, question2, qid1, qid2 = row
        for question_id, text, column in (
            (qid1, question1, "question1"),
            (qid2, question2, "question2"),
        ):
            existing = question_text_by_id.get(question_id)
            if existing is not None and clean_text(existing) != clean_text(text):
                raise ValueError(
                    f"question ID {question_id!r} has conflicting text in {column}"
                )
            question_text_by_id.setdefault(question_id, text)
    return [
        CatalogQuestion(question_id=question_id, text=question_text_by_id[question_id])
        for question_id in sorted(question_text_by_id)
    ]


def _positive_components(frame: pd.DataFrame) -> dict[str, frozenset[str]]:
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

    for left, right in frame[["qid1", "qid2"]].itertuples(index=False, name=None):
        find(left)
        find(right)
    for left, right in frame.loc[
        frame["is_duplicate"].eq(1), ["qid1", "qid2"]
    ].itertuples(index=False, name=None):
        union(left, right)

    members: dict[str, set[str]] = {}
    for question_id in parent:
        members.setdefault(find(question_id), set()).add(question_id)
    return {
        question_id: frozenset(members[find(question_id)])
        for question_id in parent
    }


def _candidate_rows(
    *,
    positive_row: int,
    positive_qid1: str,
    positive_qid2: str,
    anchor_side: str,
    anchor_qid: str,
    anchor_text: str,
    component_ids: frozenset[str],
    retriever: CatalogRetriever,
    candidate_k: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rank, hit in enumerate(
        retriever.search(anchor_text, k=min(candidate_k, retriever.size)),
        start=1,
    ):
        if hit.question_id in component_ids:
            continue
        candidates.append(
            {
                "question1": anchor_text,
                "question2": hit.text,
                "is_duplicate": 0,
                "qid1": anchor_qid,
                "qid2": hit.question_id,
                "source_positive_row": positive_row,
                "source_positive_qid1": positive_qid1,
                "source_positive_qid2": positive_qid2,
                "anchor_side": anchor_side,
                "retrieval_rank": rank,
                "retrieval_score": float(hit.retrieval_score),
            }
        )
    return candidates


def mine_hard_negatives(
    df: pd.DataFrame,
    *,
    candidate_k: int = 50,
    negatives_per_positive: int = 1,
    max_positive_rows: int | None = None,
    seed: int = 42,
    retriever_backend: str = "tfidf",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HardNegativeMiningResult:
    """Mine retrieval near-neighbour negatives while excluding known-positive components.

    The input must contain only material that is allowed to influence training or tuning.
    In particular, callers must keep the final evaluation holdout out of this frame.
    """
    candidate_k = _require_positive_integer(candidate_k, "candidate_k")
    negatives_per_positive = _require_positive_integer(
        negatives_per_positive,
        "negatives_per_positive",
    )
    seed = _require_seed(seed)
    if retriever_backend not in {"tfidf", "embedding"}:
        raise ValueError("retriever_backend must be one of: tfidf, embedding")
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("embedding_model must be a non-empty string")
    if max_positive_rows is not None:
        max_positive_rows = _require_positive_integer(max_positive_rows, "max_positive_rows")

    frame = _validate_frame(df)
    catalog = _build_catalog(frame)
    if len(catalog) < 2:
        raise ValueError("hard-negative mining requires at least two unique questions")
    components = _positive_components(frame)
    positive_rows = frame.index[frame["is_duplicate"].eq(1)].tolist()
    if not positive_rows:
        raise ValueError("hard-negative mining requires at least one positive pair")
    if max_positive_rows is not None and max_positive_rows < len(positive_rows):
        rng = np.random.default_rng(seed)
        positive_rows = sorted(
            int(index) for index in rng.choice(positive_rows, size=max_positive_rows, replace=False)
        )

    if retriever_backend == "embedding":
        retriever: CatalogRetriever = SentenceTransformerCatalogRetriever(
            embedding_model.strip()
        ).fit(catalog)
    else:
        retriever = TfidfCatalogRetriever().fit(catalog)
    generated: list[dict[str, Any]] = []
    emitted_pairs: set[tuple[str, str]] = set()
    for positive_row in positive_rows:
        source = frame.iloc[positive_row]
        positive_qid1 = str(source["qid1"])
        positive_qid2 = str(source["qid2"])
        component_ids = components[positive_qid1]
        candidates: list[dict[str, Any]] = []
        for anchor_side, anchor_qid, anchor_text in (
            ("qid1", positive_qid1, str(source["question1"])),
            ("qid2", positive_qid2, str(source["question2"])),
        ):
            candidates.extend(
                _candidate_rows(
                    positive_row=positive_row,
                    positive_qid1=positive_qid1,
                    positive_qid2=positive_qid2,
                    anchor_side=anchor_side,
                    anchor_qid=anchor_qid,
                    anchor_text=anchor_text,
                    component_ids=component_ids,
                    retriever=retriever,
                    candidate_k=candidate_k,
                )
            )

        candidates.sort(
            key=lambda row: (
                -float(row["retrieval_score"]),
                int(row["retrieval_rank"]),
                str(row["anchor_side"]),
                str(row["qid1"]),
                str(row["qid2"]),
            )
        )
        selected = 0
        for candidate in candidates:
            sorted_pair = sorted((str(candidate["qid1"]), str(candidate["qid2"])))
            pair_key: tuple[str, str] = (sorted_pair[0], sorted_pair[1])
            if pair_key in emitted_pairs:
                continue
            emitted_pairs.add(pair_key)
            generated.append(candidate)
            selected += 1
            if selected >= negatives_per_positive:
                break

    if not generated:
        raise ValueError(
            "no eligible hard negatives were found; provide more than one positive component"
        )
    pairs = pd.DataFrame(generated, columns=_OUTPUT_COLUMNS)
    metadata: dict[str, Any] = {
        "schema_version": HARD_NEGATIVE_SCHEMA_VERSION,
        "manifest": "quorabust.hard_negative_mining",
        "config": {
            "retriever": retriever_backend,
            "model_name": embedding_model.strip() if retriever_backend == "embedding" else None,
            "candidate_k": candidate_k,
            "negatives_per_positive": negatives_per_positive,
            "max_positive_rows": max_positive_rows,
            "seed": seed,
        },
        "input": {
            "rows": int(len(frame)),
            "positive_rows": int(frame["is_duplicate"].eq(1).sum()),
            "positive_rows_considered": int(len(positive_rows)),
            "catalog_size": int(len(catalog)),
            "positive_component_count": int(
                len({frozenset(value) for value in components.values()})
            ),
        },
        "output": {
            "rows": int(len(pairs)),
            "label_counts": {"0": int(len(pairs)), "1": 0},
        },
        "safeguards": {
            "positive_edges_only": True,
            "anchor_positive_component_excluded": True,
            "final_holdout_used": False,
            "raw_data_committed": False,
        },
    }
    return HardNegativeMiningResult(pairs=pairs, metadata=metadata)
