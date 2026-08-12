from __future__ import annotations

import platform
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from quorabust.lineage import git_revision
from quorabust.model import predict_proba_duplicate
from quorabust.retrieval_benchmark import peak_rss_bytes, summarize_latencies_ms

_PAIR_LENGTH_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("short", 1, 5),
    ("medium", 6, 15),
    ("long", 16, None),
)


def _pair_length_policy() -> dict[str, object]:
    return {
        "tokenization": "python_str_split_whitespace",
        "measure": "maximum_whitespace_token_count_of_question1_and_question2",
        "buckets": {
            name: {"min_tokens": minimum, "max_tokens": maximum}
            for name, minimum, maximum in _PAIR_LENGTH_BUCKETS
        },
    }


def pair_length_bucket(question1: str, question2: str) -> str:
    """Classify a pair by the longer question's whitespace-token count."""
    token_count = max(len(question1.split()), len(question2.split()))
    if token_count < 1:
        raise ValueError("question pairs must contain at least one whitespace token")
    for name, minimum, maximum in _PAIR_LENGTH_BUCKETS:
        if token_count >= minimum and (maximum is None or token_count <= maximum):
            return name
    raise RuntimeError(f"no pair-length bucket covers {token_count} tokens")


def normalize_pair_frame(
    frame: pd.DataFrame,
    *,
    label_column: str = "is_duplicate",
) -> pd.DataFrame:
    """Validate and normalize the pair-profile input without retaining source paths."""
    required = {"question1", "question2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns {missing}; found {list(frame.columns)}")
    if frame.empty:
        raise ValueError("evaluation CSV must contain at least one pair")

    normalized = frame.copy()
    for column in ("question1", "question2"):
        values = normalized[column]
        if values.isna().any() or values.astype("string").str.strip().eq("").any():
            raise ValueError(f"{column} values must not be empty")
        normalized[column] = values.astype("string").str.strip().astype(str)

    if label_column in normalized.columns:
        labels = pd.to_numeric(normalized[label_column], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("labels must contain only 0 and 1")
        normalized[label_column] = labels.astype(int)
    return normalized.reset_index(drop=True)


def _batch_indices(indexes: Sequence[int], batch_size: int) -> list[list[int]]:
    return [
        list(indexes[start : start + batch_size])
        for start in range(0, len(indexes), batch_size)
    ]


def _quality_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | dict[str, int]]:
    predictions = (probabilities >= threshold).astype(int)
    unique_labels = np.unique(labels)
    metrics: dict[str, float | int | dict[str, int]] = {
        "rows": int(len(labels)),
        "label_counts": {
            "0": int(np.sum(labels == 0)),
            "1": int(np.sum(labels == 1)),
        },
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    if len(unique_labels) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
    return metrics


def benchmark_pair_classifier(
    builder: Any,
    classifier: Any,
    frame: pd.DataFrame,
    *,
    batch_size: int = 32,
    warmup_runs: int = 1,
    repetitions: int = 3,
    timeout_seconds: float | None = 120.0,
    label_column: str = "is_duplicate",
    threshold: float = 0.5,
) -> dict[str, object]:
    """Measure serial warm pair scoring with explicit batching and length strata."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be zero or greater")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if timeout_seconds is not None and timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be greater than zero")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    normalized = normalize_pair_frame(frame, label_column=label_column)
    question1 = normalized["question1"].tolist()
    question2 = normalized["question2"].tolist()
    indexes_by_bucket: dict[str, list[int]] = {name: [] for name, _, _ in _PAIR_LENGTH_BUCKETS}
    token_counts: dict[str, list[int]] = {name: [] for name, _, _ in _PAIR_LENGTH_BUCKETS}
    for index, (left, right) in enumerate(zip(question1, question2, strict=True)):
        bucket = pair_length_bucket(left, right)
        indexes_by_bucket[bucket].append(index)
        token_counts[bucket].append(max(len(left.split()), len(right.split())))

    batches: list[tuple[str, list[int]]] = []
    for name, _, _ in _PAIR_LENGTH_BUCKETS:
        batches.extend(
            (name, indexes)
            for indexes in _batch_indices(indexes_by_bucket[name], batch_size)
        )

    started = time.perf_counter()
    deadline = started + timeout_seconds if timeout_seconds is not None else None

    def check_deadline() -> None:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError(f"pair profile exceeded {timeout_seconds:g} seconds")

    def score(indexes: list[int]) -> np.ndarray:
        probabilities = np.asarray(
            predict_proba_duplicate(
                builder,
                classifier,
                [question1[index] for index in indexes],
                [question2[index] for index in indexes],
            ),
            dtype=np.float64,
        )
        if probabilities.ndim != 2 or probabilities.shape != (len(indexes), 2):
            raise ValueError("classifier must return one binary probability pair per input")
        if not np.isfinite(probabilities).all():
            raise ValueError("classifier probabilities must be finite")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("classifier probabilities must be between 0 and 1")
        return probabilities[:, 1]

    for _ in range(warmup_runs):
        for _, indexes in batches:
            check_deadline()
            score(indexes)

    batch_latencies: list[float] = []
    per_pair_latencies: list[float] = []
    latencies_by_bucket: dict[str, list[float]] = {name: [] for name, _, _ in _PAIR_LENGTH_BUCKETS}
    per_pair_by_bucket: dict[str, list[float]] = {
        name: [] for name, _, _ in _PAIR_LENGTH_BUCKETS
    }
    first_probabilities = np.empty(len(normalized), dtype=np.float64)
    for repetition in range(repetitions):
        for bucket, indexes in batches:
            check_deadline()
            batch_started = time.perf_counter()
            probabilities = score(indexes)
            elapsed_ms = (time.perf_counter() - batch_started) * 1000.0
            batch_latencies.append(elapsed_ms)
            per_pair_ms = elapsed_ms / len(indexes)
            per_pair_latencies.append(per_pair_ms)
            latencies_by_bucket[bucket].append(elapsed_ms)
            per_pair_by_bucket[bucket].append(per_pair_ms)
            if repetition == 0:
                first_probabilities[indexes] = probabilities

    def throughput(pair_count: int, latencies: Sequence[float]) -> float | None:
        seconds = sum(latencies) / 1000.0
        return float(pair_count / seconds) if seconds > 0.0 else None

    strata: dict[str, object] = {}
    for name, _, _ in _PAIR_LENGTH_BUCKETS:
        indexes = indexes_by_bucket[name]
        if not indexes:
            continue
        strata[name] = {
            "pair_count": int(len(indexes)),
            "measured_pair_count": int(len(indexes) * repetitions),
            "measurement_count": int(len(latencies_by_bucket[name])),
            "token_count": {
                "min": int(min(token_counts[name])),
                "max": int(max(token_counts[name])),
            },
            "latency_ms": {
                "batch": summarize_latencies_ms(latencies_by_bucket[name]),
                "per_pair": summarize_latencies_ms(per_pair_by_bucket[name]),
            },
            "work": {
                "throughput_pairs_per_second": throughput(
                    len(indexes) * repetitions,
                    latencies_by_bucket[name],
                )
            },
        }

    labels: dict[str, object]
    quality: dict[str, float | int | dict[str, int]] | None
    if label_column in normalized.columns:
        label_values = normalized[label_column].to_numpy(dtype=np.int64)
        labels = {
            "present": True,
            "column": label_column,
            "counts": {
                "0": int(np.sum(label_values == 0)),
                "1": int(np.sum(label_values == 1)),
            },
        }
        quality = _quality_metrics(label_values, first_probabilities, threshold)
    else:
        labels = {"present": False, "column": label_column, "counts": None}
        quality = None

    return {
        "pair_count": int(len(normalized)),
        "measured_pair_count": int(len(normalized) * repetitions),
        "measurement_count": int(len(batch_latencies)),
        "batch_size": int(batch_size),
        "latency_ms": {
            "batch": summarize_latencies_ms(batch_latencies),
            "per_pair": summarize_latencies_ms(per_pair_latencies),
        },
        "pair_length_policy": _pair_length_policy(),
        "pair_length_strata": strata,
        "labels": labels,
        "quality": quality,
        "work": {
            "throughput_pairs_per_second": throughput(
                len(normalized) * repetitions,
                batch_latencies,
            ),
            "concurrency": 1,
            "execution": "serial",
        },
        "measurement_policy": {
            "warmup_runs": int(warmup_runs),
            "repetitions": int(repetitions),
            "timeout_seconds": timeout_seconds,
            "concurrency": 1,
            "execution": "serial",
            "timeout_behavior": "cooperative_deadline_between_batches",
        },
        "runtime": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "profile_git_revision": git_revision(),
            "peak_rss_bytes": peak_rss_bytes(),
            "rss_measurement": "process_maxrss_since_process_start",
        },
    }
