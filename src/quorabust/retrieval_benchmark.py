from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from quorabust.lineage import git_revision, sha256_file
from quorabust.retrieval import (
    CatalogRetriever,
    ScoreBatch,
    rerank_candidates,
)


@dataclass(frozen=True)
class RetrievalCase:
    """A query and its relevant catalog IDs, optionally with graded relevance."""

    query: str
    relevance: Mapping[str, float]


_QUERY_LENGTH_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("short", 1, 5),
    ("medium", 6, 15),
    ("long", 16, None),
)


def _query_token_count(query: str) -> int:
    return len(query.split())


def query_length_bucket(query: str) -> str:
    """Classify a non-empty query by its number of whitespace-delimited tokens."""
    token_count = _query_token_count(query)
    if token_count < 1:
        raise ValueError("query must contain at least one whitespace token")
    for name, minimum, maximum in _QUERY_LENGTH_BUCKETS:
        if token_count >= minimum and (maximum is None or token_count <= maximum):
            return name
    raise RuntimeError(f"no query-length bucket covers {token_count} tokens")


def _query_length_policy() -> dict[str, object]:
    return {
        "tokenization": "python_str_split_whitespace",
        "buckets": {
            name: {"min_tokens": minimum, "max_tokens": maximum}
            for name, minimum, maximum in _QUERY_LENGTH_BUCKETS
        },
    }


def _query_length_strata(
    cases: Sequence[RetrievalCase],
    retrieval_latencies: Sequence[float],
    rerank_latencies: Sequence[float],
    end_to_end_latencies: Sequence[float],
    *,
    repetitions: int,
) -> dict[str, object]:
    """Summarize measured latency by deterministic query-length bucket."""
    expected_samples = len(cases) * repetitions
    samples = (retrieval_latencies, rerank_latencies, end_to_end_latencies)
    if any(len(values) != expected_samples for values in samples):
        raise ValueError("latency samples must contain every measured query")

    indexes_by_bucket: dict[str, list[int]] = {
        name: [] for name, _, _ in _QUERY_LENGTH_BUCKETS
    }
    for index, case in enumerate(cases):
        indexes_by_bucket[query_length_bucket(case.query)].append(index)

    strata: dict[str, object] = {}
    for name, _, _ in _QUERY_LENGTH_BUCKETS:
        case_indexes = indexes_by_bucket[name]
        if not case_indexes:
            continue
        bucket_samples: list[list[float]] = [[], [], []]
        for repetition in range(repetitions):
            offset = repetition * len(cases)
            for stage_index, stage_values in enumerate(samples):
                bucket_samples[stage_index].extend(
                    stage_values[offset + case_index] for case_index in case_indexes
                )

        end_to_end = bucket_samples[2]
        total_seconds = sum(end_to_end) / 1000.0
        token_counts = [_query_token_count(cases[index].query) for index in case_indexes]
        strata[name] = {
            "query_count": int(len(case_indexes)),
            "measured_query_count": int(len(case_indexes) * repetitions),
            "token_count": {
                "min": int(min(token_counts)),
                "max": int(max(token_counts)),
            },
            "latency_ms": {
                "retrieval": summarize_latencies_ms(bucket_samples[0]),
                "rerank": summarize_latencies_ms(bucket_samples[1]),
                "end_to_end": summarize_latencies_ms(end_to_end),
            },
            "work": {
                "throughput_queries_per_second": (
                    float((len(case_indexes) * repetitions) / total_seconds)
                    if total_seconds > 0.0
                    else None
                ),
            },
        }
    return strata


def _validated_ks(ks: Sequence[int]) -> list[int]:
    values = sorted(set(ks))
    if not values or any(value < 1 for value in values):
        raise ValueError("ks must contain at least one positive integer")
    return values


def _validated_cases(cases: Sequence[RetrievalCase]) -> list[RetrievalCase]:
    if not cases:
        raise ValueError("at least one retrieval case is required")
    normalized: list[RetrievalCase] = []
    for case in cases:
        query = str(case.query).strip()
        if not query:
            raise ValueError("retrieval queries must not be empty")
        if not case.relevance:
            raise ValueError("each retrieval case must contain a relevant catalog ID")
        relevance: dict[str, float] = {}
        for question_id, raw_value in case.relevance.items():
            normalized_id = str(question_id).strip()
            value = float(raw_value)
            if not normalized_id:
                raise ValueError("relevant catalog IDs must not be empty")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("relevance values must be finite and non-negative")
            relevance[normalized_id] = value
        if not any(value > 0.0 for value in relevance.values()):
            raise ValueError("each retrieval case needs at least one positive relevance")
        normalized.append(RetrievalCase(query=query, relevance=relevance))
    return normalized


def _dcg(gains: Sequence[float]) -> float:
    total = 0.0
    for index, gain in enumerate(gains):
        total += (2.0**gain - 1.0) / math.log2(index + 2.0)
    return total


def evaluate_rankings(
    rankings: Sequence[Sequence[str]],
    cases: Sequence[RetrievalCase],
    *,
    ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Evaluate ranked catalog IDs with recall, MRR, and NDCG at each k."""
    normalized_cases = _validated_cases(cases)
    values = _validated_ks(ks)
    if len(rankings) != len(normalized_cases):
        raise ValueError("rankings and retrieval cases must have the same length")

    metrics: dict[str, dict[str, float]] = {
        "recall_at_k": {},
        "mrr_at_k": {},
        "ndcg_at_k": {},
    }
    for k in values:
        recall_values: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcg_values: list[float] = []
        for ranking, case in zip(rankings, normalized_cases, strict=True):
            gains = [float(case.relevance.get(question_id, 0.0)) for question_id in ranking[:k]]
            relevant_count = sum(value > 0.0 for value in case.relevance.values())
            recall_values.append(sum(value > 0.0 for value in gains) / relevant_count)
            reciprocal_ranks.append(
                next(
                    (1.0 / (index + 1.0) for index, value in enumerate(gains) if value > 0.0),
                    0.0,
                )
            )
            ideal_gains = sorted(case.relevance.values(), reverse=True)[:k]
            ideal_dcg = _dcg(ideal_gains)
            ndcg_values.append(_dcg(gains) / ideal_dcg if ideal_dcg else 0.0)

        key = str(k)
        metrics["recall_at_k"][key] = float(np.mean(recall_values))
        metrics["mrr_at_k"][key] = float(np.mean(reciprocal_ranks))
        metrics["ndcg_at_k"][key] = float(np.mean(ndcg_values))
    return metrics


def summarize_latencies_ms(samples: Sequence[float]) -> dict[str, float | int]:
    """Summarize positive elapsed-time samples without hiding zero-work stages."""
    if not samples:
        raise ValueError("at least one latency sample is required")
    values = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def peak_rss_bytes() -> int | None:
    """Return process peak RSS in bytes when the Unix resource API is available."""
    try:
        import resource

        raw_value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError):
        return None
    if not math.isfinite(raw_value) or raw_value <= 0.0:
        return None
    # macOS reports bytes; Linux and the other supported Unix targets report KiB.
    multiplier = 1 if platform.system() == "Darwin" else 1024
    return int(raw_value * multiplier)


def load_retrieval_qrels(
    path: str,
    *,
    catalog_ids: Collection[str] | None = None,
) -> list[RetrievalCase]:
    """Load a qrels CSV with ``query``, ``question_id``, and optional ``relevance``."""
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"query", "question_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing qrels columns {missing}; found {list(frame.columns)}")
    if frame.empty:
        raise ValueError("qrels CSV must contain at least one row")

    allowed_ids = {str(value).strip() for value in catalog_ids} if catalog_ids is not None else None
    grouped: dict[str, dict[str, float]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        raw_query = getattr(row, "query")
        raw_question_id = getattr(row, "question_id")
        if pd.isna(raw_query) or pd.isna(raw_question_id):
            raise ValueError("qrels query and question_id values must not be empty")
        query = str(raw_query).strip()
        question_id = str(raw_question_id).strip()
        if not query or not question_id:
            raise ValueError("qrels query and question_id values must not be empty")
        if allowed_ids is not None and question_id not in allowed_ids:
            raise ValueError(f"qrels references unknown catalog question ID: {question_id}")
        pair = (query, question_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate qrels pair: {query!r}, {question_id!r}")
        seen_pairs.add(pair)

        relevance = 1.0
        if "relevance" in frame.columns:
            raw_relevance = getattr(row, "relevance")
            if pd.isna(raw_relevance):
                raise ValueError("qrels relevance values must not be empty")
            try:
                relevance = float(raw_relevance)
            except (TypeError, ValueError) as exc:
                raise ValueError("qrels relevance values must be numeric") from exc
        if not math.isfinite(relevance) or relevance < 0.0:
            raise ValueError("qrels relevance values must be finite and non-negative")
        grouped.setdefault(query, {})[question_id] = relevance

    return [RetrievalCase(query=query, relevance=relevance) for query, relevance in grouped.items()]


def benchmark_retrieval(
    retriever: CatalogRetriever,
    cases: Sequence[RetrievalCase],
    *,
    ks: Sequence[int],
    candidate_k: int,
    score_batch: ScoreBatch | None = None,
    warmup_runs: int = 1,
    repetitions: int = 3,
    timeout_seconds: float | None = None,
) -> dict[str, object]:
    """Measure ranking quality and repeated per-stage latency.

    Quality is evaluated once from the first measured pass so repeated latency samples
    do not overweight any query. Warm-up and measured passes are serial and process-local;
    the timeout is a cooperative wall-clock deadline checked between queries/stages.
    """
    normalized_cases = _validated_cases(cases)
    values = _validated_ks(ks)
    if candidate_k < max(values):
        raise ValueError("candidate_k must be greater than or equal to the largest k")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")
    if isinstance(warmup_runs, bool) or warmup_runs < 0:
        raise ValueError("warmup_runs must be a non-negative integer")
    if isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if timeout_seconds is not None and (
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0
    ):
        raise ValueError("timeout_seconds must be finite and positive")

    deadline = time.perf_counter() + timeout_seconds if timeout_seconds is not None else None

    def check_deadline() -> None:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("retrieval benchmark exceeded timeout_seconds")

    def run_pass() -> tuple[
        list[list[str]], list[list[str]], list[float], list[float], list[float], int
    ]:
        first_stage_rankings: list[list[str]] = []
        final_rankings: list[list[str]] = []
        retrieval_latencies: list[float] = []
        rerank_latencies: list[float] = []
        end_to_end_latencies: list[float] = []
        rerank_pair_count = 0

        for case in normalized_cases:
            check_deadline()
            end_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            candidates = retriever.search(case.query, k=candidate_k)
            retrieval_latencies.append((time.perf_counter() - retrieval_start) * 1000.0)
            first_stage_rankings.append([hit.question_id for hit in candidates])
            check_deadline()

            if score_batch is None:
                reranked = candidates
                rerank_latencies.append(0.0)
            else:
                rerank_start = time.perf_counter()
                reranked = rerank_candidates(case.query, candidates, score_batch)
                rerank_latencies.append((time.perf_counter() - rerank_start) * 1000.0)
                rerank_pair_count += len(candidates)
            final_rankings.append([hit.question_id for hit in reranked])
            end_to_end_latencies.append((time.perf_counter() - end_start) * 1000.0)
            check_deadline()

        return (
            first_stage_rankings,
            final_rankings,
            retrieval_latencies,
            rerank_latencies,
            end_to_end_latencies,
            rerank_pair_count,
        )

    for _ in range(warmup_runs):
        check_deadline()
        run_pass()

    first_stage_rankings: list[list[str]] | None = None
    final_rankings: list[list[str]] | None = None
    retrieval_latencies: list[float] = []
    rerank_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    rerank_pair_count = 0
    for _ in range(repetitions):
        check_deadline()
        measured = run_pass()
        if first_stage_rankings is None:
            first_stage_rankings = measured[0]
            final_rankings = measured[1]
        retrieval_latencies.extend(measured[2])
        rerank_latencies.extend(measured[3])
        end_to_end_latencies.extend(measured[4])
        rerank_pair_count += measured[5]

    if first_stage_rankings is None or final_rankings is None:
        raise RuntimeError("benchmark did not produce a measured pass")

    total_seconds = sum(end_to_end_latencies) / 1000.0
    return {
        "catalog_size": int(retriever.size),
        "query_count": int(len(normalized_cases)),
        "measured_query_count": int(len(normalized_cases) * repetitions),
        "candidate_k": int(candidate_k),
        "first_stage": evaluate_rankings(first_stage_rankings, normalized_cases, ks=values),
        "final": evaluate_rankings(final_rankings, normalized_cases, ks=values),
        "latency_ms": {
            "retrieval": summarize_latencies_ms(retrieval_latencies),
            "rerank": summarize_latencies_ms(rerank_latencies),
            "end_to_end": summarize_latencies_ms(end_to_end_latencies),
        },
        "query_length_policy": _query_length_policy(),
        "query_length_strata": _query_length_strata(
            normalized_cases,
            retrieval_latencies,
            rerank_latencies,
            end_to_end_latencies,
            repetitions=repetitions,
        ),
        "work": {
            "reranker_enabled": score_batch is not None,
            "reranker_pairs": int(rerank_pair_count),
            "mean_candidates": float(
                sum(len(ranking) for ranking in first_stage_rankings) / len(first_stage_rankings)
            ),
            "throughput_queries_per_second": (
                float((len(normalized_cases) * repetitions) / total_seconds)
                if total_seconds > 0.0
                else None
            ),
        },
        "measurement_policy": {
            "warmup_runs": int(warmup_runs),
            "repetitions": int(repetitions),
            "quality_passes": 1,
            "latency_samples_per_stage": int(len(normalized_cases) * repetitions),
            "timeout_seconds": timeout_seconds,
            "concurrency": 1,
            "execution": "serial",
            "timeout_behavior": "cooperative_deadline_between_queries_and_stages",
        },
        "runtime": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "benchmark_git_revision": git_revision(),
            "peak_rss_bytes": peak_rss_bytes(),
            "rss_measurement": "process_maxrss_since_process_start",
        },
    }


def source_manifest(path: str) -> dict[str, object]:
    """Return path-light source metadata suitable for a benchmark artifact."""
    from pathlib import Path

    source = Path(path)
    return {
        "name": source.name,
        "sha256": sha256_file(source),
        "bytes": int(source.stat().st_size),
    }
