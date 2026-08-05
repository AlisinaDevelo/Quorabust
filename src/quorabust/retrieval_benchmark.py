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
        "max": float(np.max(values)),
    }


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
) -> dict[str, object]:
    """Measure first-stage/final ranking quality and per-stage latency."""
    normalized_cases = _validated_cases(cases)
    values = _validated_ks(ks)
    if candidate_k < max(values):
        raise ValueError("candidate_k must be greater than or equal to the largest k")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")

    first_stage_rankings: list[list[str]] = []
    final_rankings: list[list[str]] = []
    retrieval_latencies: list[float] = []
    rerank_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    rerank_pair_count = 0

    for case in normalized_cases:
        end_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        candidates = retriever.search(case.query, k=candidate_k)
        retrieval_latencies.append((time.perf_counter() - retrieval_start) * 1000.0)
        first_stage_rankings.append([hit.question_id for hit in candidates])

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

    total_seconds = sum(end_to_end_latencies) / 1000.0
    return {
        "catalog_size": int(retriever.size),
        "query_count": int(len(normalized_cases)),
        "candidate_k": int(candidate_k),
        "first_stage": evaluate_rankings(first_stage_rankings, normalized_cases, ks=values),
        "final": evaluate_rankings(final_rankings, normalized_cases, ks=values),
        "latency_ms": {
            "retrieval": summarize_latencies_ms(retrieval_latencies),
            "rerank": summarize_latencies_ms(rerank_latencies),
            "end_to_end": summarize_latencies_ms(end_to_end_latencies),
        },
        "work": {
            "reranker_enabled": score_batch is not None,
            "reranker_pairs": int(rerank_pair_count),
            "mean_candidates": float(
                sum(len(ranking) for ranking in first_stage_rankings) / len(first_stage_rankings)
            ),
            "throughput_queries_per_second": (
                float(len(normalized_cases) / total_seconds) if total_seconds > 0.0 else None
            ),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "benchmark_git_revision": git_revision(),
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
