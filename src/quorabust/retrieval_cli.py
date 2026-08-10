from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.retrieval import (
    CatalogRetriever,
    SentenceTransformerCatalogRetriever,
    SentenceTransformerCrossEncoderReranker,
    TfidfCatalogRetriever,
    search_and_rerank,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("retrieval request exceeded timeout_seconds")


def _write_payload(payload: dict[str, Any], out: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out is None:
        print(rendered, end="")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve likely matches from a question catalog.",
    )
    parser.add_argument("--catalog-csv", type=Path, required=True)
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Query text; repeat the flag to score multiple queries in one run",
    )
    parser.add_argument("--k", type=_positive_int, default=10)
    parser.add_argument(
        "--candidate-k",
        type=_positive_int,
        default=50,
        help="Bound the first-stage candidates passed to an optional reranker",
    )
    parser.add_argument(
        "--max-queries",
        type=_positive_int,
        default=32,
        help="Maximum number of queries accepted in one request",
    )
    parser.add_argument(
        "--max-k",
        type=_positive_int,
        default=100,
        help="Maximum number of returned hits per query",
    )
    parser.add_argument(
        "--max-candidate-k",
        type=_positive_int,
        default=100,
        help="Maximum first-stage candidate set per query",
    )
    parser.add_argument(
        "--max-query-chars",
        type=_positive_int,
        default=8192,
        help="Maximum normalized query length in characters",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=None,
        help="Cooperative query execution timeout after model setup",
    )
    parser.add_argument(
        "--retriever",
        choices=["tfidf", "embedding"],
        default="tfidf",
        help="First-stage retriever; embedding requires the optional nlp extra",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model for --retriever embedding",
    )
    parser.add_argument(
        "--embedding-model-revision",
        default=None,
        help="Immutable embedding model revision, preferably a commit SHA",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Optional CrossEncoder model for bounded candidate reranking",
    )
    parser.add_argument(
        "--reranker-model-revision",
        default=None,
        help="Immutable reranker model revision, preferably a commit SHA",
    )
    parser.add_argument("--id-column", default="question_id")
    parser.add_argument("--text-column", default="question")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON instead of stdout")
    args = parser.parse_args(argv)

    if args.embedding_model_revision is not None and args.retriever != "embedding":
        print("--embedding-model-revision requires --retriever=embedding", file=sys.stderr)
        return 1
    if args.reranker_model_revision is not None and args.reranker_model is None:
        print("--reranker-model-revision requires --reranker-model", file=sys.stderr)
        return 1
    if not args.catalog_csv.is_file():
        print(f"File not found: {args.catalog_csv}", file=sys.stderr)
        return 1
    if len(args.query) > args.max_queries:
        print(
            f"query count exceeds max-queries ({args.max_queries})",
            file=sys.stderr,
        )
        return 1
    if args.k > args.max_k:
        print(f"k exceeds max-k ({args.max_k})", file=sys.stderr)
        return 1
    effective_candidate_k = max(args.k, args.candidate_k)
    if effective_candidate_k > args.max_candidate_k:
        print(
            f"candidate-k exceeds max-candidate-k ({args.max_candidate_k})",
            file=sys.stderr,
        )
        return 1
    queries_to_run: list[str] = []
    for query in args.query:
        normalized_query = query.strip()
        if not normalized_query:
            print("query must not be empty", file=sys.stderr)
            return 1
        if len(normalized_query) > args.max_query_chars:
            print(
                f"query exceeds max-query-chars ({args.max_query_chars})",
                file=sys.stderr,
            )
            return 1
        queries_to_run.append(normalized_query)
    try:
        frame = pd.read_csv(args.catalog_csv)
        retriever: CatalogRetriever
        if args.retriever == "embedding":
            retriever = SentenceTransformerCatalogRetriever(
                args.embedding_model,
                revision=args.embedding_model_revision,
            ).fit_frame(
                frame,
                id_col=args.id_column,
                text_col=args.text_column,
            )
        else:
            retriever = TfidfCatalogRetriever().fit_frame(
                frame,
                id_col=args.id_column,
                text_col=args.text_column,
            )
        reranker = (
            SentenceTransformerCrossEncoderReranker(
                args.reranker_model,
                revision=args.reranker_model_revision,
            )
            if args.reranker_model
            else None
        )
        deadline = (
            time.perf_counter() + args.timeout_seconds
            if args.timeout_seconds is not None
            else None
        )
        queries = []
        for query in queries_to_run:
            _check_deadline(deadline)
            hits = (
                search_and_rerank(
                    retriever,
                    query,
                    k=args.k,
                    candidate_k=effective_candidate_k,
                    score_batch=reranker.score_batch,
                    deadline=deadline,
                )
                if reranker is not None
                else retriever.search(query, k=args.k)
            )
            _check_deadline(deadline)
            queries.append(
                {
                    "query": query,
                    "k": args.k,
                    "candidate_k": effective_candidate_k,
                    "hits": [hit.as_dict() for hit in hits],
                }
            )
    except (OSError, KeyError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Unable to retrieve from catalog: {exc}", file=sys.stderr)
        return 1

    _write_payload(
        {
            "catalog_size": retriever.size,
            "retriever": args.retriever,
            "embedding_model": args.embedding_model if args.retriever == "embedding" else None,
            "embedding_model_revision": (
                getattr(retriever, "model_revision", None)
                if args.retriever == "embedding"
                else None
            ),
            "reranker": args.reranker_model,
            "reranker_model_revision": (
                getattr(reranker, "model_revision", None) if reranker is not None else None
            ),
            "policy": {
                "max_queries": args.max_queries,
                "max_k": args.max_k,
                "max_candidate_k": args.max_candidate_k,
                "max_query_chars": args.max_query_chars,
                "timeout_seconds": args.timeout_seconds,
                "timeout_behavior": "cooperative_between_queries_and_stages",
            },
            "queries": queries,
        },
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
