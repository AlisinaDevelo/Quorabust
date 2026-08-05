from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.retrieval import (
    CatalogRetriever,
    SentenceTransformerCatalogRetriever,
    SentenceTransformerCrossEncoderReranker,
    TfidfCatalogRetriever,
)
from quorabust.retrieval_benchmark import (
    benchmark_retrieval,
    load_retrieval_qrels,
    source_manifest,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_int_list(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        raw = part.strip()
        if not raw:
            continue
        values.append(_positive_int(raw))
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one positive integer")
    return sorted(set(values))


def _write_payload(payload: dict[str, Any], out: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out is None:
        print(rendered, end="")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")


def _catalog_ids(frame: pd.DataFrame, id_column: str) -> set[str]:
    if id_column not in frame.columns:
        raise KeyError(f"Missing catalog column: {id_column}")
    ids: list[str] = []
    for raw_value in frame[id_column].tolist():
        if pd.isna(raw_value):
            raise ValueError("catalog question IDs must not be empty")
        value = str(raw_value).strip()
        if not value:
            raise ValueError("catalog question IDs must not be empty")
        ids.append(value)
    if len(set(ids)) != len(ids):
        raise ValueError("catalog question IDs must be unique")
    return set(ids)


def _command(argv: list[str] | None) -> str:
    arguments = sys.argv[1:] if argv is None else argv
    return shlex.join(["quorabust-retrieve-benchmark", *(str(value) for value in arguments)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark catalog retrieval and optional bounded reranking.",
    )
    parser.add_argument("--catalog-csv", type=Path, required=True)
    parser.add_argument("--qrels-csv", type=Path, required=True)
    parser.add_argument(
        "--ks",
        type=_positive_int_list,
        default=[1, 5, 10],
        help="Comma-separated cutoffs for recall, MRR, and NDCG (default: 1,5,10)",
    )
    parser.add_argument(
        "--candidate-k",
        type=_positive_int,
        default=50,
        help="Bound the first-stage candidates passed to an optional reranker",
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
        "--reranker-model",
        default=None,
        help="Optional CrossEncoder model for bounded candidate reranking",
    )
    parser.add_argument("--id-column", default="question_id")
    parser.add_argument("--text-column", default="question")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON instead of stdout")
    args = parser.parse_args(argv)

    for path in (args.catalog_csv, args.qrels_csv):
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

    try:
        catalog_frame = pd.read_csv(args.catalog_csv)
        catalog_frame.columns = [str(column).strip() for column in catalog_frame.columns]
        catalog_ids = _catalog_ids(catalog_frame, args.id_column)

        retriever: CatalogRetriever
        if args.retriever == "embedding":
            retriever = SentenceTransformerCatalogRetriever(args.embedding_model).fit_frame(
                catalog_frame,
                id_col=args.id_column,
                text_col=args.text_column,
            )
        else:
            retriever = TfidfCatalogRetriever().fit_frame(
                catalog_frame,
                id_col=args.id_column,
                text_col=args.text_column,
            )
        cases = load_retrieval_qrels(str(args.qrels_csv), catalog_ids=catalog_ids)
        reranker = (
            SentenceTransformerCrossEncoderReranker(args.reranker_model)
            if args.reranker_model
            else None
        )
        payload = benchmark_retrieval(
            retriever,
            cases,
            ks=args.ks,
            candidate_k=args.candidate_k,
            score_batch=reranker.score_batch if reranker is not None else None,
        )
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Unable to benchmark retrieval: {exc}", file=sys.stderr)
        return 1

    payload.update(
        {
            "schema_version": 1,
            "benchmark": "quorabust-retrieve-benchmark",
            "sources": {
                "catalog": source_manifest(str(args.catalog_csv)),
                "qrels": source_manifest(str(args.qrels_csv)),
            },
            "configuration": {
                "retriever": args.retriever,
                "embedding_model": args.embedding_model
                if args.retriever == "embedding"
                else None,
                "reranker_model": args.reranker_model,
                "ks": args.ks,
            },
            "command": _command(argv),
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    _write_payload(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
