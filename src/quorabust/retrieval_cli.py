from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.retrieval import TfidfCatalogRetriever


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


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
    parser.add_argument("--id-column", default="question_id")
    parser.add_argument("--text-column", default="question")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON instead of stdout")
    args = parser.parse_args(argv)

    if not args.catalog_csv.is_file():
        print(f"File not found: {args.catalog_csv}", file=sys.stderr)
        return 1
    try:
        frame = pd.read_csv(args.catalog_csv)
        retriever = TfidfCatalogRetriever().fit_frame(
            frame,
            id_col=args.id_column,
            text_col=args.text_column,
        )
        queries = [
            {
                "query": query,
                "k": args.k,
                "hits": [hit.as_dict() for hit in retriever.search(query, k=args.k)],
            }
            for query in args.query
        ]
    except (OSError, KeyError, ValueError) as exc:
        print(f"Unable to retrieve from catalog: {exc}", file=sys.stderr)
        return 1

    _write_payload(
        {
            "catalog_size": retriever.size,
            "retriever": "tfidf",
            "queries": queries,
        },
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
