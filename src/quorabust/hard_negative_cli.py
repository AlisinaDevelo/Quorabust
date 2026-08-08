from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.hard_negatives import mine_hard_negatives
from quorabust.lineage import git_revision, sha256_file


def _command(argv: list[str] | None) -> str:
    arguments = sys.argv[1:] if argv is None else argv
    return shlex.join(["quorabust-mine-hard-negatives", *(str(value) for value in arguments)])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine leakage-safe lexical hard negatives from a labeled pair CSV.",
    )
    parser.add_argument("--csv", type=Path, required=True, help="Training/tuning pair CSV")
    parser.add_argument("--out", type=Path, required=True, help="Generated negative-pair CSV")
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Provenance sidecar (default: <out stem>.meta.json)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Number of lexical candidates considered per anchor",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=1,
        help="Maximum generated negatives per source positive row",
    )
    parser.add_argument(
        "--max-positive-rows",
        type=int,
        default=None,
        help="Optional deterministic sample of positive rows to mine",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    metadata_out = args.metadata_out or args.out.with_suffix(".meta.json")
    paths = {args.csv.resolve(), args.out.resolve(), metadata_out.resolve()}
    if len(paths) != 3:
        print("--csv, --out, and --metadata-out must be different paths", file=sys.stderr)
        return 1
    if not args.csv.is_file():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 1
    existing = [path for path in (args.out, metadata_out) if path.exists()]
    if existing:
        print(
            "refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 1

    try:
        frame = pd.read_csv(args.csv, dtype={"qid1": "string", "qid2": "string"})
        frame.columns = [str(column).strip() for column in frame.columns]
        result = mine_hard_negatives(
            frame,
            candidate_k=args.candidate_k,
            negatives_per_positive=args.negatives_per_positive,
            max_positive_rows=args.max_positive_rows,
            seed=args.seed,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.pairs.to_csv(args.out, index=False, lineterminator="\n")
        output_sha256 = sha256_file(args.out)
        payload = {
            **result.metadata,
            "source": {
                "reference": f"external://{args.csv.name}",
                "sha256": sha256_file(args.csv),
                "rows": int(len(frame)),
            },
            "output": {
                **result.metadata["output"],
                "reference": f"generated://{args.out.name}",
                "sha256": output_sha256,
            },
            "provenance": {
                "git_revision": git_revision(),
                "python_version": platform.python_version(),
                "machine": f"{platform.system()}/{platform.machine()}",
                "command": _command(argv),
            },
        }
        _write_json(metadata_out, payload)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Unable to mine hard negatives: {exc}", file=sys.stderr)
        if args.out.is_file():
            args.out.unlink()
        if metadata_out.is_file():
            metadata_out.unlink()
        return 1

    print(f"generated {len(result.pairs)} negatives")
    print(f"wrote {args.out.resolve()}")
    print(f"wrote {metadata_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
