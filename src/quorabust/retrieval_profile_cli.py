from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quorabust.lineage import git_revision
from quorabust.retrieval_benchmark import (
    model_cache_manifest,
    source_manifest,
    summarize_latencies_ms,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int_list(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        raw = part.strip()
        if raw:
            values.append(_positive_int(raw))
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one positive integer")
    return sorted(set(values))


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _write_payload(payload: dict[str, Any], out: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out is None:
        print(rendered, end="")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")


def _path_light_command(argv: list[str] | None) -> str:
    arguments = sys.argv[1:] if argv is None else argv
    path_flags = {
        "--artifact",
        "--catalog-csv",
        "--model-cache",
        "--out",
        "--qrels-csv",
    }
    rendered: list[str] = []
    redact_next = False
    for raw_value in arguments:
        value = str(raw_value)
        if redact_next:
            rendered.append(f"<{Path(value).name}>")
            redact_next = False
            continue
        if value in path_flags:
            rendered.append(value)
            redact_next = True
            continue
        for flag in path_flags:
            prefix = f"{flag}="
            if value.startswith(prefix):
                rendered.append(f"{prefix}<{Path(value[len(prefix):]).name}>")
                break
        else:
            rendered.append(value)
    return shlex.join(["quorabust-retrieve-profile", *rendered])


def _child_command(args: argparse.Namespace, child_out: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "quorabust.retrieval_benchmark_cli",
        "--catalog-csv",
        str(args.catalog_csv),
        "--qrels-csv",
        str(args.qrels_csv),
        "--ks",
        ",".join(str(value) for value in args.ks),
        "--candidate-k",
        str(args.candidate_k),
        "--warmup-runs",
        str(args.warmup_runs),
        "--repetitions",
        str(args.repetitions),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--retriever",
        args.retriever,
        "--id-column",
        args.id_column,
        "--text-column",
        args.text_column,
        "--out",
        str(child_out),
    ]
    if args.retriever == "embedding":
        command.extend(["--embedding-model", args.embedding_model])
        if args.embedding_model_revision is not None:
            command.extend(["--embedding-model-revision", args.embedding_model_revision])
    if args.reranker_model:
        command.extend(["--reranker-model", args.reranker_model])
        if args.reranker_model_revision is not None:
            command.extend(["--reranker-model-revision", args.reranker_model_revision])
    return command


def _run_child(args: argparse.Namespace, child_out: Path) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            _child_command(args, child_out),
            capture_output=True,
            check=False,
            text=True,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"cold-start child exceeded {args.timeout_seconds:g} seconds"
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no child output"
        raise RuntimeError(f"cold-start child failed: {detail[-1000:]}")
    try:
        raw_payload = json.loads(child_out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cold-start child did not write valid JSON") from exc
    if not isinstance(raw_payload, dict):
        raise RuntimeError("cold-start child JSON must contain an object")
    return dict(raw_payload), elapsed_ms


def _warm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "catalog_size",
        "query_count",
        "measured_query_count",
        "candidate_k",
        "first_stage",
        "final",
        "latency_ms",
        "query_length_policy",
        "query_length_strata",
        "work",
        "measurement_policy",
        "runtime",
    )
    return {key: payload[key] for key in keys if key in payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile retrieval warm measurements in fresh subprocesses.",
    )
    parser.add_argument("--catalog-csv", type=Path, required=True)
    parser.add_argument("--qrels-csv", type=Path, required=True)
    parser.add_argument("--ks", type=_positive_int_list, default=[1, 5, 10])
    parser.add_argument("--candidate-k", type=_positive_int, default=50)
    parser.add_argument("--retriever", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-model-revision", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-model-revision", default=None)
    parser.add_argument("--id-column", default="question_id")
    parser.add_argument("--text-column", default="question")
    parser.add_argument("--warmup-runs", type=_non_negative_int, default=1)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--cold-start-repetitions", type=_positive_int, default=3)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="Optional local artifact file to hash and size; repeatable",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        action="append",
        default=[],
        help="Optional model file or cache directory to hash and size; repeatable",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.embedding_model_revision is not None and args.retriever != "embedding":
        print("--embedding-model-revision requires --retriever=embedding", file=sys.stderr)
        return 1
    if args.reranker_model_revision is not None and args.reranker_model is None:
        print("--reranker-model-revision requires --reranker-model", file=sys.stderr)
        return 1
    for path in [args.catalog_csv, args.qrels_csv, *args.artifact]:
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
    for path in args.model_cache:
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

    elapsed_samples: list[float] = []
    first_payload: dict[str, Any] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="quorabust-profile-") as temporary:
            for index in range(args.cold_start_repetitions):
                child_out = Path(temporary) / f"benchmark-{index}.json"
                child_payload, elapsed_ms = _run_child(args, child_out)
                if first_payload is None:
                    first_payload = child_payload
                elapsed_samples.append(elapsed_ms)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Unable to profile retrieval: {exc}", file=sys.stderr)
        return 1

    if first_payload is None:
        print("Unable to profile retrieval: no child measurements", file=sys.stderr)
        return 1

    try:
        model_caches = [model_cache_manifest(str(path)) for path in args.model_cache]
    except (OSError, ValueError) as exc:
        print(f"Unable to profile retrieval: {exc}", file=sys.stderr)
        return 1

    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "quorabust-retrieve-profile",
        "evidence_scope": "timing_and_size_only_no_quality_claim",
        "sources": {
            "catalog": source_manifest(str(args.catalog_csv)),
            "qrels": source_manifest(str(args.qrels_csv)),
        },
        "artifacts": [source_manifest(str(path)) for path in args.artifact],
        "model_caches": model_caches,
        "configuration": {
            "retriever": args.retriever,
            "embedding_model": args.embedding_model
            if args.retriever == "embedding"
            else None,
            "embedding_model_revision": args.embedding_model_revision
            if args.retriever == "embedding"
            else None,
            "reranker_model": args.reranker_model,
            "reranker_model_revision": args.reranker_model_revision
            if args.reranker_model
            else None,
            "ks": args.ks,
            "candidate_k": args.candidate_k,
            "warmup_runs": args.warmup_runs,
            "repetitions": args.repetitions,
            "cold_start_repetitions": args.cold_start_repetitions,
            "timeout_seconds": args.timeout_seconds,
            "model_cache_count": len(args.model_cache),
        },
        "cold_start": {
            "measurement_count": len(elapsed_samples),
            "process_to_report_ms": summarize_latencies_ms(elapsed_samples),
            "isolation": "fresh_subprocess_per_measurement",
            "timeout_seconds": args.timeout_seconds,
        },
        "warm_benchmark": _warm_payload(first_payload),
        "runtime": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "profile_git_revision": git_revision(),
        },
        "command": _path_light_command(argv),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_payload(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
