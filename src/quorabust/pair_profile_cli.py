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

import pandas as pd

from quorabust.lineage import git_revision
from quorabust.pair_profile import benchmark_pair_classifier, normalize_pair_frame
from quorabust.persist import load_classifier
from quorabust.retrieval_benchmark import source_manifest, summarize_latencies_ms


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


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _threshold(value: str) -> float:
    parsed = _positive_float(value)
    if parsed >= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
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
    path_flags = {"--model", "--eval-csv", "--dependency-lock", "--out"}
    hidden_flags = {"--_child-output"}
    rendered: list[str] = []
    redact_next = False
    skip_next = False
    for raw_value in arguments:
        value = str(raw_value)
        if skip_next:
            skip_next = False
            continue
        if value in hidden_flags:
            skip_next = True
            continue
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
    return shlex.join(["quorabust-pair-profile", *rendered])


def _model_identity(meta: dict[str, Any], artifact: Path) -> dict[str, Any]:
    allowed = (
        "artifact_format",
        "feature_backend",
        "quorabust_version",
        "git_revision",
        "embedding_model",
        "embedding_model_revision",
        "cross_encoder_model",
        "cross_encoder_model_revision",
        "cross_encoder_batch_size",
    )
    identity = {key: meta[key] for key in allowed if key in meta}
    identity.setdefault(
        "artifact_format",
        "quorabust.safe.tfidf_xgboost" if artifact.suffix == ".qmodel" else "pickle_trusted_local",
    )
    identity["artifact_name"] = artifact.name
    return identity


def _resolve_threshold(requested: float | None, meta: dict[str, Any]) -> tuple[float, str]:
    if requested is not None:
        return requested, "cli"
    candidate = meta.get("decision_threshold")
    if (
        isinstance(candidate, int | float)
        and not isinstance(candidate, bool)
        and 0.0 < float(candidate) < 1.0
    ):
        return float(candidate), "artifact_metadata"
    return 0.5, "default"


def _load_frame(path: Path, label_column: str, max_rows: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    return normalize_pair_frame(frame, label_column=label_column)


def _child_command(args: argparse.Namespace, child_out: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "quorabust.pair_profile_cli",
        "--model",
        str(args.model),
        "--eval-csv",
        str(args.eval_csv),
        "--batch-size",
        str(args.batch_size),
        "--warmup-runs",
        str(args.warmup_runs),
        "--repetitions",
        str(args.repetitions),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--label-column",
        args.label_column,
        "--_child-output",
        str(child_out),
    ]
    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    if args.threshold is not None:
        command.extend(["--threshold", str(args.threshold)])
    if args.dependency_lock is not None:
        command.extend(["--dependency-lock", str(args.dependency_lock)])
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile saved Quorabust pair-classifier artifacts.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--warmup-runs", type=_non_negative_int, default=1)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--cold-start-repetitions", type=_positive_int, default=3)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--max-rows", type=_positive_int, default=None)
    parser.add_argument("--label-column", default="is_duplicate")
    parser.add_argument(
        "--dependency-lock",
        type=Path,
        default=None,
        help="Optional dependency lock file to hash into the report",
    )
    parser.add_argument(
        "--threshold",
        type=_threshold,
        default=None,
        help="Optional quality threshold; otherwise use artifact metadata or 0.5",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--_child-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = [args.model, args.eval_csv]
    if args.dependency_lock is not None:
        paths.append(args.dependency_lock)
    for path in paths:
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

    if args._child_output is not None:
        try:
            frame = _load_frame(args.eval_csv, args.label_column, args.max_rows)
            builder, classifier, meta = load_classifier(args.model)
            threshold, threshold_source = _resolve_threshold(args.threshold, meta)
            warm_benchmark = benchmark_pair_classifier(
                builder,
                classifier,
                frame,
                batch_size=args.batch_size,
                warmup_runs=args.warmup_runs,
                repetitions=args.repetitions,
                timeout_seconds=args.timeout_seconds,
                label_column=args.label_column,
                threshold=threshold,
            )
        except (
            OSError,
            EOFError,
            KeyError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"Unable to profile pair classifier: {exc}", file=sys.stderr)
            return 1

        quality = warm_benchmark.get("quality")
        if isinstance(quality, dict):
            quality["threshold_source"] = threshold_source
        child_payload = {
            "model": _model_identity(meta, args.model),
            "warm_benchmark": warm_benchmark,
        }
        _write_payload(child_payload, args._child_output)
        return 0

    elapsed_samples: list[float] = []
    first_payload: dict[str, Any] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="quorabust-pair-profile-") as temporary:
            for index in range(args.cold_start_repetitions):
                child_out = Path(temporary) / f"profile-{index}.json"
                child_payload, elapsed_ms = _run_child(args, child_out)
                if first_payload is None:
                    first_payload = child_payload
                elapsed_samples.append(elapsed_ms)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Unable to profile pair classifier: {exc}", file=sys.stderr)
        return 1

    if first_payload is None:
        print("Unable to profile pair classifier: no child measurements", file=sys.stderr)
        return 1

    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "quorabust-pair-profile",
        "evidence_scope": "pair_classifier_timing_and_optional_quality",
        "sources": {"evaluation": source_manifest(str(args.eval_csv))},
        "artifacts": [source_manifest(str(args.model))],
        "dependency_lock": (
            source_manifest(str(args.dependency_lock))
            if args.dependency_lock is not None
            else None
        ),
        "model": first_payload["model"],
        "configuration": {
            "batch_size": args.batch_size,
            "warmup_runs": args.warmup_runs,
            "repetitions": args.repetitions,
            "cold_start_repetitions": args.cold_start_repetitions,
            "timeout_seconds": args.timeout_seconds,
            "max_rows": args.max_rows,
            "label_column": args.label_column,
            "threshold": args.threshold,
        },
        "warm_benchmark": first_payload["warm_benchmark"],
        "cold_start": {
            "measurement_count": len(elapsed_samples),
            "process_to_report_ms": summarize_latencies_ms(elapsed_samples),
            "isolation": "fresh_subprocess_per_measurement",
            "timeout_seconds": args.timeout_seconds,
        },
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
