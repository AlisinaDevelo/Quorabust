from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_BENCHMARK = "quorabust-pair-profile"
_EVIDENCE_SCOPE = "pair_classifier_timing_and_optional_quality"
_LATENCY_FIELDS = ("count", "mean", "p50", "p95", "p99", "max")
_BUCKETS = {
    "short": {"min_tokens": 1, "max_tokens": 5},
    "medium": {"min_tokens": 6, "max_tokens": 15},
    "long": {"min_tokens": 16, "max_tokens": None},
}
_PAIR_LENGTH_MEASURE = "maximum_whitespace_token_count_of_question1_and_question2"


def _missing_keys(value: Any, keys: set[str]) -> list[str]:
    if not isinstance(value, dict):
        return sorted(keys)
    return sorted(keys - set(value))


def _number(value: Any, field: str, errors: list[str], *, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{field} must be a number")
        return
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        errors.append(f"{field} must be finite and at least {minimum:g}")


def _positive_integer(value: Any, field: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        errors.append(f"{field} must be a positive integer")


def _non_negative_integer(value: Any, field: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{field} must be a non-negative integer")


def _validate_manifest(value: Any, field: str, errors: list[str]) -> None:
    for key in _missing_keys(value, {"name", "sha256", "bytes"}):
        errors.append(f"missing {field} field: {key}")
    if not isinstance(value, dict):
        return
    if not isinstance(value.get("name"), str) or not value["name"].strip():
        errors.append(f"{field}.name must be a non-empty string")
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        errors.append(f"{field}.sha256 must be a 64-character hex digest")
    _non_negative_integer(value.get("bytes"), f"{field}.bytes", errors)


def _validate_latency(
    value: Any,
    field: str,
    errors: list[str],
    expected_count: int | None,
) -> None:
    for key in _missing_keys(value, set(_LATENCY_FIELDS)):
        errors.append(f"missing {field} field: {key}")
    if not isinstance(value, dict):
        return
    count = value.get("count")
    _positive_integer(count, f"{field}.count", errors)
    if expected_count is not None and count != expected_count:
        errors.append(f"{field}.count must equal expected measurement count ({expected_count})")
    _number(value.get("mean"), f"{field}.mean", errors)
    previous = 0.0
    for key in ("p50", "p95", "p99", "max"):
        raw_value = value.get(key)
        _number(raw_value, f"{field}.{key}", errors)
        if isinstance(raw_value, int | float) and not isinstance(raw_value, bool):
            numeric = float(raw_value)
            if math.isfinite(numeric) and numeric < previous:
                errors.append(f"{field}.{key} must not be below the previous percentile")
            if math.isfinite(numeric):
                previous = numeric


def _validate_runtime(value: Any, field: str, errors: list[str]) -> None:
    for key in ("python_version", "system", "machine", "profile_git_revision"):
        if not isinstance(value, dict) or not isinstance(value.get(key), str) or not value[key]:
            errors.append(f"{field}.{key} must be a non-empty string")


def _validate_strata(
    warm: dict[str, Any],
    *,
    pair_count: int | None,
    measured_pair_count: int | None,
    repetitions: int | None,
    errors: list[str],
) -> None:
    policy = warm.get("pair_length_policy")
    expected_policy = {
        "tokenization": "python_str_split_whitespace",
        "measure": _PAIR_LENGTH_MEASURE,
        "buckets": _BUCKETS,
    }
    if policy != expected_policy:
        errors.append("pair_length_policy does not match the documented policy")

    strata = warm.get("pair_length_strata")
    if not isinstance(strata, dict) or not strata:
        errors.append("pair_length_strata must be a non-empty object")
        return

    total_pairs = 0
    total_measured = 0
    total_measurements = 0
    for bucket, value in strata.items():
        if bucket not in _BUCKETS:
            errors.append(f"pair_length_strata has unknown bucket: {bucket}")
            continue
        if not isinstance(value, dict):
            errors.append(f"pair_length_strata.{bucket} must be an object")
            continue
        bucket_pairs = value.get("pair_count")
        bucket_measured = value.get("measured_pair_count")
        bucket_measurements = value.get("measurement_count")
        _positive_integer(bucket_pairs, f"pair_length_strata.{bucket}.pair_count", errors)
        _positive_integer(
            bucket_measured,
            f"pair_length_strata.{bucket}.measured_pair_count",
            errors,
        )
        _positive_integer(
            bucket_measurements,
            f"pair_length_strata.{bucket}.measurement_count",
            errors,
        )
        if isinstance(bucket_pairs, int) and not isinstance(bucket_pairs, bool):
            total_pairs += bucket_pairs
        if isinstance(bucket_measured, int) and not isinstance(bucket_measured, bool):
            total_measured += bucket_measured
            if isinstance(bucket_pairs, int) and isinstance(repetitions, int):
                if bucket_measured != bucket_pairs * repetitions:
                    errors.append(
                        f"pair_length_strata.{bucket}.measured_pair_count must equal "
                        "pair_count multiplied by repetitions"
                    )
        if isinstance(bucket_measurements, int) and not isinstance(bucket_measurements, bool):
            total_measurements += bucket_measurements

        token_count = value.get("token_count")
        for key in _missing_keys(token_count, {"min", "max"}):
            errors.append(f"missing pair_length_strata.{bucket}.token_count field: {key}")
        if isinstance(token_count, dict):
            _positive_integer(
                token_count.get("min"),
                f"pair_length_strata.{bucket}.token_count.min",
                errors,
            )
            _positive_integer(
                token_count.get("max"),
                f"pair_length_strata.{bucket}.token_count.max",
                errors,
            )

        latency = value.get("latency_ms")
        _validate_latency(
            latency.get("batch") if isinstance(latency, dict) else None,
            f"pair_length_strata.{bucket}.latency_ms.batch",
            errors,
            bucket_measurements if isinstance(bucket_measurements, int) else None,
        )
        _validate_latency(
            latency.get("per_pair") if isinstance(latency, dict) else None,
            f"pair_length_strata.{bucket}.latency_ms.per_pair",
            errors,
            bucket_measurements if isinstance(bucket_measurements, int) else None,
        )
        work = value.get("work")
        if not isinstance(work, dict):
            errors.append(f"pair_length_strata.{bucket}.work must be an object")
        else:
            _number(
                work.get("throughput_pairs_per_second"),
                f"pair_length_strata.{bucket}.work.throughput_pairs_per_second",
                errors,
            )

    if isinstance(pair_count, int) and total_pairs != pair_count:
        errors.append("pair_length_strata pair_count total must equal pair_count")
    if isinstance(measured_pair_count, int) and total_measured != measured_pair_count:
        errors.append("pair_length_strata measured_pair_count total must equal measured_pair_count")
    measurement_count = warm.get("measurement_count")
    if isinstance(measurement_count, int) and total_measurements != measurement_count:
        errors.append("pair_length_strata measurement_count total must equal measurement_count")


def _validate_quality(warm: dict[str, Any], pair_count: int | None, errors: list[str]) -> None:
    labels = warm.get("labels")
    if not isinstance(labels, dict) or not isinstance(labels.get("present"), bool):
        errors.append("labels.present must be a boolean")
        return
    counts = labels.get("counts")
    if labels["present"]:
        if not isinstance(counts, dict) or set(counts) != {"0", "1"}:
            errors.append("labels.counts must contain 0 and 1 when labels are present")
        else:
            for key in ("0", "1"):
                _non_negative_integer(counts.get(key), f"labels.counts.{key}", errors)
            if (
                isinstance(pair_count, int)
                and all(isinstance(counts.get(key), int) for key in ("0", "1"))
                and sum(counts.values()) != pair_count
            ):
                errors.append("labels.counts total must equal pair_count")
        quality = warm.get("quality")
        if not isinstance(quality, dict):
            errors.append("quality must be an object when labels are present")
            return
        if quality.get("rows") != pair_count:
            errors.append("quality.rows must equal pair_count")
        for field in ("threshold", "accuracy", "precision", "recall", "f1", "log_loss"):
            _number(quality.get(field), f"quality.{field}", errors)
        for field in ("accuracy", "precision", "recall", "f1"):
            value = quality.get(field)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and float(value) > 1.0
            ):
                errors.append(f"quality.{field} must be at most 1")
        if "roc_auc" in quality:
            _number(quality.get("roc_auc"), "quality.roc_auc", errors)
            if isinstance(quality.get("roc_auc"), int | float) and float(quality["roc_auc"]) > 1.0:
                errors.append("quality.roc_auc must be at most 1")
    elif counts is not None or warm.get("quality") is not None:
        errors.append("quality and label counts must be null when labels are absent")


def _validate_outer(payload: Any, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        errors.append("report must be a JSON object")
        return None
    if payload.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if payload.get("benchmark") != _BENCHMARK:
        errors.append(f"benchmark must be {_BENCHMARK!r}")
    if payload.get("evidence_scope") != _EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must equal {_EVIDENCE_SCOPE!r}")
    for key in (
        "sources",
        "artifacts",
        "model",
        "configuration",
        "warm_benchmark",
        "cold_start",
        "runtime",
        "command",
        "generated_at_utc",
    ):
        if key not in payload:
            errors.append(f"missing top-level field: {key}")
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        errors.append("sources must be an object")
    else:
        _validate_manifest(sources.get("evaluation"), "sources.evaluation", errors)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a non-empty array")
    else:
        for index, artifact in enumerate(artifacts):
            _validate_manifest(artifact, f"artifacts.{index}", errors)
    if payload.get("dependency_lock") is not None:
        _validate_manifest(payload.get("dependency_lock"), "dependency_lock", errors)
    for key in ("command", "generated_at_utc"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            errors.append(f"{key} must be a non-empty string")
    _validate_runtime(payload.get("runtime"), "runtime", errors)
    return (
        payload.get("warm_benchmark") if isinstance(payload.get("warm_benchmark"), dict) else None
    )


def _validate_warm(warm: dict[str, Any], errors: list[str]) -> None:
    required = {
        "pair_count",
        "measured_pair_count",
        "measurement_count",
        "batch_size",
        "latency_ms",
        "pair_length_policy",
        "pair_length_strata",
        "labels",
        "quality",
        "work",
        "measurement_policy",
        "runtime",
    }
    for key in _missing_keys(warm, required):
        errors.append(f"missing warm_benchmark field: {key}")
    pair_count = warm.get("pair_count")
    measured_pair_count = warm.get("measured_pair_count")
    measurement_count = warm.get("measurement_count")
    _positive_integer(pair_count, "warm_benchmark.pair_count", errors)
    _positive_integer(measured_pair_count, "warm_benchmark.measured_pair_count", errors)
    _positive_integer(measurement_count, "warm_benchmark.measurement_count", errors)
    _positive_integer(warm.get("batch_size"), "warm_benchmark.batch_size", errors)

    policy = warm.get("measurement_policy")
    for key in {
        "warmup_runs",
        "repetitions",
        "timeout_seconds",
        "concurrency",
        "execution",
        "timeout_behavior",
    }:
        if not isinstance(policy, dict) or key not in policy:
            errors.append(f"missing warm_benchmark.measurement_policy field: {key}")
    repetitions: int | None = None
    if isinstance(policy, dict):
        _non_negative_integer(
            policy.get("warmup_runs"), "warm_benchmark.measurement_policy.warmup_runs", errors
        )
        repetitions = policy.get("repetitions")
        _positive_integer(repetitions, "warm_benchmark.measurement_policy.repetitions", errors)
        _number(
            policy.get("timeout_seconds"),
            "warm_benchmark.measurement_policy.timeout_seconds",
            errors,
            minimum=0.0,
        )
        if policy.get("concurrency") != 1 or policy.get("execution") != "serial":
            errors.append("warm_benchmark.measurement_policy must describe serial concurrency of 1")
        if not isinstance(policy.get("timeout_behavior"), str):
            errors.append("warm_benchmark.measurement_policy.timeout_behavior must be a string")
    if (
        isinstance(pair_count, int)
        and isinstance(repetitions, int)
        and measured_pair_count != pair_count * repetitions
    ):
        errors.append(
            "warm_benchmark.measured_pair_count must equal pair_count multiplied by repetitions"
        )

    latency = warm.get("latency_ms")
    _validate_latency(
        latency.get("batch") if isinstance(latency, dict) else None,
        "warm_benchmark.latency_ms.batch",
        errors,
        measurement_count if isinstance(measurement_count, int) else None,
    )
    _validate_latency(
        latency.get("per_pair") if isinstance(latency, dict) else None,
        "warm_benchmark.latency_ms.per_pair",
        errors,
        measurement_count if isinstance(measurement_count, int) else None,
    )
    _validate_runtime(warm.get("runtime"), "warm_benchmark.runtime", errors)
    runtime = warm.get("runtime")
    if isinstance(runtime, dict) and runtime.get("peak_rss_bytes") is not None:
        _non_negative_integer(
            runtime.get("peak_rss_bytes"), "warm_benchmark.runtime.peak_rss_bytes", errors
        )

    work = warm.get("work")
    if not isinstance(work, dict):
        errors.append("warm_benchmark.work must be an object")
    else:
        _number(
            work.get("throughput_pairs_per_second"),
            "warm_benchmark.work.throughput_pairs_per_second",
            errors,
        )
        if work.get("concurrency") != 1 or work.get("execution") != "serial":
            errors.append("warm_benchmark.work must describe serial concurrency of 1")
    _validate_quality(warm, pair_count if isinstance(pair_count, int) else None, errors)
    _validate_strata(
        warm,
        pair_count=pair_count if isinstance(pair_count, int) else None,
        measured_pair_count=measured_pair_count if isinstance(measured_pair_count, int) else None,
        repetitions=repetitions,
        errors=errors,
    )


def _policy_number(
    value: Any,
    field: str,
    maximum: float | None,
    minimum: float | None,
    errors: list[str],
) -> None:
    if maximum is not None and (not isinstance(value, int | float) or isinstance(value, bool)):
        errors.append(f"{field} must be numeric for the policy")
    elif maximum is not None and float(value) > maximum:
        errors.append(f"{field}={float(value):g} exceeds policy maximum {maximum:g}")
    if minimum is not None and (not isinstance(value, int | float) or isinstance(value, bool)):
        errors.append(f"{field} must be numeric for the policy")
    elif minimum is not None and float(value) < minimum:
        errors.append(f"{field}={float(value):g} is below policy minimum {minimum:g}")


def validate_pair_profile_payload(
    payload: Any,
    *,
    max_cold_start_p95_ms: float | None = None,
    max_warm_batch_p95_ms: float | None = None,
    max_warm_per_pair_p95_ms: float | None = None,
    max_peak_rss_bytes: int | None = None,
    max_total_artifact_bytes: int | None = None,
    min_quality_f1: float | None = None,
    min_quality_roc_auc: float | None = None,
    min_throughput_pairs_per_second: float | None = None,
) -> list[str]:
    """Return structural and caller-owned policy errors for a pair profile."""
    errors: list[str] = []
    warm = _validate_outer(payload, errors)
    if warm is None:
        return errors
    _validate_warm(warm, errors)

    cold_start = payload.get("cold_start")
    for key in {"measurement_count", "process_to_report_ms", "isolation", "timeout_seconds"}:
        if not isinstance(cold_start, dict) or key not in cold_start:
            errors.append(f"missing cold_start field: {key}")
    if isinstance(cold_start, dict):
        count = cold_start.get("measurement_count")
        _positive_integer(count, "cold_start.measurement_count", errors)
        _validate_latency(
            cold_start.get("process_to_report_ms"),
            "cold_start.process_to_report_ms",
            errors,
            count if isinstance(count, int) else None,
        )
        if cold_start.get("isolation") != "fresh_subprocess_per_measurement":
            errors.append("cold_start.isolation must describe fresh subprocess measurements")
        _number(
            cold_start.get("timeout_seconds"), "cold_start.timeout_seconds", errors, minimum=0.0
        )

    warm_latency = warm.get("latency_ms")
    batch_latency = warm_latency.get("batch") if isinstance(warm_latency, dict) else None
    per_pair_latency = warm_latency.get("per_pair") if isinstance(warm_latency, dict) else None
    cold_latency = cold_start.get("process_to_report_ms") if isinstance(cold_start, dict) else None
    _policy_number(
        cold_latency.get("p95") if isinstance(cold_latency, dict) else None,
        "cold_start.process_to_report_ms.p95",
        max_cold_start_p95_ms,
        None,
        errors,
    )
    _policy_number(
        batch_latency.get("p95") if isinstance(batch_latency, dict) else None,
        "warm_benchmark.latency_ms.batch.p95",
        max_warm_batch_p95_ms,
        None,
        errors,
    )
    _policy_number(
        per_pair_latency.get("p95") if isinstance(per_pair_latency, dict) else None,
        "warm_benchmark.latency_ms.per_pair.p95",
        max_warm_per_pair_p95_ms,
        None,
        errors,
    )

    runtime = warm.get("runtime")
    observed_rss = runtime.get("peak_rss_bytes") if isinstance(runtime, dict) else None
    if max_peak_rss_bytes is not None:
        if not isinstance(observed_rss, int) or isinstance(observed_rss, bool):
            errors.append(
                "warm_benchmark.runtime.peak_rss_bytes must be an integer for the RSS policy"
            )
        elif observed_rss > max_peak_rss_bytes:
            errors.append(
                f"warm_benchmark.runtime.peak_rss_bytes={observed_rss} exceeds policy maximum "
                f"{max_peak_rss_bytes}"
            )

    if max_total_artifact_bytes is not None:
        artifacts = payload.get("artifacts")
        sizes: list[int] = []
        artifacts_are_valid = isinstance(artifacts, list) and bool(artifacts)
        if artifacts_are_valid:
            assert isinstance(artifacts, list)
            for artifact in artifacts:
                size = artifact.get("bytes") if isinstance(artifact, dict) else None
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    artifacts_are_valid = False
                    break
                sizes.append(size)
        if not artifacts_are_valid:
            errors.append("artifact-size policy requires valid declared artifacts")
        elif sum(sizes) > max_total_artifact_bytes:
            errors.append(
                f"total artifact bytes={sum(sizes)} exceeds policy maximum "
                f"{max_total_artifact_bytes}"
            )

    quality = warm.get("quality")
    for field, minimum in (
        ("f1", min_quality_f1),
        ("roc_auc", min_quality_roc_auc),
    ):
        if minimum is None:
            continue
        if not isinstance(quality, dict):
            errors.append("quality policy requires labeled quality metrics")
        else:
            _policy_number(
                quality.get(field), f"warm_benchmark.quality.{field}", None, minimum, errors
            )

    work = warm.get("work")
    if min_throughput_pairs_per_second is not None:
        throughput = work.get("throughput_pairs_per_second") if isinstance(work, dict) else None
        _policy_number(
            throughput,
            "warm_benchmark.work.throughput_pairs_per_second",
            None,
            min_throughput_pairs_per_second,
            errors,
        )
    return errors


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _fraction(value: str) -> float:
    parsed = _non_negative_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate pair-classifier cost evidence and optional release policies.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-cold-start-p95-ms", type=_non_negative_float, default=None)
    parser.add_argument("--max-warm-batch-p95-ms", type=_non_negative_float, default=None)
    parser.add_argument("--max-warm-per-pair-p95-ms", type=_non_negative_float, default=None)
    parser.add_argument("--max-peak-rss-bytes", type=_non_negative_int, default=None)
    parser.add_argument("--max-total-artifact-bytes", type=_non_negative_int, default=None)
    parser.add_argument("--min-quality-f1", type=_fraction, default=None)
    parser.add_argument("--min-quality-roc-auc", type=_fraction, default=None)
    parser.add_argument(
        "--min-throughput-pairs-per-second",
        type=_non_negative_float,
        default=None,
    )
    args = parser.parse_args(argv)

    if not args.report.is_file():
        print(f"File not found: {args.report}", file=sys.stderr)
        return 1
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid pair profile report: {exc}", file=sys.stderr)
        return 1

    errors = validate_pair_profile_payload(
        payload,
        max_cold_start_p95_ms=args.max_cold_start_p95_ms,
        max_warm_batch_p95_ms=args.max_warm_batch_p95_ms,
        max_warm_per_pair_p95_ms=args.max_warm_per_pair_p95_ms,
        max_peak_rss_bytes=args.max_peak_rss_bytes,
        max_total_artifact_bytes=args.max_total_artifact_bytes,
        min_quality_f1=args.min_quality_f1,
        min_quality_roc_auc=args.min_quality_roc_auc,
        min_throughput_pairs_per_second=args.min_throughput_pairs_per_second,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
