from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_BENCHMARK_NAME = "quorabust-retrieve-benchmark"
_PROFILE_NAME = "quorabust-retrieve-profile"
_EVIDENCE_SCOPE = "timing_and_size_only_no_quality_claim"
_STAGES = ("retrieval", "rerank", "end_to_end")
_METRICS = ("recall_at_k", "mrr_at_k", "ndcg_at_k")
_LATENCY_FIELDS = ("count", "mean", "p50", "p95", "p99", "max")
_BUCKETS = {
    "short": {"min_tokens": 1, "max_tokens": 5},
    "medium": {"min_tokens": 6, "max_tokens": 15},
    "long": {"min_tokens": 16, "max_tokens": None},
}


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


def _validate_file_manifest(manifest: Any, field: str, errors: list[str]) -> None:
    for key in _missing_keys(manifest, {"name", "sha256", "bytes"}):
        errors.append(f"missing {field} field: {key}")
    if not isinstance(manifest, dict):
        return
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        errors.append(f"{field}.name must be a non-empty string")
    digest = manifest.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        errors.append(f"{field}.sha256 must be a 64-character hex digest")
    _non_negative_integer(manifest.get("bytes"), f"{field}.bytes", errors)


def _validate_source_manifests(sources: Any, errors: list[str]) -> None:
    if not isinstance(sources, dict):
        errors.append("sources must be an object")
        return
    for name in ("catalog", "qrels"):
        _validate_file_manifest(sources.get(name), f"sources.{name}", errors)


def _validate_artifact_manifests(artifacts: Any, errors: list[str]) -> None:
    if not isinstance(artifacts, list):
        errors.append("artifacts must be an array")
        return
    for index, manifest in enumerate(artifacts):
        _validate_file_manifest(manifest, f"artifacts.{index}", errors)


def _validate_latency_summary(
    value: Any,
    field: str,
    errors: list[str],
    expected_count: int,
) -> None:
    for key in _missing_keys(value, set(_LATENCY_FIELDS)):
        errors.append(f"missing {field} field: {key}")
    if not isinstance(value, dict):
        return
    count = value.get("count")
    _positive_integer(count, f"{field}.count", errors)
    if count != expected_count:
        errors.append(f"{field}.count must equal measured_query_count ({expected_count})")
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


def _validate_metrics(value: Any, field: str, errors: list[str]) -> None:
    for metric in _METRICS:
        metric_values = value.get(metric) if isinstance(value, dict) else None
        if not isinstance(metric_values, dict) or not metric_values:
            errors.append(f"{field}.{metric} must be a non-empty object")
            continue
        for cutoff, raw_value in metric_values.items():
            _number(raw_value, f"{field}.{metric}.{cutoff}", errors)
            if isinstance(raw_value, int | float) and not isinstance(raw_value, bool):
                if float(raw_value) > 1.0:
                    errors.append(f"{field}.{metric}.{cutoff} must be at most 1")


def _validate_query_length_strata(
    policy: Any,
    strata: Any,
    *,
    query_count: int,
    measured_query_count: int,
    errors: list[str],
) -> None:
    if not isinstance(policy, dict):
        errors.append("query_length_policy must be an object")
    else:
        if policy.get("tokenization") != "python_str_split_whitespace":
            errors.append("query_length_policy.tokenization is unsupported")
        if policy.get("buckets") != _BUCKETS:
            errors.append("query_length_policy.buckets does not match the documented policy")

    if not isinstance(strata, dict) or not strata:
        errors.append("query_length_strata must be a non-empty object")
        return

    total_queries = 0
    total_measured = 0
    for bucket, value in strata.items():
        if bucket not in _BUCKETS:
            errors.append(f"query_length_strata has unknown bucket: {bucket}")
            continue
        if not isinstance(value, dict):
            errors.append(f"query_length_strata.{bucket} must be an object")
            continue
        query_bucket_count = value.get("query_count")
        measured_bucket_count = value.get("measured_query_count")
        _positive_integer(query_bucket_count, f"query_length_strata.{bucket}.query_count", errors)
        _positive_integer(
            measured_bucket_count,
            f"query_length_strata.{bucket}.measured_query_count",
            errors,
        )
        if isinstance(query_bucket_count, int) and not isinstance(query_bucket_count, bool):
            total_queries += query_bucket_count
        if isinstance(measured_bucket_count, int) and not isinstance(measured_bucket_count, bool):
            total_measured += measured_bucket_count

        token_count = value.get("token_count")
        for key in _missing_keys(token_count, {"min", "max"}):
            errors.append(f"missing query_length_strata.{bucket}.token_count field: {key}")
        if isinstance(token_count, dict):
            _positive_integer(
                token_count.get("min"),
                f"query_length_strata.{bucket}.token_count.min",
                errors,
            )
            _positive_integer(
                token_count.get("max"),
                f"query_length_strata.{bucket}.token_count.max",
                errors,
            )

        latency = value.get("latency_ms")
        if not isinstance(measured_bucket_count, int) or isinstance(measured_bucket_count, bool):
            continue
        for stage in _STAGES:
            _validate_latency_summary(
                latency.get(stage) if isinstance(latency, dict) else None,
                f"query_length_strata.{bucket}.latency_ms.{stage}",
                errors,
                measured_bucket_count,
            )

    if total_queries != query_count:
        errors.append(
            "query_length_strata query_count total must equal "
            f"query_count ({query_count})"
        )
    if total_measured != measured_query_count:
        errors.append(
            "query_length_strata measured_query_count total must equal "
            f"measured_query_count ({measured_query_count})"
        )


def _validate_core_payload(payload: Any, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        errors.append("benchmark payload must be an object")
        return None
    required = {
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
    }
    for key in _missing_keys(payload, required):
        errors.append(f"missing benchmark field: {key}")

    query_count = payload.get("query_count")
    measured_query_count = payload.get("measured_query_count")
    _positive_integer(query_count, "query_count", errors)
    _positive_integer(measured_query_count, "measured_query_count", errors)
    _positive_integer(payload.get("catalog_size"), "catalog_size", errors)
    _positive_integer(payload.get("candidate_k"), "candidate_k", errors)

    for stage in ("first_stage", "final"):
        _validate_metrics(payload.get(stage), stage, errors)

    latency = payload.get("latency_ms")
    for stage in _STAGES:
        _validate_latency_summary(
            latency.get(stage) if isinstance(latency, dict) else None,
            f"latency_ms.{stage}",
            errors,
            measured_query_count if isinstance(measured_query_count, int) else -1,
        )

    measurement_policy = payload.get("measurement_policy")
    for key in _missing_keys(
        measurement_policy,
        {
            "warmup_runs",
            "repetitions",
            "quality_passes",
            "latency_samples_per_stage",
            "timeout_seconds",
            "concurrency",
            "execution",
            "timeout_behavior",
        },
    ):
        errors.append(f"missing measurement_policy field: {key}")
    repetitions = None
    if isinstance(measurement_policy, dict):
        _non_negative_integer(
            measurement_policy.get("warmup_runs"),
            "measurement_policy.warmup_runs",
            errors,
        )
        repetitions = measurement_policy.get("repetitions")
        _positive_integer(repetitions, "measurement_policy.repetitions", errors)
        if measurement_policy.get("quality_passes") != 1:
            errors.append("measurement_policy.quality_passes must equal 1")
        if measurement_policy.get("latency_samples_per_stage") != measured_query_count:
            errors.append(
                "measurement_policy.latency_samples_per_stage must equal measured_query_count"
            )
        if (
            measurement_policy.get("concurrency") != 1
            or measurement_policy.get("execution") != "serial"
        ):
            errors.append("measurement_policy must describe serial concurrency of 1")
        if not isinstance(measurement_policy.get("timeout_behavior"), str):
            errors.append("measurement_policy.timeout_behavior must be a string")
    if (
        isinstance(query_count, int)
        and not isinstance(query_count, bool)
        and isinstance(repetitions, int)
        and not isinstance(repetitions, bool)
        and measured_query_count != query_count * repetitions
    ):
        errors.append("measured_query_count must equal query_count multiplied by repetitions")

    runtime = payload.get("runtime")
    for key in {"python_version", "system", "machine", "benchmark_git_revision"}:
        if (
            not isinstance(runtime, dict)
            or not isinstance(runtime.get(key), str)
            or not runtime[key]
        ):
            errors.append(f"runtime.{key} must be a non-empty string")

    work = payload.get("work")
    for key in {
        "reranker_enabled",
        "reranker_pairs",
        "mean_candidates",
        "throughput_queries_per_second",
    }:
        if not isinstance(work, dict) or key not in work:
            errors.append(f"missing work field: {key}")
    if isinstance(work, dict):
        if not isinstance(work.get("reranker_enabled"), bool):
            errors.append("work.reranker_enabled must be a boolean")
        _non_negative_integer(work.get("reranker_pairs"), "work.reranker_pairs", errors)
        _number(work.get("mean_candidates"), "work.mean_candidates", errors)
        throughput = work.get("throughput_queries_per_second")
        if throughput is not None:
            _number(throughput, "work.throughput_queries_per_second", errors)

    if isinstance(query_count, int) and not isinstance(query_count, bool):
        _validate_query_length_strata(
            payload.get("query_length_policy"),
            payload.get("query_length_strata"),
            query_count=query_count,
            measured_query_count=(
                measured_query_count if isinstance(measured_query_count, int) else -1
            ),
            errors=errors,
        )
    return payload


def _validate_outer_payload(payload: Any, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        errors.append("report must be a JSON object")
        return None
    if payload.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    benchmark = payload.get("benchmark")
    if benchmark not in {_BENCHMARK_NAME, _PROFILE_NAME}:
        errors.append(f"benchmark must be {_BENCHMARK_NAME!r} or {_PROFILE_NAME!r}")
        return None
    for key in ("sources", "configuration", "command", "generated_at_utc"):
        if key not in payload:
            errors.append(f"missing top-level field: {key}")
    _validate_source_manifests(payload.get("sources"), errors)
    if not isinstance(payload.get("configuration"), dict):
        errors.append("configuration must be an object")
    if not isinstance(payload.get("command"), str) or not payload["command"].strip():
        errors.append("command must be a non-empty string")
    if (
        not isinstance(payload.get("generated_at_utc"), str)
        or not payload["generated_at_utc"].strip()
    ):
        errors.append("generated_at_utc must be a non-empty string")

    if benchmark == _PROFILE_NAME:
        if payload.get("evidence_scope") != _EVIDENCE_SCOPE:
            errors.append(f"evidence_scope must equal {_EVIDENCE_SCOPE!r}")
        if "artifacts" not in payload:
            errors.append("missing top-level field: artifacts")
        _validate_artifact_manifests(payload.get("artifacts"), errors)
        profile_runtime = payload.get("runtime")
        for key in {"python_version", "system", "machine", "profile_git_revision"}:
            if (
                not isinstance(profile_runtime, dict)
                or not isinstance(profile_runtime.get(key), str)
                or not profile_runtime[key]
            ):
                errors.append(f"runtime.{key} must be a non-empty string")
        cold_start = payload.get("cold_start")
        for key in {"measurement_count", "process_to_report_ms", "isolation", "timeout_seconds"}:
            if not isinstance(cold_start, dict) or key not in cold_start:
                errors.append(f"missing cold_start field: {key}")
        if isinstance(cold_start, dict):
            measurement_count = cold_start.get("measurement_count")
            _positive_integer(measurement_count, "cold_start.measurement_count", errors)
            _validate_latency_summary(
                cold_start.get("process_to_report_ms"),
                "cold_start.process_to_report_ms",
                errors,
                measurement_count if isinstance(measurement_count, int) else -1,
            )
            if cold_start.get("isolation") != "fresh_subprocess_per_measurement":
                errors.append("cold_start.isolation must describe fresh subprocess measurements")
            _number(
                cold_start.get("timeout_seconds"),
                "cold_start.timeout_seconds",
                errors,
                minimum=0.0,
            )
        warm_payload = payload.get("warm_benchmark")
        if not isinstance(warm_payload, dict):
            errors.append("warm_benchmark must be an object")
            return None
        return warm_payload
    return payload


def validate_retrieval_payload(
    payload: Any,
    *,
    min_final_recall_at_k: Mapping[int, float] | None = None,
    max_end_to_end_p95_ms: float | None = None,
    max_cold_start_p95_ms: float | None = None,
    max_peak_rss_bytes: int | None = None,
    max_total_artifact_bytes: int | None = None,
) -> list[str]:
    """Return structural and caller-supplied policy errors for a retrieval report."""
    errors: list[str] = []
    core = _validate_outer_payload(payload, errors)
    if core is None:
        return errors
    _validate_core_payload(core, errors)

    final = core.get("final") if isinstance(core, dict) else None
    if min_final_recall_at_k:
        final_recall = final.get("recall_at_k") if isinstance(final, dict) else None
        for cutoff, minimum in sorted(min_final_recall_at_k.items()):
            raw_value = final_recall.get(str(cutoff)) if isinstance(final_recall, dict) else None
            if raw_value is None:
                errors.append(f"final.recall_at_k is missing cutoff {cutoff}")
            elif not isinstance(raw_value, int | float) or isinstance(raw_value, bool):
                errors.append(f"final.recall_at_k.{cutoff} must be numeric")
            elif float(raw_value) < minimum:
                errors.append(
                    f"final.recall_at_k.{cutoff}={float(raw_value):g} is below policy "
                    f"minimum {minimum:g}"
                )

    if max_end_to_end_p95_ms is not None:
        latency = core.get("latency_ms") if isinstance(core, dict) else None
        end_to_end = latency.get("end_to_end") if isinstance(latency, dict) else None
        observed = end_to_end.get("p95") if isinstance(end_to_end, dict) else None
        if not isinstance(observed, int | float) or isinstance(observed, bool):
            errors.append("latency_ms.end_to_end.p95 must be numeric for the latency policy")
        elif float(observed) > max_end_to_end_p95_ms:
            errors.append(
                f"latency_ms.end_to_end.p95={float(observed):g} exceeds policy maximum "
                f"{max_end_to_end_p95_ms:g}"
            )

    is_profile = isinstance(payload, dict) and payload.get("benchmark") == _PROFILE_NAME
    if max_cold_start_p95_ms is not None:
        if not is_profile:
            errors.append("cold-start policy requires a retrieval profile report")
        else:
            cold_start = payload.get("cold_start")
            latency = (
                cold_start.get("process_to_report_ms")
                if isinstance(cold_start, dict)
                else None
            )
            observed = latency.get("p95") if isinstance(latency, dict) else None
            if not isinstance(observed, int | float) or isinstance(observed, bool):
                errors.append(
                    "cold_start.process_to_report_ms.p95 must be numeric for the "
                    "cold-start policy"
                )
            elif float(observed) > max_cold_start_p95_ms:
                errors.append(
                    "cold_start.process_to_report_ms.p95="
                    f"{float(observed):g} exceeds policy maximum "
                    f"{max_cold_start_p95_ms:g}"
                )

    if max_peak_rss_bytes is not None:
        runtime = core.get("runtime") if isinstance(core, dict) else None
        observed = runtime.get("peak_rss_bytes") if isinstance(runtime, dict) else None
        if not isinstance(observed, int) or isinstance(observed, bool):
            errors.append("runtime.peak_rss_bytes must be an integer for the RSS policy")
        elif observed > max_peak_rss_bytes:
            errors.append(
                f"runtime.peak_rss_bytes={observed} exceeds policy maximum "
                f"{max_peak_rss_bytes}"
            )

    if max_total_artifact_bytes is not None:
        if not is_profile:
            errors.append("artifact-size policy requires a retrieval profile report")
        else:
            artifacts = payload.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                errors.append("artifact-size policy requires at least one declared artifact")
            else:
                sizes: list[int] = []
                for artifact in artifacts:
                    size = artifact.get("bytes") if isinstance(artifact, dict) else None
                    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                        break
                    sizes.append(size)
                if len(sizes) == len(artifacts):
                    total_bytes = sum(sizes)
                    if total_bytes > max_total_artifact_bytes:
                        errors.append(
                            f"total artifact bytes={total_bytes} exceeds policy maximum "
                            f"{max_total_artifact_bytes}"
                        )
    return errors


def _recall_policy(value: str) -> tuple[int, float]:
    raw_cutoff, separator, raw_minimum = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must use K=VALUE syntax")
    try:
        cutoff = int(raw_cutoff)
        minimum = float(raw_minimum)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("K and VALUE must be numeric") from exc
    if cutoff < 1 or not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise argparse.ArgumentTypeError("K must be positive and VALUE must be between 0 and 1")
    return cutoff, minimum


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
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
        description="Validate retrieval benchmark evidence and optional release policies.",
    )
    parser.add_argument("--report", type=Path, required=True, help="Benchmark or profile JSON")
    parser.add_argument(
        "--min-final-recall-at-k",
        type=_recall_policy,
        action="append",
        default=[],
        metavar="K=VALUE",
        help="Require final recall at K to be at least VALUE; repeatable",
    )
    parser.add_argument(
        "--max-end-to-end-p95-ms",
        type=_non_negative_float,
        default=None,
        help="Require aggregate end-to-end p95 latency to stay at or below VALUE milliseconds",
    )
    parser.add_argument(
        "--max-cold-start-p95-ms",
        type=_non_negative_float,
        default=None,
        help="Require profile process-to-report p95 to stay at or below VALUE milliseconds",
    )
    parser.add_argument(
        "--max-peak-rss-bytes",
        type=_non_negative_int,
        default=None,
        help="Require warm benchmark process peak RSS to stay at or below VALUE bytes",
    )
    parser.add_argument(
        "--max-total-artifact-bytes",
        type=_non_negative_int,
        default=None,
        help="Require total declared profile artifact size to stay at or below VALUE bytes",
    )
    args = parser.parse_args(argv)

    if not args.report.is_file():
        print(f"File not found: {args.report}", file=sys.stderr)
        return 1
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid retrieval report: {exc}", file=sys.stderr)
        return 1

    errors = validate_retrieval_payload(
        payload,
        min_final_recall_at_k=dict(args.min_final_recall_at_k),
        max_end_to_end_p95_ms=args.max_end_to_end_p95_ms,
        max_cold_start_p95_ms=args.max_cold_start_p95_ms,
        max_peak_rss_bytes=args.max_peak_rss_bytes,
        max_total_artifact_bytes=args.max_total_artifact_bytes,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
