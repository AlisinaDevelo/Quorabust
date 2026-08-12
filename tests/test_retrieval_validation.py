import copy
import json
from pathlib import Path

from quorabust.retrieval_benchmark import summarize_latencies_ms
from quorabust.retrieval_benchmark_cli import main as benchmark_main
from quorabust.retrieval_validation import main, validate_retrieval_payload


def _benchmark_report(tmp_path: Path) -> tuple[Path, dict]:
    report = tmp_path / "retrieval.json"
    root = Path(__file__).parents[1]
    assert (
        benchmark_main(
            [
                "--catalog-csv",
                str(root / "examples/retrieval_catalog.csv"),
                "--qrels-csv",
                str(root / "examples/retrieval_qrels.csv"),
                "--ks",
                "1,2",
                "--candidate-k",
                "2",
                "--warmup-runs",
                "0",
                "--repetitions",
                "2",
                "--timeout-seconds",
                "10",
                "--out",
                str(report),
            ]
        )
        == 0
    )
    return report, json.loads(report.read_text(encoding="utf-8"))


def _profile_report(payload: dict) -> dict:
    core_keys = (
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
    return {
        "schema_version": 1,
        "benchmark": "quorabust-retrieve-profile",
        "evidence_scope": "timing_and_size_only_no_quality_claim",
        "sources": payload["sources"],
        "artifacts": [
            {
                "name": "retriever.bin",
                "sha256": "c" * 64,
                "bytes": 25,
            }
        ],
        "model_caches": [
            {
                "name": "model-cache",
                "sha256": "d" * 64,
                "bytes": 30,
                "kind": "directory",
                "file_count": 2,
            }
        ],
        "configuration": payload["configuration"],
        "cold_start": {
            "measurement_count": 2,
            "process_to_report_ms": summarize_latencies_ms([1.0, 2.0]),
            "isolation": "fresh_subprocess_per_measurement",
            "timeout_seconds": 10.0,
        },
        "warm_benchmark": {key: payload[key] for key in core_keys},
        "runtime": {
            "python_version": "3.12.0",
            "system": "Darwin",
            "machine": "arm64",
            "profile_git_revision": "abc123",
        },
        "command": "quorabust-retrieve-profile --catalog-csv catalog.csv",
        "generated_at_utc": "2026-08-06T00:00:00Z",
    }


def test_validate_retrieval_payload_accepts_benchmark_and_profile(tmp_path):
    _, payload = _benchmark_report(tmp_path)
    profile = _profile_report(payload)
    profile["warm_benchmark"]["runtime"]["peak_rss_bytes"] = 4096

    assert validate_retrieval_payload(
        payload,
        min_final_recall_at_k={1: 1.0},
        max_end_to_end_p95_ms=1000.0,
        max_catalog_size=3,
        max_candidate_k=2,
    ) == []
    assert validate_retrieval_payload(
        profile,
        max_cold_start_p95_ms=2.0,
        max_peak_rss_bytes=4096,
        max_total_artifact_bytes=25,
        max_total_model_cache_bytes=30,
        max_catalog_size=3,
        max_candidate_k=2,
    ) == []


def test_validate_retrieval_payload_enforces_profile_resource_budgets(tmp_path):
    _, payload = _benchmark_report(tmp_path)
    profile = _profile_report(payload)
    profile["warm_benchmark"]["runtime"]["peak_rss_bytes"] = 4096

    errors = validate_retrieval_payload(
        profile,
        max_cold_start_p95_ms=1.0,
        max_peak_rss_bytes=4095,
        max_total_artifact_bytes=24,
        max_total_model_cache_bytes=29,
        max_catalog_size=2,
        max_candidate_k=1,
    )

    assert any("cold_start.process_to_report_ms.p95" in error for error in errors)
    assert any("runtime.peak_rss_bytes" in error for error in errors)
    assert any("total artifact bytes" in error for error in errors)
    assert any("total model cache bytes" in error for error in errors)
    assert any("catalog_size=3" in error for error in errors)
    assert any("candidate_k=2" in error for error in errors)


def test_validate_retrieval_payload_fails_closed_without_profile_evidence(tmp_path):
    _, payload = _benchmark_report(tmp_path)

    errors = validate_retrieval_payload(
        payload,
        max_cold_start_p95_ms=1000.0,
        max_total_artifact_bytes=1000,
    )

    assert "cold-start policy requires a retrieval profile report" in errors
    assert "artifact-size policy requires a retrieval profile report" in errors

    profile = _profile_report(payload)
    profile["artifacts"] = []
    errors = validate_retrieval_payload(profile, max_total_artifact_bytes=1000)
    assert "artifact-size policy requires at least one declared artifact" in errors


def test_validate_retrieval_payload_requires_profile_provenance(tmp_path):
    _, payload = _benchmark_report(tmp_path)
    profile = _profile_report(payload)
    del profile["runtime"]["profile_git_revision"]
    profile["artifacts"][0]["sha256"] = "not-a-digest"

    errors = validate_retrieval_payload(profile)

    assert "runtime.profile_git_revision must be a non-empty string" in errors
    assert "artifacts.0.sha256 must be a 64-character hex digest" in errors


def test_validate_retrieval_payload_rejects_malformed_model_cache_manifest(tmp_path):
    _, payload = _benchmark_report(tmp_path)
    profile = _profile_report(payload)
    profile["model_caches"][0]["kind"] = "archive"
    del profile["model_caches"][0]["file_count"]

    errors = validate_retrieval_payload(profile)

    assert "model_caches.0.kind must be file or directory" in errors
    assert "missing model_caches.0 field: file_count" in errors


def test_validate_retrieval_payload_requires_core_fields():
    payload = {
        "schema_version": 1,
        "benchmark": "quorabust-retrieve-benchmark",
        "sources": {
            "catalog": {"name": "catalog.csv", "sha256": "a" * 64, "bytes": 1},
            "qrels": {"name": "qrels.csv", "sha256": "b" * 64, "bytes": 1},
        },
        "configuration": {},
        "command": "benchmark",
        "generated_at_utc": "2026-08-06T00:00:00Z",
    }
    errors = validate_retrieval_payload(payload, max_end_to_end_p95_ms=0.0)
    assert "missing benchmark field: final" in errors


def test_validate_retrieval_payload_reports_invariants_and_missing_provenance(tmp_path):
    _, payload = _benchmark_report(tmp_path)
    broken = copy.deepcopy(payload)
    del broken["sources"]["catalog"]["sha256"]
    broken["measured_query_count"] = 3
    del broken["query_length_strata"]

    errors = validate_retrieval_payload(broken)

    assert "missing sources.catalog field: sha256" in errors
    assert "measured_query_count must equal query_count multiplied by repetitions" in errors
    assert "missing benchmark field: query_length_strata" in errors


def test_validate_retrieval_cli_passes_and_fails_policy(tmp_path, capsys):
    report, payload = _benchmark_report(tmp_path)

    assert (
        main(
            [
                "--report",
                str(report),
                "--min-final-recall-at-k",
                "1=1.0",
                "--max-end-to-end-p95-ms",
                "1000",
                "--max-catalog-size",
                "3",
                "--max-candidate-k",
                "2",
            ]
        )
        == 0
    )
    assert "validated" in capsys.readouterr().out

    assert main(["--report", str(report), "--max-end-to-end-p95-ms", "0"]) == 1
    assert "exceeds policy maximum" in capsys.readouterr().err

    profile = _profile_report(payload)
    profile["warm_benchmark"]["runtime"]["peak_rss_bytes"] = 4096
    profile_report = tmp_path / "retrieval-profile.json"
    profile_report.write_text(json.dumps(profile), encoding="utf-8")
    assert (
        main(
            [
                "--report",
                str(profile_report),
                "--max-cold-start-p95-ms",
                "2",
                "--max-peak-rss-bytes",
                "4096",
                "--max-total-artifact-bytes",
                "25",
                "--max-catalog-size",
                "3",
                "--max-candidate-k",
                "2",
            ]
        )
        == 0
    )
