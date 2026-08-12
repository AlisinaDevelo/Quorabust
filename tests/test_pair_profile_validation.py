import copy
import json

from quorabust.pair_profile_validation import main, validate_pair_profile_payload


def _latency(count: int, value: float) -> dict[str, float | int]:
    return {
        "count": count,
        "mean": value,
        "p50": value,
        "p95": value,
        "p99": value,
        "max": value,
    }


def _manifest(name: str, size: int, letter: str) -> dict[str, str | int]:
    return {"name": name, "sha256": letter * 64, "bytes": size}


def _payload() -> dict:
    strata = {
        "short": {
            "pair_count": 2,
            "measured_pair_count": 4,
            "measurement_count": 2,
            "token_count": {"min": 1, "max": 5},
            "latency_ms": {"batch": _latency(2, 2.0), "per_pair": _latency(2, 1.0)},
            "work": {"throughput_pairs_per_second": 100.0},
        },
        "medium": {
            "pair_count": 2,
            "measured_pair_count": 4,
            "measurement_count": 2,
            "token_count": {"min": 6, "max": 15},
            "latency_ms": {"batch": _latency(2, 2.0), "per_pair": _latency(2, 1.0)},
            "work": {"throughput_pairs_per_second": 100.0},
        },
    }
    return {
        "schema_version": 1,
        "benchmark": "quorabust-pair-profile",
        "evidence_scope": "pair_classifier_timing_and_optional_quality",
        "sources": {"evaluation": _manifest("evaluation.csv", 100, "a")},
        "artifacts": [_manifest("model.qmodel", 25, "b")],
        "dependency_lock": _manifest("requirements.txt", 50, "c"),
        "model": {"artifact_format": "quorabust.safe.tfidf_xgboost"},
        "configuration": {"batch_size": 2},
        "warm_benchmark": {
            "pair_count": 4,
            "measured_pair_count": 8,
            "measurement_count": 4,
            "batch_size": 2,
            "latency_ms": {"batch": _latency(4, 2.0), "per_pair": _latency(4, 1.0)},
            "pair_length_policy": {
                "tokenization": "python_str_split_whitespace",
                "measure": "maximum_whitespace_token_count_of_question1_and_question2",
                "buckets": {
                    "short": {"min_tokens": 1, "max_tokens": 5},
                    "medium": {"min_tokens": 6, "max_tokens": 15},
                    "long": {"min_tokens": 16, "max_tokens": None},
                },
            },
            "pair_length_strata": strata,
            "labels": {"present": True, "column": "is_duplicate", "counts": {"0": 2, "1": 2}},
            "quality": {
                "rows": 4,
                "label_counts": {"0": 2, "1": 2},
                "threshold": 0.5,
                "threshold_source": "artifact_metadata",
                "accuracy": 0.75,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "log_loss": 0.3,
                "roc_auc": 0.9,
                "pr_auc": 0.9,
            },
            "work": {"throughput_pairs_per_second": 50.0, "concurrency": 1, "execution": "serial"},
            "measurement_policy": {
                "warmup_runs": 1,
                "repetitions": 2,
                "timeout_seconds": 30.0,
                "concurrency": 1,
                "execution": "serial",
                "timeout_behavior": "cooperative_deadline_between_batches",
            },
            "runtime": {
                "python_version": "3.12.0",
                "system": "Linux",
                "machine": "x86_64",
                "profile_git_revision": "abc123",
                "peak_rss_bytes": 4096,
                "rss_measurement": "process_maxrss_since_process_start",
            },
        },
        "cold_start": {
            "measurement_count": 2,
            "process_to_report_ms": _latency(2, 4.0),
            "isolation": "fresh_subprocess_per_measurement",
            "timeout_seconds": 30.0,
        },
        "runtime": {
            "python_version": "3.12.0",
            "system": "Linux",
            "machine": "x86_64",
            "profile_git_revision": "abc123",
        },
        "command": "quorabust-pair-profile --model <model.qmodel>",
        "generated_at_utc": "2026-08-10T00:00:00Z",
    }


def test_validate_pair_profile_accepts_quality_and_cost_policies():
    assert (
        validate_pair_profile_payload(
            _payload(),
            max_cold_start_p95_ms=4.0,
            max_warm_batch_p95_ms=2.0,
            max_warm_per_pair_p95_ms=1.0,
            max_peak_rss_bytes=4096,
            max_total_artifact_bytes=25,
            min_quality_f1=0.8,
            min_quality_roc_auc=0.9,
            min_quality_pr_auc=0.9,
            min_throughput_pairs_per_second=50.0,
        )
        == []
    )


def test_validate_pair_profile_reports_policy_failures():
    errors = validate_pair_profile_payload(
        _payload(),
        max_cold_start_p95_ms=3.0,
        max_warm_batch_p95_ms=1.0,
        max_peak_rss_bytes=4095,
        max_total_artifact_bytes=24,
        min_quality_f1=0.9,
        min_quality_pr_auc=0.95,
        min_throughput_pairs_per_second=60.0,
    )

    assert any("cold_start.process_to_report_ms.p95" in error for error in errors)
    assert any("warm_benchmark.latency_ms.batch.p95" in error for error in errors)
    assert any("warm_benchmark.runtime.peak_rss_bytes" in error for error in errors)
    assert any("total artifact bytes" in error for error in errors)
    assert any("quality.f1" in error for error in errors)
    assert any("quality.pr_auc" in error for error in errors)
    assert any("throughput_pairs_per_second" in error for error in errors)


def test_validate_pair_profile_cli_fails_quality_policy_without_labels(tmp_path, capsys):
    payload = _payload()
    payload["warm_benchmark"]["labels"]["present"] = False
    payload["warm_benchmark"]["labels"]["counts"] = None
    payload["warm_benchmark"]["quality"] = None
    report = tmp_path / "pair-profile.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--report", str(report)]) == 0
    assert "validated" in capsys.readouterr().out
    assert main(["--report", str(report), "--min-quality-f1", "0.5"]) == 1
    assert "quality policy requires labeled" in capsys.readouterr().err

    assert main(["--report", str(report), "--min-quality-pr-auc", "0.5"]) == 1
    assert "quality policy requires labeled" in capsys.readouterr().err

    broken = copy.deepcopy(payload)
    del broken["warm_benchmark"]["measurement_policy"]
    report.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["--report", str(report)]) == 1
