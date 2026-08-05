import json

import pandas as pd

from quorabust.retrieval_profile_cli import main


def _write_inputs(tmp_path):
    catalog = tmp_path / "catalog.csv"
    qrels = tmp_path / "qrels.csv"
    artifact = tmp_path / "local-artifact.bin"
    pd.DataFrame(
        {
            "question_id": ["q1", "q2", "q3"],
            "question": [
                "How do I learn Python?",
                "Where can I buy train tickets?",
                "How should I cache API responses?",
            ],
        }
    ).to_csv(catalog, index=False)
    pd.DataFrame(
        {
            "query": ["best way to learn Python", "where to buy train tickets"],
            "question_id": ["q1", "q2"],
            "relevance": [2, 1],
        }
    ).to_csv(qrels, index=False)
    artifact.write_bytes(b"trusted-local-artifact")
    return catalog, qrels, artifact


def test_profile_cli_reports_fresh_processes_sizes_and_path_light_output(tmp_path):
    catalog, qrels, artifact = _write_inputs(tmp_path)
    output = tmp_path / "profile.json"

    assert (
        main(
            [
                "--catalog-csv",
                str(catalog),
                "--qrels-csv",
                str(qrels),
                "--ks",
                "1,2",
                "--candidate-k",
                "2",
                "--warmup-runs",
                "0",
                "--repetitions",
                "1",
                "--cold-start-repetitions",
                "2",
                "--timeout-seconds",
                "10",
                "--artifact",
                str(artifact),
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "quorabust-retrieve-profile"
    assert payload["evidence_scope"] == "timing_and_size_only_no_quality_claim"
    assert payload["cold_start"]["measurement_count"] == 2
    assert payload["cold_start"]["process_to_report_ms"]["p99"] > 0.0
    assert payload["warm_benchmark"]["query_count"] == 2
    assert payload["sources"]["catalog"]["bytes"] == catalog.stat().st_size
    assert payload["artifacts"][0]["bytes"] == artifact.stat().st_size
    assert payload["artifacts"][0]["name"] == artifact.name
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_profile_cli_rejects_missing_artifact(tmp_path, capsys):
    catalog, qrels, _ = _write_inputs(tmp_path)

    assert (
        main(
            [
                "--catalog-csv",
                str(catalog),
                "--qrels-csv",
                str(qrels),
                "--artifact",
                str(tmp_path / "missing.bin"),
            ]
        )
        == 1
    )
    assert "File not found" in capsys.readouterr().err
