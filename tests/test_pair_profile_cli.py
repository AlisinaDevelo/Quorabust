import json

import pandas as pd

from quorabust.model import train_duplicate_classifier
from quorabust.pair_profile_cli import main
from quorabust.persist import save_classifier


def _write_inputs(tmp_path):
    train = pd.DataFrame(
        {
            "question1": [
                "Python?",
                "Where can I buy train tickets?",
                "How do I learn Java?",
                "Where can I buy plane tickets?",
            ],
            "question2": [
                "Python?",
                "Where can I purchase train tickets?",
                "What is the best way to learn Java?",
                "Where can I purchase plane tickets?",
            ],
            "is_duplicate": [1, 1, 1, 1],
        }
    )
    # Keep both classes in the training fixture while retaining readable examples.
    train.loc[1, "is_duplicate"] = 0
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)

    evaluation = pd.DataFrame(
        {
            "question1": [
                "Python?",
                "Where can I buy train tickets today?",
                "How should I cache an API response for production systems?",
                "What is the best way to learn Python?",
            ],
            "question2": [
                "Python?",
                "Which site sells train tickets?",
                "How should I cache API responses?",
                "Which color is the moon?",
            ],
            "is_duplicate": [1, 0, 1, 0],
        }
    )
    evaluation_path = tmp_path / "evaluation.csv"
    evaluation.to_csv(evaluation_path, index=False)

    builder, classifier = train_duplicate_classifier(
        train,
        xgb_params={
            "n_estimators": 3,
            "max_depth": 2,
            "n_jobs": 1,
        },
    )
    model_path = tmp_path / "model.pkl"
    save_classifier(model_path, builder, classifier, meta={"feature_backend": "tfidf"})
    return model_path, evaluation_path


def test_pair_profile_cli_reports_cost_provenance_and_length_strata(tmp_path):
    model_path, evaluation_path = _write_inputs(tmp_path)
    output = tmp_path / "pair-profile.json"

    assert (
        main(
            [
                "--model",
                str(model_path),
                "--eval-csv",
                str(evaluation_path),
                "--batch-size",
                "2",
                "--warmup-runs",
                "0",
                "--repetitions",
                "2",
                "--cold-start-repetitions",
                "2",
                "--timeout-seconds",
                "30",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    warm = payload["warm_benchmark"]
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "quorabust-pair-profile"
    assert payload["evidence_scope"] == "pair_classifier_timing_and_optional_quality"
    assert payload["artifacts"][0]["name"] == model_path.name
    assert payload["sources"]["evaluation"]["name"] == evaluation_path.name
    assert payload["cold_start"]["measurement_count"] == 2
    assert payload["cold_start"]["process_to_report_ms"]["p99"] > 0.0
    assert warm["pair_count"] == 4
    assert warm["measured_pair_count"] == 8
    assert warm["measurement_count"] == 6
    assert warm["latency_ms"]["batch"]["count"] == 6
    assert warm["work"]["throughput_pairs_per_second"] > 0.0
    assert set(warm["pair_length_strata"]) == {"short", "medium"}
    assert warm["labels"]["counts"] == {"0": 2, "1": 2}
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_pair_profile_cli_rejects_non_binary_labels(tmp_path, capsys):
    model_path, evaluation_path = _write_inputs(tmp_path)
    frame = pd.read_csv(evaluation_path)
    frame.loc[0, "is_duplicate"] = 2
    frame.to_csv(evaluation_path, index=False)

    assert main(["--model", str(model_path), "--eval-csv", str(evaluation_path)]) == 1
    assert "labels must contain only 0 and 1" in capsys.readouterr().err
