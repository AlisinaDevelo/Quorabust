import json

import pandas as pd

from quorabust.data_audit import audit_dataframe, main


def test_audit_dataframe_reports_quality_and_question_id_signals():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "qid1": [10, 10, 20],
            "qid2": [11, 11, 21],
            "question1": ["How do I learn Python?", "How do I learn Python?", ""],
            "question2": ["Best way to learn Python?", "Best way to learn Python?", "Other"],
            "is_duplicate": [1, 1, 0],
        }
    )

    audit = audit_dataframe(
        df,
        source_name="train.csv",
        source_sha256="a" * 64,
        require_question_ids=True,
    )

    assert audit["status"] == "pass"
    assert audit["source"] == {
        "name": "train.csv",
        "sha256": "a" * 64,
        "rows": 3,
        "columns": 6,
    }
    assert audit["labels"] == {
        "counts": {"0": 1, "1": 2},
        "positive_rate": 2 / 3,
        "invalid_count": 0,
    }
    assert audit["pairs"]["empty_question1"] == 1
    assert audit["pairs"]["duplicate_pair_rows"] == 2
    assert audit["policy"] == {
        "require_question_ids": True,
        "require_question_text": False,
    }
    assert audit["question_ids"]["present"] is True
    assert audit["question_ids"]["unique_count"] == 4
    assert audit["checks"][0]["status"] == "pass"
    assert audit["checks"][-1]["observed"]["required"] is True


def test_audit_dataframe_can_fail_closed_when_question_ids_are_missing_or_blank():
    df = pd.DataFrame(
        {
            "question1": ["a", "b"],
            "question2": ["c", "d"],
            "is_duplicate": [0, 1],
            "qid1": ["q1", ""],
            "qid2": ["q2", "q4"],
        }
    )

    audit = audit_dataframe(df, require_question_ids=True)

    assert audit["status"] == "fail"
    question_id_check = next(check for check in audit["checks"] if check["name"] == "question_ids")
    assert question_id_check["status"] == "fail"
    assert question_id_check["observed"]["required"] is True
    assert "required for this benchmark protocol" in question_id_check["message"]


def test_audit_dataframe_can_fail_closed_when_question_text_is_missing_or_blank():
    df = pd.DataFrame(
        {
            "question1": ["a", "  "],
            "question2": ["b", None],
            "is_duplicate": [0, 1],
        }
    )

    audit = audit_dataframe(df, require_question_text=True)

    assert audit["status"] == "fail"
    question_text_check = next(
        check for check in audit["checks"] if check["name"] == "question_text"
    )
    assert question_text_check["status"] == "fail"
    assert question_text_check["observed"]["required"] is True
    assert "required for this benchmark protocol" in question_text_check["message"]


def test_audit_dataframe_fails_for_schema_and_label_contract_violations():
    df = pd.DataFrame(
        {
            "question1": ["a", "b"],
            "is_duplicate": [0, 2],
        }
    )

    audit = audit_dataframe(df)

    assert audit["status"] == "fail"
    assert audit["missing_columns"] == ["question2"]
    assert audit["missing_optional_columns"] == ["qid1", "qid2"]
    assert audit["question_ids"]["present"] is False
    assert audit["labels"]["invalid_count"] == 1
    assert {check["name"] for check in audit["checks"] if check["status"] == "fail"} == {
        "required_columns",
        "binary_labels",
    }


def test_audit_cli_writes_json_and_returns_nonzero_for_invalid_data(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    report_path = tmp_path / "audit.json"
    pd.DataFrame(
        {
            "question1": ["a"],
            "question2": ["b"],
            "is_duplicate": [3],
        }
    ).to_csv(csv_path, index=False)

    result = main(["--csv", str(csv_path), "--out", str(report_path)])

    assert result == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["source"]["sha256"]


def test_audit_cli_strict_question_ids_fails_without_complete_ids(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    report_path = tmp_path / "audit.json"
    pd.DataFrame(
        {
            "question1": ["a"],
            "question2": ["b"],
            "is_duplicate": [0],
        }
    ).to_csv(csv_path, index=False)

    result = main(
        [
            "--csv",
            str(csv_path),
            "--out",
            str(report_path),
            "--require-question-ids",
        ]
    )

    assert result == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    question_id_check = next(
        check for check in payload["checks"] if check["name"] == "question_ids"
    )
    assert question_id_check["status"] == "fail"


def test_audit_cli_strict_question_text_fails_for_null_text(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    report_path = tmp_path / "audit.json"
    pd.DataFrame(
        {
            "question1": ["a", "b"],
            "question2": ["c", None],
            "is_duplicate": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    result = main(
        [
            "--csv",
            str(csv_path),
            "--out",
            str(report_path),
            "--require-question-text",
        ]
    )

    assert result == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["policy"]["require_question_text"] is True
    question_text_check = next(
        check for check in payload["checks"] if check["name"] == "question_text"
    )
    assert question_text_check["status"] == "fail"
