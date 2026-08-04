from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.lineage import sha256_file

REQUIRED_COLUMNS = ("question1", "question2", "is_duplicate")
OPTIONAL_QUESTION_ID_COLUMNS = ("qid1", "qid2")


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].fillna("").astype(str).str.strip()


def _label_summary(df: pd.DataFrame) -> dict[str, Any]:
    labels = df["is_duplicate"]
    valid = labels.isin([0, 1])
    counts = labels[valid].value_counts().to_dict()
    positive_count = int(counts.get(1, 0))
    row_count = len(labels)
    return {
        "counts": {"0": int(counts.get(0, 0)), "1": positive_count},
        "positive_rate": float(positive_count / row_count) if row_count else 0.0,
        "invalid_count": int((~valid).sum()),
    }


def _pair_summary(df: pd.DataFrame) -> dict[str, int]:
    q1 = _text_series(df, "question1")
    q2 = _text_series(df, "question2")
    canonical_pairs = pd.DataFrame(
        {
            "left": q1.where(q1 <= q2, q2),
            "right": q2.where(q1 <= q2, q1),
        }
    )
    duplicate_mask = canonical_pairs.duplicated(keep=False)
    return {
        "empty_question1": int((q1 == "").sum()),
        "empty_question2": int((q2 == "").sum()),
        "identical_question_rows": int((q1 == q2).sum()),
        "duplicate_pair_rows": int(duplicate_mask.sum()),
    }


def _question_id_summary(df: pd.DataFrame) -> dict[str, Any]:
    present = all(column in df.columns for column in OPTIONAL_QUESTION_ID_COLUMNS)
    if not present:
        return {
            "present": False,
            "complete_rows": 0,
            "unique_count": 0,
            "repeated_id_count": 0,
            "component_count": 0,
        }

    ids = df[list(OPTIONAL_QUESTION_ID_COLUMNS)]
    complete = ids.notna().all(axis=1)
    values = pd.Series(ids.to_numpy().reshape(-1)).dropna().astype(str)
    value_counts = values.value_counts()

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in ids.loc[complete].itertuples(index=False, name=None):
        left_value, right_value = str(left), str(right)
        left_root, right_root = find(left_value), find(right_value)
        if left_root != right_root:
            parent[right_root] = left_root

    components = {find(value) for value in values}
    return {
        "present": bool(complete.all()),
        "complete_rows": int(complete.sum()),
        "unique_count": int(value_counts.size),
        "repeated_id_count": int((value_counts > 1).sum()),
        "component_count": int(len(components)),
    }


def _check(name: str, status: str, message: str, **observed: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "observed": observed}


def audit_dataframe(
    df: pd.DataFrame,
    *,
    source_name: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a path-light quality and leakage preflight manifest for a pair dataset."""
    columns = [str(column).strip() for column in df.columns]
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(columns))
    missing_optional_columns = sorted(set(OPTIONAL_QUESTION_ID_COLUMNS) - set(columns))
    source: dict[str, Any] = {
        "name": source_name or "",
        "rows": int(len(df)),
        "columns": int(len(columns)),
    }
    if source_sha256 is not None:
        source["sha256"] = source_sha256

    if "is_duplicate" in df.columns:
        labels = _label_summary(df)
    else:
        labels = {"counts": {"0": 0, "1": 0}, "positive_rate": 0.0, "invalid_count": 0}
    if {"question1", "question2"}.issubset(df.columns):
        pairs = _pair_summary(df)
    else:
        pairs = {
            "empty_question1": 0,
            "empty_question2": 0,
            "identical_question_rows": 0,
            "duplicate_pair_rows": 0,
        }

    if set(OPTIONAL_QUESTION_ID_COLUMNS).issubset(df.columns):
        question_ids = _question_id_summary(df)
    else:
        question_ids = {
            "present": False,
            "complete_rows": 0,
            "unique_count": 0,
            "repeated_id_count": 0,
            "component_count": 0,
        }

    checks = [
        _check(
            "required_columns",
            "pass" if not missing_columns else "fail",
            "required pair and label columns are present"
            if not missing_columns
            else "required columns are missing",
            missing=missing_columns,
        ),
        _check(
            "non_empty_dataset",
            "pass" if len(df) else "fail",
            "dataset contains at least one row" if len(df) else "dataset is empty",
            rows=int(len(df)),
        ),
        _check(
            "binary_labels",
            "pass" if labels["invalid_count"] == 0 else "fail",
            "labels are restricted to 0 and 1"
            if labels["invalid_count"] == 0
            else "labels contain values other than 0 and 1",
            invalid_count=labels["invalid_count"],
        ),
        _check(
            "question_text",
            "warn"
            if pairs["empty_question1"] or pairs["empty_question2"]
            else "pass",
            "all question fields contain text"
            if not pairs["empty_question1"] and not pairs["empty_question2"]
            else "some question fields are empty",
            empty_question1=pairs["empty_question1"],
            empty_question2=pairs["empty_question2"],
        ),
        _check(
            "duplicate_pairs",
            "warn" if pairs["duplicate_pair_rows"] else "pass",
            "no repeated unordered pairs detected"
            if not pairs["duplicate_pair_rows"]
            else "repeated unordered pairs may overweight examples",
            duplicate_pair_rows=pairs["duplicate_pair_rows"],
        ),
        _check(
            "question_ids",
            "pass" if question_ids["present"] else "warn",
            "complete question IDs support leakage-aware splitting"
            if question_ids["present"]
            else "qid1 and qid2 are absent or incomplete; row-level split may leak questions",
            missing=missing_optional_columns,
        ),
    ]
    status = "fail" if any(item["status"] == "fail" for item in checks) else "pass"
    return {
        "schema_version": 1,
        "status": status,
        "source": source,
        "missing_columns": missing_columns,
        "missing_optional_columns": missing_optional_columns,
        "labels": labels,
        "pairs": pairs,
        "question_ids": question_ids,
        "checks": checks,
    }


def audit_csv(path: str | Path) -> dict[str, Any]:
    """Read a CSV and return its dataset audit manifest."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip() for column in df.columns]
    return audit_dataframe(
        df,
        source_name=csv_path.name,
        source_sha256=sha256_file(csv_path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a Quorabust pair CSV before training or evaluation.",
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input question-pair CSV")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON audit manifest")
    args = parser.parse_args(argv)

    if not args.csv.is_file():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 1
    try:
        audit = audit_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"Unable to audit {args.csv}: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"status={audit['status']} rows={audit['source']['rows']} "
        f"duplicate_pair_rows={audit['pairs']['duplicate_pair_rows']} "
        f"wrote {args.out.resolve()}"
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
