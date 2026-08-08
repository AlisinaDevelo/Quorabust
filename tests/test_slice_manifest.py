import json

import pytest

from quorabust.lineage import sha256_file
from quorabust.slice_manifest import (
    load_slice_manifest,
    validate_slice_manifest_payload,
    validate_slice_provenance_payload,
)


def _manifest(eval_csv, *, rows=2, columns=None):
    return {
        "schema_version": 1,
        "source": {
            "reference": "synthetic://evaluation.csv",
            "sha256": sha256_file(eval_csv),
            "rows": rows,
        },
        "columns": columns
        or {
            "language": {
                "labeling_method": "synthetic fixture owner labels",
                "description": "Controlled CI-only labels.",
            }
        },
    }


def test_load_slice_manifest_binds_hash_rows_and_columns(tmp_path):
    eval_csv = tmp_path / "evaluation.csv"
    eval_csv.write_text("question1,question2,is_duplicate,language\na,b,0,en\n", encoding="utf-8")
    path = tmp_path / "slices.json"
    path.write_text(
        json.dumps(
            _manifest(
                eval_csv,
                rows=1,
                columns={
                    "language": {"labeling_method": "synthetic fixture owner labels"},
                    "domain": {"labeling_method": "synthetic fixture domain labels"},
                },
            )
        ),
        encoding="utf-8",
    )

    loaded = load_slice_manifest(
        path,
        eval_path=eval_csv,
        eval_rows=1,
        slice_columns=["domain", "language"],
    )

    assert list(loaded["columns"]) == ["domain", "language"]
    assert loaded["source"]["rows"] == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda payload: payload["source"].update({"sha256": "0" * 64}), "source.sha256"),
        (lambda payload: payload["source"].update({"rows": 3}), "source.rows"),
        (lambda payload: payload["columns"]["language"].pop("labeling_method"), "labeling_method"),
    ],
)
def test_load_slice_manifest_rejects_stale_or_incomplete_metadata(tmp_path, change, message):
    eval_csv = tmp_path / "evaluation.csv"
    eval_csv.write_text("question1,question2,is_duplicate,language\na,b,0,en\n", encoding="utf-8")
    payload = _manifest(eval_csv, rows=1)
    change(payload)
    path = tmp_path / "slices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_slice_manifest(path, eval_path=eval_csv, eval_rows=1, slice_columns=["language"])


def test_validate_slice_provenance_requires_complete_observed_counts():
    manifest = {
        "schema_version": 1,
        "source": {"reference": "synthetic://evaluation.csv", "sha256": "a" * 64, "rows": 3},
        "columns": {"language": {"labeling_method": "owner labels"}},
    }
    valid = {
        "manifest": manifest,
        "observed_row_counts": {"language": {"rows": 3, "labels": {"en": 2, "it": 1}}},
    }
    assert validate_slice_manifest_payload(manifest) == []
    assert validate_slice_provenance_payload(valid) == []

    invalid = {**valid, "observed_row_counts": {"language": {"rows": 3, "labels": {"en": 1}}}}
    assert "slice_provenance.observed_row_counts.language.labels must sum to rows" in (
        validate_slice_provenance_payload(invalid)
    )
