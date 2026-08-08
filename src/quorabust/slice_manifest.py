from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from quorabust.lineage import sha256_file

SLICE_MANIFEST_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_LABELING_METHOD_LENGTH = 200
_MAX_DESCRIPTION_LENGTH = 500


def _non_empty_string(value: Any, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        errors.append(f"{field} must be a non-empty single-line string")
        return None
    return value.strip()


def validate_slice_manifest_payload(payload: Any) -> list[str]:
    """Return fail-closed validation errors for a slice provenance sidecar."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["slice manifest must be a JSON object"]

    schema_version = payload.get("schema_version")
    if schema_version != SLICE_MANIFEST_SCHEMA_VERSION:
        errors.append(
            "slice manifest schema_version must be "
            f"{SLICE_MANIFEST_SCHEMA_VERSION}"
        )

    source = payload.get("source")
    if not isinstance(source, dict):
        errors.append("slice manifest source must be an object")
    else:
        _non_empty_string(source.get("reference"), "slice manifest source.reference", errors)
        source_sha256 = source.get("sha256")
        if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
            errors.append("slice manifest source.sha256 must be a 64-character hex digest")
        rows = source.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            errors.append("slice manifest source.rows must be a positive integer")

    columns = payload.get("columns")
    if not isinstance(columns, dict) or not columns:
        errors.append("slice manifest columns must be a non-empty object")
    else:
        for column, metadata in columns.items():
            if not isinstance(column, str) or not column.strip():
                errors.append("slice manifest column names must be non-empty strings")
                continue
            if not isinstance(metadata, dict):
                errors.append(f"slice manifest columns.{column} must be an object")
                continue
            method = _non_empty_string(
                metadata.get("labeling_method"),
                f"slice manifest columns.{column}.labeling_method",
                errors,
            )
            if method is not None and len(method) > _MAX_LABELING_METHOD_LENGTH:
                errors.append(
                    f"slice manifest columns.{column}.labeling_method exceeds "
                    f"{_MAX_LABELING_METHOD_LENGTH} characters"
                )
            description = metadata.get("description")
            if description is not None:
                normalized_description = _non_empty_string(
                    description,
                    f"slice manifest columns.{column}.description",
                    errors,
                )
                if (
                    normalized_description is not None
                    and len(normalized_description) > _MAX_DESCRIPTION_LENGTH
                ):
                    errors.append(
                        f"slice manifest columns.{column}.description exceeds "
                        f"{_MAX_DESCRIPTION_LENGTH} characters"
                    )
    return errors


def _canonical_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload["source"]
    columns = payload["columns"]
    canonical_columns: dict[str, dict[str, str]] = {}
    for column in sorted(columns):
        metadata = columns[column]
        canonical_columns[column.strip()] = {
            "labeling_method": metadata["labeling_method"].strip(),
            **(
                {"description": metadata["description"].strip()}
                if "description" in metadata
                else {}
            ),
        }
    return {
        "schema_version": SLICE_MANIFEST_SCHEMA_VERSION,
        "source": {
            "reference": source["reference"].strip(),
            "sha256": source["sha256"].lower(),
            "rows": int(source["rows"]),
        },
        "columns": canonical_columns,
    }


def load_slice_manifest(
    path: Path,
    *,
    eval_path: Path,
    eval_rows: int,
    slice_columns: list[str],
) -> dict[str, Any]:
    """Load and bind a slice sidecar to the exact evaluated CSV bytes."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read slice manifest {path}: {exc}") from exc

    errors = validate_slice_manifest_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
    assert isinstance(payload, dict)
    manifest = _canonical_manifest(payload)
    expected_columns = {column.strip() for column in slice_columns}
    actual_columns = set(manifest["columns"])
    if actual_columns != expected_columns:
        missing = sorted(expected_columns - actual_columns)
        extra = sorted(actual_columns - expected_columns)
        details = []
        if missing:
            details.append(f"missing columns {missing}")
        if extra:
            details.append(f"unexpected columns {extra}")
        raise ValueError(
            "slice manifest columns do not match requested slices: " + "; ".join(details)
        )

    source = manifest["source"]
    actual_sha256 = sha256_file(eval_path)
    if source["sha256"] != actual_sha256:
        raise ValueError(
            "slice manifest source.sha256 must match evaluated CSV "
            f"{actual_sha256}"
        )
    if source["rows"] != eval_rows:
        raise ValueError(
            "slice manifest source.rows must match evaluated CSV row count "
            f"{eval_rows}"
        )
    return manifest


def validate_slice_provenance_payload(payload: Any) -> list[str]:
    """Validate report-embedded slice provenance and observed row counts."""
    if not isinstance(payload, dict):
        return ["slice_provenance must be an object"]
    errors: list[str] = []
    manifest = payload.get("manifest")
    errors.extend(validate_slice_manifest_payload(manifest))
    if not isinstance(manifest, dict):
        return errors

    observed = payload.get("observed_row_counts")
    if not isinstance(observed, dict):
        errors.append("slice_provenance.observed_row_counts must be an object")
        return errors
    manifest_columns = set(manifest.get("columns", {}))
    if set(observed) != manifest_columns:
        errors.append("slice_provenance observed columns must match manifest columns")
    source = manifest.get("source", {})
    expected_rows = source.get("rows")
    for column, record in observed.items():
        if not isinstance(record, dict):
            errors.append(f"slice_provenance.observed_row_counts.{column} must be an object")
            continue
        rows = record.get("rows")
        if rows != expected_rows:
            errors.append(
                f"slice_provenance.observed_row_counts.{column}.rows must equal "
                "manifest source.rows"
            )
        label_counts = record.get("labels")
        if not isinstance(label_counts, dict) or not label_counts:
            errors.append(
                f"slice_provenance.observed_row_counts.{column}.labels must be a non-empty object"
            )
            continue
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 1
            for count in label_counts.values()
        ):
            errors.append(
                f"slice_provenance.observed_row_counts.{column}.labels must contain "
                "positive integer counts"
            )
        elif isinstance(rows, int) and sum(label_counts.values()) != rows:
            errors.append(
                f"slice_provenance.observed_row_counts.{column}.labels must sum to rows"
            )
    return errors
