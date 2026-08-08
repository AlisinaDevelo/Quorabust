"""Validation for reproducible, leakage-aware benchmark protocol manifests."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_QUESTION_ID_COLUMNS = {"qid1", "qid2"}
_ROLE_ACTIVITIES = {
    "train": {"fit"},
    "tuning": {"model_selection", "threshold_selection"},
    "calibration": {"probability_calibration"},
    "final_holdout": {"final_evaluation"},
}
_THRESHOLD_METRICS = {"accuracy", "precision", "recall", "f1"}
_CALIBRATION_METHODS = {"sigmoid", "isotonic"}
_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "protocol_name",
    "evidence_scope",
    "dataset",
    "roles",
    "split",
    "decision_policy",
    "provenance",
    "safeguards",
}
_REQUIRED_DATASET = {
    "name",
    "source_reference",
    "sha256",
    "license",
    "terms",
    "raw_data_policy",
    "audit",
}
_REQUIRED_AUDIT = {"reference", "sha256", "status", "source_sha256", "require_question_ids"}
_REQUIRED_ROLE = {"purpose", "allowed_activities", "artifact"}
_REQUIRED_ARTIFACT = {"reference", "sha256"}
_REQUIRED_SPLIT = {
    "strategy",
    "question_id_columns",
    "seed",
    "eval_fraction",
    "manifest",
}
_REQUIRED_DECISION_POLICY = {
    "threshold_metric",
    "threshold_candidates",
    "threshold_source_role",
    "calibration_method",
    "calibration_source_role",
    "final_holdout_role",
}
_REQUIRED_PROVENANCE = {
    "git_revision",
    "python_version",
    "dependency_lock",
    "command",
    "machine",
}
_REQUIRED_SAFEGUARDS = {
    "roles_are_disjoint",
    "final_holdout_used_for_tuning",
    "final_holdout_used_for_calibration",
    "final_holdout_used_for_model_selection",
    "raw_data_committed",
    "public_quality_claims_allowed",
}


def _missing_keys(value: Any, keys: set[str]) -> list[str]:
    if not isinstance(value, dict):
        return sorted(keys)
    return sorted(keys - set(value))


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_text(value: Any, label: str, errors: list[str]) -> None:
    if not _non_empty_text(value):
        errors.append(f"{label} must be a non-empty string")


def _validate_sha256(value: Any, label: str, errors: list[str]) -> bool:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{label} must be a 64-character hexadecimal SHA-256 digest")
        return False
    return True


def _validate_artifact(value: Any, label: str, errors: list[str]) -> str | None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return None
    for key in _missing_keys(value, _REQUIRED_ARTIFACT):
        errors.append(f"missing {label} field: {key}")
    _validate_text(value.get("reference"), f"{label}.reference", errors)
    digest = value.get("sha256")
    digest_valid = _validate_sha256(digest, f"{label}.sha256", errors)
    return digest.lower() if digest_valid and isinstance(digest, str) else None


def _validate_dataset(dataset: Any, errors: list[str]) -> None:
    if not isinstance(dataset, dict):
        errors.append("dataset must be an object")
        return
    for key in _missing_keys(dataset, _REQUIRED_DATASET):
        errors.append(f"missing dataset field: {key}")
    for key in ("name", "source_reference", "license", "terms"):
        _validate_text(dataset.get(key), f"dataset.{key}", errors)
    if dataset.get("raw_data_policy") != "external_not_committed":
        errors.append("dataset.raw_data_policy must be external_not_committed")
    source_digest = dataset.get("sha256")
    source_digest_valid = _validate_sha256(source_digest, "dataset.sha256", errors)

    audit = dataset.get("audit")
    if not isinstance(audit, dict):
        errors.append("dataset.audit must be an object")
        return
    for key in _missing_keys(audit, _REQUIRED_AUDIT):
        errors.append(f"missing dataset.audit field: {key}")
    _validate_text(audit.get("reference"), "dataset.audit.reference", errors)
    _validate_sha256(audit.get("sha256"), "dataset.audit.sha256", errors)
    if audit.get("status") != "pass":
        errors.append("dataset.audit.status must be pass")
    if audit.get("require_question_ids") is not True:
        errors.append("dataset.audit.require_question_ids must be true")
    audit_source_digest = audit.get("source_sha256")
    audit_digest_valid = _validate_sha256(
        audit_source_digest,
        "dataset.audit.source_sha256",
        errors,
    )
    if (
        source_digest_valid
        and audit_digest_valid
        and isinstance(source_digest, str)
        and isinstance(audit_source_digest, str)
        and source_digest.lower() != audit_source_digest.lower()
    ):
        errors.append("dataset.audit.source_sha256 must match dataset.sha256")


def _validate_roles(roles: Any, errors: list[str]) -> None:
    if not isinstance(roles, dict):
        errors.append("roles must be an object")
        return
    expected_roles = set(_ROLE_ACTIVITIES)
    for role in sorted(expected_roles - set(roles)):
        errors.append(f"missing roles field: {role}")
    for role in sorted(set(roles) - expected_roles):
        errors.append(f"roles contains unsupported role: {role}")

    artifact_digests: dict[str, str] = {}
    for role, expected_activities in _ROLE_ACTIVITIES.items():
        value = roles.get(role)
        if not isinstance(value, dict):
            errors.append(f"roles.{role} must be an object")
            continue
        for key in _missing_keys(value, _REQUIRED_ROLE):
            errors.append(f"missing roles.{role} field: {key}")
        _validate_text(value.get("purpose"), f"roles.{role}.purpose", errors)
        activities = value.get("allowed_activities")
        if not isinstance(activities, list) or not all(
            isinstance(activity, str) for activity in activities
        ):
            errors.append(f"roles.{role}.allowed_activities must be a list of strings")
        elif set(activities) != expected_activities:
            expected = ", ".join(sorted(expected_activities))
            errors.append(
                f"roles.{role}.allowed_activities must contain exactly: {expected}"
            )
        digest = _validate_artifact(value.get("artifact"), f"roles.{role}.artifact", errors)
        if digest is not None:
            artifact_digests[role] = digest

    by_digest: dict[str, list[str]] = {}
    for role, digest in artifact_digests.items():
        by_digest.setdefault(digest, []).append(role)
    for digest, matching_roles in sorted(by_digest.items()):
        if len(matching_roles) > 1:
            errors.append(
                "roles artifacts must have distinct SHA-256 digests; shared by: "
                + ", ".join(sorted(matching_roles))
            )


def _validate_split(split: Any, errors: list[str]) -> None:
    if not isinstance(split, dict):
        errors.append("split must be an object")
        return
    for key in _missing_keys(split, _REQUIRED_SPLIT):
        errors.append(f"missing split field: {key}")
    if split.get("strategy") != "question_component_holdout":
        errors.append("split.strategy must be question_component_holdout")

    question_id_columns = split.get("question_id_columns")
    if not isinstance(question_id_columns, list) or not all(
        isinstance(column, str) for column in question_id_columns
    ):
        errors.append("split.question_id_columns must be a list containing qid1 and qid2")
    elif not _QUESTION_ID_COLUMNS.issubset(question_id_columns):
        missing = ", ".join(sorted(_QUESTION_ID_COLUMNS - set(question_id_columns)))
        errors.append(f"split.question_id_columns is missing: {missing}")
    elif len(question_id_columns) != len(set(question_id_columns)):
        errors.append("split.question_id_columns must not contain duplicates")

    seed = split.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append("split.seed must be a non-negative integer")
    eval_fraction = split.get("eval_fraction")
    if (
        not isinstance(eval_fraction, (int, float))
        or isinstance(eval_fraction, bool)
        or not math.isfinite(float(eval_fraction))
        or not 0.0 < float(eval_fraction) < 1.0
    ):
        errors.append("split.eval_fraction must be finite and strictly between 0 and 1")
    _validate_artifact(split.get("manifest"), "split.manifest", errors)


def _validate_decision_policy(policy: Any, errors: list[str]) -> None:
    if not isinstance(policy, dict):
        errors.append("decision_policy must be an object")
        return
    for key in _missing_keys(policy, _REQUIRED_DECISION_POLICY):
        errors.append(f"missing decision_policy field: {key}")
    if policy.get("threshold_metric") not in _THRESHOLD_METRICS:
        errors.append(
            "decision_policy.threshold_metric must be one of: "
            + ", ".join(sorted(_THRESHOLD_METRICS))
        )
    candidates = policy.get("threshold_candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("decision_policy.threshold_candidates must be a non-empty list")
    else:
        valid_candidates = True
        for candidate in candidates:
            if (
                not isinstance(candidate, (int, float))
                or isinstance(candidate, bool)
                or not math.isfinite(float(candidate))
                or not 0.0 < float(candidate) < 1.0
            ):
                valid_candidates = False
        if not valid_candidates:
            errors.append(
                "decision_policy.threshold_candidates must contain finite values "
                "strictly between 0 and 1"
            )
        elif len(candidates) != len(set(candidates)):
            errors.append("decision_policy.threshold_candidates must not contain duplicates")
    if policy.get("threshold_source_role") != "tuning":
        errors.append("decision_policy.threshold_source_role must be tuning")
    if policy.get("calibration_method") not in _CALIBRATION_METHODS:
        errors.append(
            "decision_policy.calibration_method must be one of: "
            + ", ".join(sorted(_CALIBRATION_METHODS))
        )
    if policy.get("calibration_source_role") != "calibration":
        errors.append("decision_policy.calibration_source_role must be calibration")
    if policy.get("final_holdout_role") != "final_holdout":
        errors.append("decision_policy.final_holdout_role must be final_holdout")


def _validate_provenance(provenance: Any, errors: list[str]) -> None:
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
        return
    for key in _missing_keys(provenance, _REQUIRED_PROVENANCE):
        errors.append(f"missing provenance field: {key}")
    git_revision = provenance.get("git_revision")
    if not isinstance(git_revision, str) or _GIT_REVISION_RE.fullmatch(git_revision) is None:
        errors.append("provenance.git_revision must be a 40-character hexadecimal commit SHA")
    for key in ("python_version", "command", "machine"):
        _validate_text(provenance.get(key), f"provenance.{key}", errors)
    _validate_artifact(provenance.get("dependency_lock"), "provenance.dependency_lock", errors)


def _validate_safeguards(safeguards: Any, errors: list[str]) -> None:
    if not isinstance(safeguards, dict):
        errors.append("safeguards must be an object")
        return
    for key in _missing_keys(safeguards, _REQUIRED_SAFEGUARDS):
        errors.append(f"missing safeguards field: {key}")
    expected_false = {
        "final_holdout_used_for_tuning",
        "final_holdout_used_for_calibration",
        "final_holdout_used_for_model_selection",
        "raw_data_committed",
        "public_quality_claims_allowed",
    }
    if safeguards.get("roles_are_disjoint") is not True:
        errors.append("safeguards.roles_are_disjoint must be true")
    for key in sorted(expected_false):
        if safeguards.get(key) is not False:
            errors.append(f"safeguards.{key} must be false")


def validate_protocol_payload(payload: Any) -> list[str]:
    """Return validation errors for a benchmark protocol manifest."""
    if not isinstance(payload, dict):
        return ["protocol must be a JSON object"]

    errors: list[str] = []
    for key in _missing_keys(payload, _REQUIRED_TOP_LEVEL):
        errors.append(f"missing top-level field: {key}")
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PROTOCOL_SCHEMA_VERSION}")
    _validate_text(payload.get("protocol_name"), "protocol_name", errors)
    if payload.get("evidence_scope") != "protocol_only_no_quality_claim":
        errors.append("evidence_scope must be protocol_only_no_quality_claim")
    _validate_dataset(payload.get("dataset"), errors)
    _validate_roles(payload.get("roles"), errors)
    _validate_split(payload.get("split"), errors)
    _validate_decision_policy(payload.get("decision_policy"), errors)
    _validate_provenance(payload.get("provenance"), errors)
    _validate_safeguards(payload.get("safeguards"), errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a leakage-aware Quorabust benchmark protocol manifest.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Path to the versioned benchmark protocol JSON manifest",
    )
    args = parser.parse_args(argv)

    if not args.protocol.is_file():
        print(f"File not found: {args.protocol}", file=sys.stderr)
        return 1
    try:
        payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid protocol JSON: {exc}", file=sys.stderr)
        return 1

    errors = validate_protocol_payload(payload)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {args.protocol.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
