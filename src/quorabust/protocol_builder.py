"""Build canonical benchmark protocol manifests from audited artifact files."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

from quorabust.benchmark_protocol import (
    _CALIBRATION_METHODS,
    _ROLE_ACTIVITIES,
    _THRESHOLD_METRICS,
    validate_protocol_payload,
)
from quorabust.lineage import git_revision, sha256_file

_REQUIRED_CONFIG = {
    "protocol_name",
    "dataset",
    "roles",
    "split",
    "decision_policy",
    "dependency_lock_path",
    "repository_path",
    "command",
}
_REQUIRED_DATASET = {
    "name",
    "source_path",
    "source_reference",
    "license",
    "terms",
    "audit_path",
}
_REQUIRED_ROLE = {"purpose", "path"}
_REQUIRED_SPLIT = {"manifest_path", "seed", "eval_fraction", "question_id_columns"}
_REQUIRED_DECISION_POLICY = {
    "threshold_metric",
    "threshold_candidates",
    "calibration_method",
}


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _missing_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ValueError(f"missing {label} field(s): {', '.join(missing)}")


def _require_non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_fraction(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError(f"{label} must be finite and strictly between 0 and 1")
    return float(value)


def _require_cost(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _require_question_id_columns(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a non-empty list of strings")
    columns = [item.strip() for item in value]
    if not all(columns):
        raise ValueError(f"{label} must not contain blank column names")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{label} must not contain duplicates")
    if not {"qid1", "qid2"}.issubset(columns):
        raise ValueError(f"{label} must contain qid1 and qid2")
    return sorted(columns)


def _require_threshold_candidates(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list of numbers")
    candidates: list[float] = []
    for candidate in value:
        if (
            not isinstance(candidate, (int, float))
            or isinstance(candidate, bool)
            or not math.isfinite(float(candidate))
            or not 0.0 < float(candidate) < 1.0
        ):
            raise ValueError(f"{label} must contain finite values strictly between 0 and 1")
        candidates.append(float(candidate))
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"{label} must not contain duplicates")
    return sorted(candidates)


def _validate_config_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Validate scalar config policy before opening any external artifact."""
    _missing_keys(config, _REQUIRED_CONFIG, "config")
    _require_text(config["protocol_name"], "config.protocol_name")
    _require_text(config["command"], "config.command")
    _require_text(config["dependency_lock_path"], "config.dependency_lock_path")
    _require_text(config["repository_path"], "config.repository_path")

    dataset = _require_object(config["dataset"], "config.dataset")
    _missing_keys(dataset, _REQUIRED_DATASET, "config.dataset")
    for key in (
        "name",
        "source_path",
        "source_reference",
        "license",
        "terms",
        "audit_path",
    ):
        _require_text(dataset[key], f"config.dataset.{key}")

    roles = _require_object(config["roles"], "config.roles")
    expected_roles = set(_ROLE_ACTIVITIES)
    missing_roles = sorted(expected_roles - set(roles))
    if missing_roles:
        raise ValueError("missing config.roles field(s): " + ", ".join(missing_roles))
    unsupported_roles = sorted(set(roles) - expected_roles)
    if unsupported_roles:
        raise ValueError("unsupported config.roles field(s): " + ", ".join(unsupported_roles))
    for role in _ROLE_ACTIVITIES:
        role_config = _require_object(roles[role], f"config.roles.{role}")
        _missing_keys(role_config, _REQUIRED_ROLE, f"config.roles.{role}")
        _require_text(role_config["purpose"], f"config.roles.{role}.purpose")
        _require_text(role_config["path"], f"config.roles.{role}.path")

    split = _require_object(config["split"], "config.split")
    _missing_keys(split, _REQUIRED_SPLIT, "config.split")
    _require_text(split["manifest_path"], "config.split.manifest_path")
    question_id_columns = _require_question_id_columns(
        split["question_id_columns"],
        "config.split.question_id_columns",
    )
    seed = _require_non_negative_int(split["seed"], "config.split.seed")
    eval_fraction = _require_fraction(split["eval_fraction"], "config.split.eval_fraction")

    policy = _require_object(config["decision_policy"], "config.decision_policy")
    _missing_keys(policy, _REQUIRED_DECISION_POLICY, "config.decision_policy")
    threshold_metric = _require_text(
        policy["threshold_metric"],
        "config.decision_policy.threshold_metric",
    )
    if threshold_metric not in _THRESHOLD_METRICS:
        allowed = ", ".join(sorted(_THRESHOLD_METRICS))
        raise ValueError(
            "config.decision_policy.threshold_metric must be one of: " + allowed
        )
    calibration_method = _require_text(
        policy["calibration_method"],
        "config.decision_policy.calibration_method",
    )
    if calibration_method not in _CALIBRATION_METHODS:
        allowed = ", ".join(sorted(_CALIBRATION_METHODS))
        raise ValueError(
            "config.decision_policy.calibration_method must be one of: " + allowed
        )
    threshold_candidates = _require_threshold_candidates(
        policy["threshold_candidates"],
        "config.decision_policy.threshold_candidates",
    )
    threshold_costs: dict[str, float] = {}
    cost_keys = ("false_positive_cost", "false_negative_cost")
    present_cost_keys = [key for key in cost_keys if key in policy]
    if present_cost_keys or threshold_metric == "expected_cost":
        missing_cost_keys = [key for key in cost_keys if key not in policy]
        if missing_cost_keys:
            raise ValueError(
                "config.decision_policy expected_cost requires: "
                + ", ".join(missing_cost_keys)
            )
        threshold_costs = {
            key: _require_cost(policy[key], f"config.decision_policy.{key}")
            for key in cost_keys
        }
        if sum(threshold_costs.values()) == 0.0:
            raise ValueError(
                "config.decision_policy requires at least one positive threshold cost"
            )
    normalized = {
        "question_id_columns": question_id_columns,
        "seed": seed,
        "eval_fraction": eval_fraction,
        "threshold_metric": threshold_metric,
        "threshold_candidates": threshold_candidates,
        "calibration_method": calibration_method,
    }
    if threshold_costs:
        normalized["threshold_costs"] = threshold_costs
    return normalized


def _resolve_file(value: Any, label: str, base_dir: Path) -> Path:
    raw = _require_text(value, label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    return path


def _resolve_directory(value: Any, label: str, base_dir: Path) -> Path:
    raw = _require_text(value, label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"{label} does not exist or is not a directory: {path}")
    return path


def _reference(value: Any, default: str, label: str) -> str:
    if value is None:
        return default
    return _require_text(value, label)


def _artifact(path: Path, reference: str) -> dict[str, str]:
    return {"reference": reference, "sha256": sha256_file(path)}


def _load_audit(path: Path, source_sha256: str) -> dict[str, Any]:
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read audit manifest {path}: {exc}") from exc
    if not isinstance(audit, dict):
        raise ValueError("audit manifest must be a JSON object")
    if audit.get("status") != "pass":
        raise ValueError("audit manifest status must be pass")
    question_ids = audit.get("question_ids")
    if not isinstance(question_ids, dict) or question_ids.get("present") is not True:
        raise ValueError("audit manifest must record complete question IDs")
    source = audit.get("source")
    audit_source_sha256 = source.get("sha256") if isinstance(source, dict) else None
    if (
        not isinstance(audit_source_sha256, str)
        or audit_source_sha256.lower() != source_sha256.lower()
    ):
        raise ValueError("audit manifest source SHA-256 does not match source_path bytes")
    return audit


def build_protocol_payload(
    config: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Build and validate a protocol payload from a path-based config."""
    config = _require_object(config, "config")
    normalized = _validate_config_contract(config)
    root = (base_dir or Path.cwd()).resolve()

    protocol_name = _require_text(config["protocol_name"], "config.protocol_name")
    command = _require_text(config["command"], "config.command")
    dataset_config = _require_object(config["dataset"], "config.dataset")
    _missing_keys(dataset_config, _REQUIRED_DATASET, "config.dataset")
    source_path = _resolve_file(dataset_config["source_path"], "config.dataset.source_path", root)
    audit_path = _resolve_file(dataset_config["audit_path"], "config.dataset.audit_path", root)
    source_sha256 = sha256_file(source_path)
    audit = _load_audit(audit_path, source_sha256)

    roles_config = _require_object(config["roles"], "config.roles")
    expected_roles = set(_ROLE_ACTIVITIES)
    missing_roles = sorted(expected_roles - set(roles_config))
    if missing_roles:
        raise ValueError("missing config.roles field(s): " + ", ".join(missing_roles))
    unsupported_roles = sorted(set(roles_config) - expected_roles)
    if unsupported_roles:
        raise ValueError("unsupported config.roles field(s): " + ", ".join(unsupported_roles))
    roles: dict[str, Any] = {}
    for role, activities in _ROLE_ACTIVITIES.items():
        role_config = _require_object(roles_config[role], f"config.roles.{role}")
        _missing_keys(role_config, _REQUIRED_ROLE, f"config.roles.{role}")
        role_path = _resolve_file(role_config["path"], f"config.roles.{role}.path", root)
        roles[role] = {
            "purpose": _require_text(
                role_config["purpose"], f"config.roles.{role}.purpose"
            ),
            "allowed_activities": sorted(activities),
            "artifact": _artifact(
                role_path,
                _reference(
                    role_config.get("reference"),
                    role_path.name,
                    f"config.roles.{role}.reference",
                ),
            ),
        }

    split_config = _require_object(config["split"], "config.split")
    _missing_keys(split_config, _REQUIRED_SPLIT, "config.split")
    split_manifest_path = _resolve_file(
        split_config["manifest_path"],
        "config.split.manifest_path",
        root,
    )
    split = {
        "strategy": "question_component_holdout",
        "question_id_columns": normalized["question_id_columns"],
        "seed": normalized["seed"],
        "eval_fraction": normalized["eval_fraction"],
        "manifest": _artifact(
            split_manifest_path,
            _reference(
                split_config.get("manifest_reference"),
                split_manifest_path.name,
                "config.split.manifest_reference",
            ),
        ),
    }

    policy_config = _require_object(config["decision_policy"], "config.decision_policy")
    _missing_keys(policy_config, _REQUIRED_DECISION_POLICY, "config.decision_policy")
    decision_policy = {
        "threshold_metric": normalized["threshold_metric"],
        "threshold_candidates": normalized["threshold_candidates"],
        "threshold_source_role": "tuning",
        "calibration_method": normalized["calibration_method"],
        "calibration_source_role": "calibration",
        "final_holdout_role": "final_holdout",
    }
    if normalized.get("threshold_costs"):
        decision_policy.update(normalized["threshold_costs"])

    dependency_lock_path = _resolve_file(
        config["dependency_lock_path"],
        "config.dependency_lock_path",
        root,
    )
    repository_path = _resolve_directory(
        config["repository_path"],
        "config.repository_path",
        root,
    )
    revision = git_revision(cwd=repository_path)
    if revision == "unknown":
        raise ValueError(f"unable to resolve a Git commit from repository path: {repository_path}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_name": protocol_name,
        "evidence_scope": "protocol_only_no_quality_claim",
        "dataset": {
            "name": _require_text(dataset_config["name"], "config.dataset.name"),
            "source_reference": _require_text(
                dataset_config["source_reference"],
                "config.dataset.source_reference",
            ),
            "sha256": source_sha256,
            "license": _require_text(dataset_config["license"], "config.dataset.license"),
            "terms": _require_text(dataset_config["terms"], "config.dataset.terms"),
            "raw_data_policy": "external_not_committed",
            "audit": {
                "reference": _reference(
                    dataset_config.get("audit_reference"),
                    audit_path.name,
                    "config.dataset.audit_reference",
                ),
                "sha256": sha256_file(audit_path),
                "status": audit["status"],
                "source_sha256": source_sha256,
                "require_question_ids": True,
            },
        },
        "roles": roles,
        "split": split,
        "decision_policy": decision_policy,
        "provenance": {
            "git_revision": revision,
            "python_version": platform.python_version(),
            "dependency_lock": _artifact(
                dependency_lock_path,
                _reference(
                    config.get("dependency_lock_reference"),
                    dependency_lock_path.name,
                    "config.dependency_lock_reference",
                ),
            ),
            "command": command,
            "machine": f"{platform.system()}/{platform.machine()}",
        },
        "safeguards": {
            "roles_are_disjoint": True,
            "final_holdout_used_for_tuning": False,
            "final_holdout_used_for_calibration": False,
            "final_holdout_used_for_model_selection": False,
            "raw_data_committed": False,
            "public_quality_claims_allowed": False,
        },
    }
    errors = validate_protocol_payload(payload)
    if errors:
        raise ValueError("generated protocol failed validation: " + "; ".join(errors))
    return payload


def build_protocol(
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Read a config, build a canonical manifest, and write it to disk."""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid builder config JSON: {exc}") from exc
    payload = build_protocol_payload(config, base_dir=config_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a canonical Quorabust benchmark protocol from audited artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the path-based protocol builder JSON config",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output protocol JSON manifest",
    )
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"File not found: {args.config}", file=sys.stderr)
        return 1
    try:
        payload = build_protocol(args.config, args.out)
    except (OSError, ValueError) as exc:
        print(f"Unable to build protocol: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.out.resolve()} source_sha256={payload['dataset']['sha256']} "
        f"git_revision={payload['provenance']['git_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
