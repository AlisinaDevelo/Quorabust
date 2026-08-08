from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from quorabust.benchmark_protocol import validate_protocol_payload

_REQUIRED_TOP_LEVEL = {
    "artifact",
    "generated_by",
    "intended_use",
    "training_metadata",
    "persisted_evaluation",
    "serving_contract",
    "caveats",
}
_REQUIRED_TRAINING_METADATA = {"feature_backend", "feature_schema"}
_REQUIRED_SERVING_OUTPUT = {"proba_duplicate", "is_duplicate", "decision_threshold"}
_REQUIRED_HOLDOUT = {
    "n",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "log_loss",
    "positive_rate",
    "predicted_positive_rate",
}
_REQUIRED_CALIBRATION = {
    "n_bins",
    "brier_score",
    "expected_calibration_error",
    "mean_predicted_probability",
    "mean_observed_rate",
    "bins",
}
_REQUIRED_MANIFEST = {
    "schema_version",
    "artifact",
    "evaluation_dataset",
    "evaluation_policy",
    "training_lineage",
    "runtime",
    "command",
    "generated_at_utc",
}
_REQUIRED_MANIFEST_SECTIONS = {
    "artifact": {"label", "sha256"},
    "evaluation_dataset": {
        "sha256",
        "rows",
        "columns",
        "positive_count",
        "positive_rate",
    },
    "evaluation_policy": {"threshold", "thresholds", "calibration_bins"},
    "runtime": {"python_version", "system", "machine", "report_git_revision"},
}
_REQUIRED_QUESTION_ID_COLUMNS = {"qid1", "qid2"}


def _missing_keys(obj: Any, keys: set[str]) -> list[str]:
    if not isinstance(obj, dict):
        return sorted(keys)
    return sorted(keys - set(obj))


def _validate_question_component_split(manifest: dict[str, Any], errors: list[str]) -> None:
    lineage = manifest.get("training_lineage")
    if not isinstance(lineage, dict):
        errors.append("evaluation_manifest.training_lineage must be an object")
    else:
        if lineage.get("split_strategy") != "question_component_holdout":
            errors.append(
                "evaluation_manifest.training_lineage.split_strategy must be "
                "question_component_holdout"
            )
        if lineage.get("require_question_ids") is not True:
            errors.append(
                "evaluation_manifest.training_lineage.require_question_ids must be true"
            )
        question_id_columns = lineage.get("question_id_columns")
        if not isinstance(question_id_columns, list) or not all(
            isinstance(column, str) for column in question_id_columns
        ):
            errors.append(
                "evaluation_manifest.training_lineage.question_id_columns must include qid1/qid2"
            )
        else:
            missing = sorted(_REQUIRED_QUESTION_ID_COLUMNS - set(question_id_columns))
            if missing:
                errors.append(
                    "evaluation_manifest.training_lineage.question_id_columns is missing: "
                    + ", ".join(missing)
                )

    evaluation_dataset = manifest.get("evaluation_dataset")
    if not isinstance(evaluation_dataset, dict):
        errors.append("evaluation_manifest.evaluation_dataset must be an object")
        return
    dataset_columns = evaluation_dataset.get("columns")
    if not isinstance(dataset_columns, list) or not all(
        isinstance(column, str) for column in dataset_columns
    ):
        errors.append("evaluation_manifest.evaluation_dataset.columns must include qid1/qid2")
        return
    missing = sorted(_REQUIRED_QUESTION_ID_COLUMNS - set(dataset_columns))
    if missing:
        errors.append(
            "evaluation_manifest.evaluation_dataset.columns is missing: " + ", ".join(missing)
        )


def _normalized_numbers(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    numbers: list[float] = []
    for item in value:
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
        ):
            return None
        numbers.append(float(item))
    return sorted(numbers)


def _validate_protocol_binding(
    manifest: Any,
    protocol_payload: Any,
    errors: list[str],
) -> None:
    protocol_errors = validate_protocol_payload(protocol_payload)
    errors.extend(f"protocol: {error}" for error in protocol_errors)
    if protocol_errors or not isinstance(protocol_payload, dict):
        return
    if not isinstance(manifest, dict):
        errors.append("protocol binding requires evaluation_manifest")
        return

    protocol_dataset = protocol_payload["dataset"]
    protocol_roles = protocol_payload["roles"]
    protocol_split = protocol_payload["split"]
    protocol_policy = protocol_payload["decision_policy"]
    protocol_dataset_hash = protocol_dataset["sha256"]
    protocol_final_hash = protocol_roles["final_holdout"]["artifact"]["sha256"]

    evaluation_dataset = manifest.get("evaluation_dataset")
    if isinstance(evaluation_dataset, dict):
        report_final_hash = evaluation_dataset.get("sha256")
        if not (
            isinstance(report_final_hash, str)
            and isinstance(protocol_final_hash, str)
            and report_final_hash.lower() == protocol_final_hash.lower()
        ):
            errors.append(
                "evaluation_manifest.evaluation_dataset.sha256 must match "
                "protocol.roles.final_holdout.artifact.sha256"
            )
    lineage = manifest.get("training_lineage")
    if not isinstance(lineage, dict):
        errors.append("protocol binding requires evaluation_manifest.training_lineage")
        return
    report_dataset_hash = lineage.get("csv_sha256")
    if not (
        isinstance(report_dataset_hash, str)
        and isinstance(protocol_dataset_hash, str)
        and report_dataset_hash.lower() == protocol_dataset_hash.lower()
    ):
        errors.append(
            "evaluation_manifest.training_lineage.csv_sha256 must match protocol.dataset.sha256"
        )
    if lineage.get("split_strategy") != protocol_split["strategy"]:
        errors.append(
            "evaluation_manifest.training_lineage.split_strategy must match protocol.split.strategy"
        )
    if lineage.get("question_id_columns") != protocol_split["question_id_columns"]:
        errors.append(
            "evaluation_manifest.training_lineage.question_id_columns must match "
            "protocol.split.question_id_columns"
        )
    if lineage.get("seed") != protocol_split["seed"]:
        errors.append("evaluation_manifest.training_lineage.seed must match protocol.split.seed")
    report_eval_fraction = lineage.get("eval_fraction")
    protocol_eval_fraction = protocol_split["eval_fraction"]
    if (
        not isinstance(report_eval_fraction, (int, float))
        or isinstance(report_eval_fraction, bool)
        or not math.isclose(float(report_eval_fraction), float(protocol_eval_fraction))
    ):
        errors.append(
            "evaluation_manifest.training_lineage.eval_fraction must match "
            "protocol.split.eval_fraction"
        )
    if lineage.get("threshold_metric") != protocol_policy["threshold_metric"]:
        errors.append(
            "evaluation_manifest.training_lineage.threshold_metric must match "
            "protocol.decision_policy.threshold_metric"
        )

    report_policy = manifest.get("evaluation_policy")
    if not isinstance(report_policy, dict):
        errors.append("protocol binding requires evaluation_manifest.evaluation_policy")
        return
    report_thresholds = _normalized_numbers(report_policy.get("thresholds"))
    protocol_thresholds = _normalized_numbers(protocol_policy["threshold_candidates"])
    if report_thresholds != protocol_thresholds:
        errors.append(
            "evaluation_manifest.evaluation_policy.thresholds must match "
            "protocol.decision_policy.threshold_candidates"
        )


def validate_report_payload(
    payload: Any,
    *,
    require_holdout: bool = False,
    require_calibration: bool = False,
    require_manifest: bool = False,
    require_question_component_split: bool = False,
    protocol_payload: Any | None = None,
) -> list[str]:
    """Return validation errors for a machine-readable Quorabust report payload."""
    if not isinstance(payload, dict):
        return ["report must be a JSON object"]

    errors: list[str] = []
    protocol_bound = protocol_payload is not None
    if protocol_bound:
        require_holdout = True
        require_calibration = True
        require_manifest = True
        require_question_component_split = True
    for key in _missing_keys(payload, _REQUIRED_TOP_LEVEL):
        errors.append(f"missing top-level field: {key}")

    for key in _missing_keys(payload.get("training_metadata"), _REQUIRED_TRAINING_METADATA):
        errors.append(f"missing training_metadata field: {key}")

    serving_contract = payload.get("serving_contract")
    if not isinstance(serving_contract, dict):
        errors.append("serving_contract must be an object")
    else:
        output = serving_contract.get("output")
        for key in _missing_keys(output, _REQUIRED_SERVING_OUTPUT):
            errors.append(f"missing serving_contract.output field: {key}")

    holdout = payload.get("holdout_evaluation")
    if require_holdout and holdout is None:
        errors.append("missing holdout_evaluation")
    if holdout is not None:
        for key in _missing_keys(holdout, _REQUIRED_HOLDOUT):
            errors.append(f"missing holdout_evaluation field: {key}")

    calibration = payload.get("calibration")
    if require_calibration and calibration is None:
        errors.append("missing calibration")
    if calibration is not None:
        for key in _missing_keys(calibration, _REQUIRED_CALIBRATION):
            errors.append(f"missing calibration field: {key}")
        bins = calibration.get("bins") if isinstance(calibration, dict) else None
        if not isinstance(bins, list) or not bins:
            errors.append("calibration.bins must be a non-empty list")

    manifest = payload.get("evaluation_manifest")
    if (
        require_manifest or require_question_component_split or protocol_bound
    ) and manifest is None:
        errors.append("missing evaluation_manifest")
    if manifest is not None:
        for key in _missing_keys(manifest, _REQUIRED_MANIFEST):
            errors.append(f"missing evaluation_manifest field: {key}")
        if isinstance(manifest, dict):
            for section, keys in _REQUIRED_MANIFEST_SECTIONS.items():
                value = manifest.get(section)
                if not isinstance(value, dict):
                    errors.append(f"evaluation_manifest.{section} must be an object")
                    continue
                for key in _missing_keys(value, keys):
                    errors.append(f"missing evaluation_manifest.{section} field: {key}")
            if require_question_component_split:
                _validate_question_component_split(manifest, errors)
    if protocol_bound:
        _validate_protocol_binding(manifest, protocol_payload, errors)

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a machine-readable Quorabust JSON report.",
    )
    parser.add_argument("--report", type=Path, required=True, help="Path to report JSON")
    parser.add_argument(
        "--require-holdout",
        action="store_true",
        help="Fail unless holdout_evaluation is present",
    )
    parser.add_argument(
        "--require-calibration",
        action="store_true",
        help="Fail unless calibration diagnostics are present",
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Fail unless holdout reproducibility metadata is present",
    )
    parser.add_argument(
        "--require-question-component-split",
        action="store_true",
        help="Fail unless the report uses a leakage-safe qid1/qid2 component holdout",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Optional benchmark protocol JSON to bind to the report manifest",
    )
    args = parser.parse_args(argv)

    if not args.report.is_file():
        print(f"File not found: {args.report}", file=sys.stderr)
        return 1
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        return 1

    protocol_payload = None
    if args.protocol is not None:
        if not args.protocol.is_file():
            print(f"File not found: {args.protocol}", file=sys.stderr)
            return 1
        try:
            protocol_payload = json.loads(args.protocol.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Invalid protocol JSON: {exc}", file=sys.stderr)
            return 1

    errors = validate_report_payload(
        payload,
        require_holdout=args.require_holdout or protocol_payload is not None,
        require_calibration=args.require_calibration or protocol_payload is not None,
        require_manifest=args.require_manifest or protocol_payload is not None,
        require_question_component_split=(
            args.require_question_component_split or protocol_payload is not None
        ),
        protocol_payload=protocol_payload,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
