from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def _missing_keys(obj: Any, keys: set[str]) -> list[str]:
    if not isinstance(obj, dict):
        return sorted(keys)
    return sorted(keys - set(obj))


def validate_report_payload(
    payload: Any,
    *,
    require_holdout: bool = False,
    require_calibration: bool = False,
) -> list[str]:
    """Return validation errors for a machine-readable Quorabust report payload."""
    if not isinstance(payload, dict):
        return ["report must be a JSON object"]

    errors: list[str] = []
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
    args = parser.parse_args(argv)

    if not args.report.is_file():
        print(f"File not found: {args.report}", file=sys.stderr)
        return 1
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        return 1

    errors = validate_report_payload(
        payload,
        require_holdout=args.require_holdout,
        require_calibration=args.require_calibration,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
