from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

from quorabust.calibration import calibrate_classifier
from quorabust.cli import _load_quora_csv, _parse_thresholds
from quorabust.lineage import git_revision, sha256_file
from quorabust.model import select_decision_threshold
from quorabust.persist import load_classifier, save_classifier, save_metadata_sidecar
from quorabust.registry import append_model_record
from quorabust.report import calibration_summary


def _calibration_command(argv: list[str] | None) -> str:
    arguments = sys.argv[1:] if argv is None else argv
    return shlex.join(["quorabust-calibrate", *(str(value) for value in arguments)])


def _calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    summary = calibration_summary(labels, probabilities)
    return {
        "n": len(labels),
        "n_bins": summary["n_bins"],
        "brier_score": summary["brier_score"],
        "expected_calibration_error": summary["expected_calibration_error"],
        "mean_predicted_probability": summary["mean_predicted_probability"],
        "mean_observed_rate": summary["mean_observed_rate"],
        "bins": summary["bins"],
    }


def _question_ids_are_disjoint(left: Any, right: Any) -> bool:
    columns = {"qid1", "qid2"}
    if not columns.issubset(left.columns) or not columns.issubset(right.columns):
        return True
    if left[list(columns)].isna().any().any() or right[list(columns)].isna().any().any():
        return True
    left_ids = set(left["qid1"].astype(str)) | set(left["qid2"].astype(str))
    right_ids = set(right["qid1"].astype(str)) | set(right["qid2"].astype(str))
    return left_ids.isdisjoint(right_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate a saved Quorabust artifact on independent labeled data.",
    )
    parser.add_argument("--model", type=Path, required=True, help="Input artifact (.pkl)")
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        required=True,
        help="Independent labeled CSV used only to fit the probability calibrator",
    )
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        required=True,
        help="Independent labeled CSV used only to select the decision threshold",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output calibrated artifact")
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability mapping to fit on the calibration CSV",
    )
    parser.add_argument(
        "--thresholds",
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated calibrated thresholds for policy selection",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "precision", "recall", "f1"],
        default="f1",
        help="Metric to optimize on the threshold CSV",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Write calibrated artifact metadata as JSON",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Append a calibrated artifact record to a JSONL registry",
    )
    args = parser.parse_args(argv)

    for path in (args.model, args.calibration_csv, args.threshold_csv):
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
    if args.out.resolve() == args.model.resolve():
        print("--out must be different from --model", file=sys.stderr)
        return 1
    if args.calibration_csv.resolve() == args.threshold_csv.resolve():
        print("--calibration-csv and --threshold-csv must be different files", file=sys.stderr)
        return 1
    try:
        thresholds = _parse_thresholds(args.thresholds)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        calibration_df = _load_quora_csv(args.calibration_csv)
        threshold_df = _load_quora_csv(args.threshold_csv)
        builder, classifier, meta = load_classifier(args.model)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"Unable to load calibration inputs: {exc}", file=sys.stderr)
        return 1

    if meta.get("calibration_method"):
        print("input artifact is already calibrated", file=sys.stderr)
        return 1

    calibration_sha256 = sha256_file(args.calibration_csv)
    threshold_sha256 = sha256_file(args.threshold_csv)
    if calibration_sha256 == threshold_sha256:
        print(
            "calibration and threshold CSVs have identical bytes; provide independent data",
            file=sys.stderr,
        )
        return 1
    source_hashes = {meta.get("csv_sha256"), meta.get("eval_csv_sha256")}
    if calibration_sha256 in source_hashes or threshold_sha256 in source_hashes:
        print(
            "calibration and threshold data must not reuse the training/evaluation CSV",
            file=sys.stderr,
        )
        return 1
    if not _question_ids_are_disjoint(calibration_df, threshold_df):
        print(
            "calibration and threshold question IDs overlap; provide disjoint components",
            file=sys.stderr,
        )
        return 1

    y_calibration = calibration_df["is_duplicate"].astype(int).to_numpy()
    y_threshold = threshold_df["is_duplicate"].astype(int).to_numpy()
    threshold_classes = np.unique(y_threshold)
    if len(threshold_classes) < 2:
        print("threshold CSV must contain both label classes", file=sys.stderr)
        return 1

    try:
        calibration_features = builder.transform_frame(calibration_df)
        threshold_features = builder.transform_frame(threshold_df)
        raw_calibration = classifier.predict_proba(calibration_features)[:, 1]
        calibrated_classifier = calibrate_classifier(
            classifier,
            calibration_features,
            y_calibration,
            method=args.calibration_method,
        )
        calibrated_calibration = calibrated_classifier.predict_proba(calibration_features)[:, 1]
        calibrated_threshold = calibrated_classifier.predict_proba(threshold_features)[:, 1]
        selected_threshold = select_decision_threshold(
            y_threshold,
            calibrated_threshold,
            thresholds=thresholds,
            optimize_for=args.threshold_metric,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Unable to fit calibration: {exc}", file=sys.stderr)
        return 1

    input_artifact_sha256 = sha256_file(args.model)
    calibrated_meta = {
        **meta,
        "calibration_method": args.calibration_method,
        "calibration_csv_sha256": calibration_sha256,
        "threshold_csv_sha256": threshold_sha256,
        "n_calibration": len(calibration_df),
        "n_threshold": len(threshold_df),
        "calibration_command": _calibration_command(argv),
        "calibration_git_revision": git_revision(),
        "calibrated_from_artifact_sha256": input_artifact_sha256,
        "calibration_metrics": {
            "raw": _calibration_metrics(y_calibration, raw_calibration),
            "calibrated": _calibration_metrics(y_calibration, calibrated_calibration),
        },
        "decision_threshold": selected_threshold["threshold"],
        "decision_threshold_source": "calibration_threshold_csv",
        "decision_threshold_metric": args.threshold_metric,
        "decision_threshold_metrics": {
            key: value for key, value in selected_threshold.items() if key != "threshold"
        },
    }

    save_classifier(args.out, builder, calibrated_classifier, meta=calibrated_meta)
    artifact_sha256 = sha256_file(args.out)
    if args.metadata_out is not None:
        save_metadata_sidecar(
            args.metadata_out,
            {**calibrated_meta, "artifact_sha256": artifact_sha256},
        )
    if args.registry_dir is not None:
        append_model_record(
            args.registry_dir,
            {
                "artifact": str(args.out.resolve()),
                "artifact_sha256": artifact_sha256,
                "feature_backend": calibrated_meta.get("feature_backend"),
                "git_revision": calibrated_meta.get("git_revision"),
                "calibration_method": args.calibration_method,
                "calibration_csv_sha256": calibration_sha256,
                "threshold_csv_sha256": threshold_sha256,
                "decision_threshold": selected_threshold["threshold"],
                "decision_threshold_metric": args.threshold_metric,
                "calibrated_from_artifact_sha256": input_artifact_sha256,
            },
        )

    raw_ece = calibrated_meta["calibration_metrics"]["raw"]["expected_calibration_error"]
    calibrated_ece = calibrated_meta["calibration_metrics"]["calibrated"][
        "expected_calibration_error"
    ]
    print(
        f"calibration: method={args.calibration_method}, raw_ece={raw_ece:.4f}, "
        f"calibrated_ece={calibrated_ece:.4f}, "
        f"threshold={selected_threshold['threshold']:.4f}"
    )
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
