from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from quorabust.model import predict_proba_duplicate
from quorabust.persist import load_classifier

_REQUIRED_COLUMNS = {"question1", "question2", "is_duplicate"}
_METADATA_KEYS = [
    "feature_backend",
    "feature_schema",
    "n_train",
    "n_eval",
    "seed",
    "quorabust_version",
    "git_revision",
    "csv_sha256",
]


def _package_version() -> str:
    try:
        return version("Quorabust")
    except PackageNotFoundError:
        return "0.0.0"


def _load_eval_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(df.columns)}")
    return df


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int | str):
        return str(value)
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_format_value(v) for v in row) + " |")
    return "\n".join(out)


def _metadata_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: meta[k] for k in _METADATA_KEYS if k in meta}


def _persisted_metrics_from_meta(meta: dict[str, Any]) -> dict[str, float]:
    return {
        k.removeprefix("eval_"): float(v)
        for k, v in sorted(meta.items())
        if k.startswith("eval_") and isinstance(v, int | float)
    }


def _parse_thresholds(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            threshold = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid threshold: {value}") from exc
        if not 0.0 < threshold < 1.0:
            raise ValueError("thresholds must be between 0 and 1")
        out.append(threshold)
    if not out:
        raise ValueError("at least one threshold is required")
    return out


def _metrics_at_threshold(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def calibration_summary(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Summarize whether predicted probabilities match observed positive rates."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    bins: list[dict[str, Any]] = []
    total = len(y)
    expected_calibration_error = 0.0
    for idx in range(n_bins):
        lower = idx / n_bins
        upper = (idx + 1) / n_bins
        if idx == n_bins - 1:
            mask = (proba >= lower) & (proba <= upper)
        else:
            mask = (proba >= lower) & (proba < upper)
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_predicted = float(np.mean(proba[mask]))
        observed_rate = float(np.mean(y[mask]))
        absolute_error = abs(mean_predicted - observed_rate)
        expected_calibration_error += (count / total) * absolute_error
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_predicted_probability": mean_predicted,
                "observed_positive_rate": observed_rate,
                "absolute_error": float(absolute_error),
            }
        )

    return {
        "n_bins": int(n_bins),
        "brier_score": float(brier_score_loss(y, proba)),
        "expected_calibration_error": float(expected_calibration_error),
        "mean_predicted_probability": float(np.mean(proba)),
        "mean_observed_rate": float(np.mean(y)),
        "bins": bins,
    }


def _holdout_proba(
    builder: Any,
    clf: Any,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    y = df["is_duplicate"].astype(int).to_numpy()
    proba = predict_proba_duplicate(
        builder,
        clf,
        df["question1"].astype(str).tolist(),
        df["question2"].astype(str).tolist(),
    )[:, 1]
    return y, proba


def evaluate_holdout(
    builder: Any,
    clf: Any,
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate a loaded artifact against a labeled Quora-style dataframe."""
    y, proba = _holdout_proba(builder, clf, df)
    threshold_metrics = _metrics_at_threshold(y, proba, threshold)
    metrics: dict[str, Any] = {
        "n": int(len(df)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "positive_rate": float(np.mean(y)),
        "calibration": calibration_summary(y, proba, n_bins=calibration_bins),
        **threshold_metrics,
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    return metrics


def threshold_sweep(
    builder: Any,
    clf: Any,
    df: pd.DataFrame,
    *,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Evaluate precision/recall tradeoffs across decision thresholds."""
    y, proba = _holdout_proba(builder, clf, df)
    return [_metrics_at_threshold(y, proba, threshold) for threshold in thresholds]


def render_model_card(
    *,
    artifact: str,
    meta: dict[str, Any],
    holdout_metrics: dict[str, Any] | None = None,
    sweep_metrics: list[dict[str, Any]] | None = None,
) -> str:
    """Render artifact metadata and optional holdout metrics as Markdown."""
    artifact_rows = [
        ["artifact", artifact],
        ["generated_by", f"Quorabust {_package_version()}"],
    ]

    metadata_rows = [[k, v] for k, v in _metadata_from_meta(meta).items()]
    persisted_metric_rows = [[k, v] for k, v in _persisted_metrics_from_meta(meta).items()]

    parts = [
        "# Quorabust Model Card",
        "",
        "## Artifact",
        "",
        _markdown_table(["field", "value"], artifact_rows),
        "",
        "## Intended Use",
        "",
        (
            "Scores pairs of short natural-language questions and returns the probability "
            "that the pair is semantically duplicate. Use it as a ranking or moderation "
            "signal, not as a sole automated decision system."
        ),
        "",
        "## Training Metadata",
        "",
        (
            _markdown_table(["field", "value"], metadata_rows)
            if metadata_rows
            else "_No metadata found._"
        ),
        "",
        "## Persisted Evaluation",
        "",
        (
            _markdown_table(["metric", "value"], persisted_metric_rows)
            if persisted_metric_rows
            else "_No persisted evaluation metrics found._"
        ),
    ]

    if holdout_metrics is not None:
        calibration = holdout_metrics.get("calibration")
        metric_keys = [
            "n",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "log_loss",
            "roc_auc",
            "positive_rate",
            "predicted_positive_rate",
        ]
        metric_rows = [[k, holdout_metrics[k]] for k in metric_keys if k in holdout_metrics]
        confusion_rows = [
            ["actual 0", holdout_metrics["tn"], holdout_metrics["fp"]],
            ["actual 1", holdout_metrics["fn"], holdout_metrics["tp"]],
        ]
        parts.extend(
            [
                "",
                "## Holdout Evaluation",
                "",
                _markdown_table(["metric", "value"], metric_rows),
                "",
                "## Confusion Matrix",
                "",
                _markdown_table(["", "predicted 0", "predicted 1"], confusion_rows),
            ]
        )
        if isinstance(calibration, dict):
            calibration_rows = [
                ["n_bins", calibration["n_bins"]],
                ["brier_score", calibration["brier_score"]],
                ["expected_calibration_error", calibration["expected_calibration_error"]],
                ["mean_predicted_probability", calibration["mean_predicted_probability"]],
                ["mean_observed_rate", calibration["mean_observed_rate"]],
            ]
            bin_rows = [
                [
                    f"{row['lower']:.2f}-{row['upper']:.2f}",
                    row["count"],
                    row["mean_predicted_probability"],
                    row["observed_positive_rate"],
                    row["absolute_error"],
                ]
                for row in calibration.get("bins", [])
            ]
            parts.extend(
                [
                    "",
                    "## Calibration Summary",
                    "",
                    _markdown_table(["metric", "value"], calibration_rows),
                    "",
                    "## Calibration Bins",
                    "",
                    _markdown_table(
                        [
                            "probability_bin",
                            "count",
                            "mean_predicted_probability",
                            "observed_positive_rate",
                            "absolute_error",
                        ],
                        bin_rows,
                    ),
                ]
            )
        if sweep_metrics:
            sweep_rows = [
                [
                    row["threshold"],
                    row["precision"],
                    row["recall"],
                    row["f1"],
                    row["accuracy"],
                    row["predicted_positive_rate"],
                ]
                for row in sweep_metrics
            ]
            parts.extend(
                [
                    "",
                    "## Threshold Sweep",
                    "",
                    _markdown_table(
                        [
                            "threshold",
                            "precision",
                            "recall",
                            "f1",
                            "accuracy",
                            "predicted_positive_rate",
                        ],
                        sweep_rows,
                    ),
                ]
            )

    parts.extend(
        [
            "",
            "## Serving Contract",
            "",
            (
                "`POST /predict` accepts `question1` and `question2` arrays of equal length "
                "and returns `proba_duplicate`, `is_duplicate`, and `decision_threshold` "
                "in the same order. `GET /metrics` exposes Prometheus counters and "
                "latency histograms."
            ),
            "",
            "## Caveats",
            "",
            (
                "Performance depends on the training data distribution and threshold. "
                "Re-run this card on a current holdout set before comparing artifacts."
            ),
            "",
        ]
    )
    return "\n".join(parts)


def build_report_payload(
    *,
    artifact: str,
    meta: dict[str, Any],
    holdout_metrics: dict[str, Any] | None = None,
    sweep_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable report payload for CI and model comparisons."""
    payload: dict[str, Any] = {
        "artifact": artifact,
        "generated_by": f"Quorabust {_package_version()}",
        "intended_use": (
            "Scores pairs of short natural-language questions and returns the probability "
            "that the pair is semantically duplicate."
        ),
        "training_metadata": _metadata_from_meta(meta),
        "persisted_evaluation": _persisted_metrics_from_meta(meta),
        "serving_contract": {
            "predict": "POST /predict",
            "metrics": "GET /metrics",
            "input": {"question1": "list[str]", "question2": "list[str]"},
            "output": {
                "proba_duplicate": "list[float]",
                "is_duplicate": "list[bool]",
                "decision_threshold": "float",
            },
        },
        "caveats": [
            "Performance depends on the training data distribution and threshold.",
            "Re-run on a current holdout set before comparing artifacts.",
        ],
    }
    if holdout_metrics is not None:
        payload["holdout_evaluation"] = {
            k: v
            for k, v in holdout_metrics.items()
            if k not in {"tn", "fp", "fn", "tp", "calibration"}
        }
        calibration = holdout_metrics.get("calibration")
        if calibration is not None:
            payload["calibration"] = calibration
        payload["confusion_matrix"] = {
            "labels": ["not_duplicate", "duplicate"],
            "actual_0": {
                "predicted_0": holdout_metrics["tn"],
                "predicted_1": holdout_metrics["fp"],
            },
            "actual_1": {
                "predicted_0": holdout_metrics["fn"],
                "predicted_1": holdout_metrics["tp"],
            },
        }
        if sweep_metrics:
            payload["threshold_sweep"] = sweep_metrics
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a model report for a Quorabust artifact.",
    )
    parser.add_argument("--model", type=Path, required=True, help="Saved .pkl artifact")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Optional labeled CSV with question1, question2, is_duplicate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for the optional confusion matrix",
    )
    parser.add_argument(
        "--thresholds",
        default="0.3,0.5,0.7",
        help="Comma-separated thresholds for the optional holdout sweep",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of probability bins for the optional calibration report",
    )
    parser.add_argument(
        "--artifact-label",
        default=None,
        help="Public artifact label to print instead of the local model path",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Report format to print or write",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write report here")
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(f"File not found: {args.model}", file=sys.stderr)
        return 1
    if not 0.0 < args.threshold < 1.0:
        print("--threshold must be between 0 and 1", file=sys.stderr)
        return 1
    if args.calibration_bins < 1:
        print("--calibration-bins must be at least 1", file=sys.stderr)
        return 1
    try:
        thresholds = _parse_thresholds(args.thresholds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    builder, clf, meta = load_classifier(args.model)
    holdout_metrics = None
    sweep_metrics = None
    if args.eval_csv is not None:
        if not args.eval_csv.is_file():
            print(f"File not found: {args.eval_csv}", file=sys.stderr)
            return 1
        try:
            eval_df = _load_eval_csv(args.eval_csv)
            holdout_metrics = evaluate_holdout(
                builder,
                clf,
                eval_df,
                threshold=args.threshold,
                calibration_bins=args.calibration_bins,
            )
            sweep_metrics = threshold_sweep(
                builder,
                clf,
                eval_df,
                thresholds=thresholds,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    artifact = args.artifact_label or str(args.model.resolve())
    if args.format == "json":
        report = json.dumps(
            build_report_payload(
                artifact=artifact,
                meta=meta,
                holdout_metrics=holdout_metrics,
                sweep_metrics=sweep_metrics,
            ),
            indent=2,
            sort_keys=True,
        )
    else:
        report = render_model_card(
            artifact=artifact,
            meta=meta,
            holdout_metrics=holdout_metrics,
            sweep_metrics=sweep_metrics,
        )
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
