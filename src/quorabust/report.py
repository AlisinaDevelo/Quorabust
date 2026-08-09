from __future__ import annotations

import argparse
import json
import math
import platform
import shlex
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from quorabust.lineage import git_revision, sha256_file
from quorabust.model import predict_proba_duplicate
from quorabust.persist import load_classifier
from quorabust.slice_manifest import load_slice_manifest

_REQUIRED_COLUMNS = {"question1", "question2", "is_duplicate"}
_DEFAULT_MAX_SLICES = 50
_SLICE_HEADERS = [
    "slice_column",
    "slice",
    "n",
    "positive_rate",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "log_loss",
    "brier_score",
    "expected_calibration_error",
    "positive_rate_ci95",
    "accuracy_ci95",
    "precision_ci95",
    "recall_ci95",
]
_SLICE_UNCERTAINTY_NOTE = (
    "Wilson 95% confidence intervals cover positive rate, accuracy, precision, and recall. "
    "Log loss, F1, ROC-AUC, Brier score, and ECE remain point estimates."
)
_METADATA_KEYS = [
    "feature_backend",
    "feature_schema",
    "embedding_model",
    "embedding_model_revision",
    "cross_encoder_model",
    "cross_encoder_model_revision",
    "cross_encoder_batch_size",
    "n_train",
    "n_eval",
    "eval_csv_sha256",
    "eval_fraction",
    "eval_split_source",
    "split_strategy",
    "question_id_columns",
    "max_rows",
    "require_question_ids",
    "require_question_text",
    "seed",
    "threshold_candidates",
    "threshold_metric",
    "quorabust_version",
    "git_revision",
    "csv_sha256",
    "decision_threshold",
    "decision_threshold_source",
    "decision_threshold_metric",
    "decision_threshold_costs",
    "calibration_method",
    "calibration_csv_sha256",
    "threshold_csv_sha256",
    "threshold_reuses_evaluation_role",
    "calibration_git_revision",
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


def _wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> list[float] | None:
    """Return a Wilson interval for a binomial rate or None if undefined."""
    if trials < 1 or successes < 0 or successes > trials:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    proportion = successes / trials
    denominator = 1 + (z * z / trials)
    center = (proportion + (z * z / (2 * trials))) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1 - proportion) / trials)
            + (z * z / (4 * trials * trials))
        )
        / denominator
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    if successes == 0:
        lower = 0.0
    if successes == trials:
        upper = 1.0
    return [lower, upper]


def _slice_uncertainty(metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize uncertainty for rate metrics derived from a confusion matrix."""
    n = int(metrics["n"])
    tn = int(metrics["tn"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    tp = int(metrics["tp"])
    return {
        "confidence_level": 0.95,
        "method": "wilson_binomial_interval",
        "confidence_intervals": {
            "positive_rate": _wilson_interval(tp + fn, n),
            "accuracy": _wilson_interval(tn + tp, n),
            "precision": _wilson_interval(tp, tp + fp),
            "recall": _wilson_interval(tp, tp + fn),
        },
        "caveat": _SLICE_UNCERTAINTY_NOTE,
    }


def _slice_table_rows(
    evaluation_slices: dict[str, list[dict[str, Any]]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for column in sorted(evaluation_slices):
        for metrics in evaluation_slices[column]:
            calibration = metrics.get("calibration", {})
            uncertainty = metrics.get("uncertainty", {})
            intervals = uncertainty.get("confidence_intervals", {})
            rows.append(
                [
                    column,
                    metrics["value"],
                    metrics["n"],
                    metrics["positive_rate"],
                    metrics["threshold"],
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["log_loss"],
                    calibration.get("brier_score"),
                    calibration.get("expected_calibration_error"),
                    intervals.get("positive_rate"),
                    intervals.get("accuracy"),
                    intervals.get("precision"),
                    intervals.get("recall"),
                ]
            )
    return rows


def _metadata_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: meta[k] for k in _METADATA_KEYS if k in meta}


def _persisted_metrics_from_meta(meta: dict[str, Any]) -> dict[str, float]:
    return {
        k.removeprefix("eval_"): float(v)
        for k, v in sorted(meta.items())
        if k in {"eval_accuracy", "eval_log_loss", "eval_roc_auc"}
        and isinstance(v, int | float)
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


def _report_command(argv: list[str] | None) -> str:
    arguments = sys.argv[1:] if argv is None else argv
    return shlex.join(["quorabust-report", *(str(value) for value in arguments)])


def build_evaluation_manifest(
    *,
    artifact_path: Path,
    artifact_label: str,
    eval_path: Path,
    eval_df: pd.DataFrame,
    meta: dict[str, Any],
    threshold: float,
    thresholds: list[float],
    calibration_bins: int,
    command: str,
    slice_columns: list[str] | None = None,
    slice_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an auditable, path-light record for a holdout evaluation run."""
    labels = eval_df["is_duplicate"].astype(int)
    lineage_keys = [
        "git_revision",
        "quorabust_version",
        "csv_sha256",
        "feature_backend",
        "embedding_model",
        "embedding_model_revision",
        "cross_encoder_model",
        "cross_encoder_model_revision",
        "cross_encoder_batch_size",
        "n_train",
        "n_eval",
        "eval_csv_sha256",
        "eval_fraction",
        "eval_split_source",
        "split_strategy",
        "question_id_columns",
        "max_rows",
        "require_question_ids",
        "require_question_text",
        "seed",
        "threshold_candidates",
        "threshold_metric",
        "training_command",
        "decision_threshold",
        "decision_threshold_source",
        "decision_threshold_metric",
        "decision_threshold_costs",
        "calibration_method",
        "calibration_csv_sha256",
        "threshold_csv_sha256",
        "threshold_reuses_evaluation_role",
        "calibration_git_revision",
    ]
    training_lineage = {key: meta[key] for key in lineage_keys if key in meta}
    evaluation_policy: dict[str, Any] = {
        "threshold": float(threshold),
        "thresholds": [float(value) for value in thresholds],
        "calibration_bins": int(calibration_bins),
    }
    normalized_slice_columns = [column.strip() for column in slice_columns or []]
    if normalized_slice_columns:
        evaluation_policy["slice_columns"] = normalized_slice_columns

    manifest = {
        "schema_version": 1,
        "artifact": {
            "label": artifact_label,
            "sha256": sha256_file(artifact_path),
        },
        "evaluation_dataset": {
            "sha256": sha256_file(eval_path),
            "rows": int(len(eval_df)),
            "columns": list(eval_df.columns),
            "positive_count": int(labels.sum()),
            "positive_rate": float(labels.mean()),
        },
        "evaluation_policy": evaluation_policy,
        "training_lineage": training_lineage,
        "runtime": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "report_git_revision": git_revision(),
        },
        "command": command,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if slice_provenance is not None:
        manifest["slice_provenance"] = slice_provenance
    return manifest


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
        metrics["pr_auc"] = float(average_precision_score(y, proba))
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


def evaluate_slices(
    builder: Any,
    clf: Any,
    df: pd.DataFrame,
    *,
    slice_columns: list[str],
    threshold: float,
    calibration_bins: int,
    max_slices: int = _DEFAULT_MAX_SLICES,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate explicitly supplied, bounded dataset slices."""
    if max_slices < 1:
        raise ValueError("max_slices must be at least 1")
    normalized_columns = [column.strip() for column in slice_columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("--slice-column values must be unique")

    output: dict[str, list[dict[str, Any]]] = {}
    for normalized_column in normalized_columns:
        if not normalized_column:
            raise ValueError("--slice-column cannot be empty")
        if normalized_column not in df.columns:
            raise ValueError(
                f"Unknown slice column {normalized_column!r}; found {list(df.columns)}"
            )
        values = df[normalized_column]
        if values.isna().any():
            raise ValueError(f"slice column {normalized_column!r} contains missing labels")
        labels = values.astype(str).str.strip()
        if (labels == "").any():
            raise ValueError(f"slice column {normalized_column!r} contains blank labels")
        unique_labels = sorted(set(labels.tolist()))
        if len(unique_labels) > max_slices:
            raise ValueError(
                f"slice column {normalized_column!r} has {len(unique_labels)} labels; "
                f"maximum is {max_slices}"
            )

        rows: list[dict[str, Any]] = []
        for label in unique_labels:
            subset = df.loc[labels == label]
            metrics = evaluate_holdout(
                builder,
                clf,
                subset,
                threshold=threshold,
                calibration_bins=calibration_bins,
            )
            metrics["uncertainty"] = _slice_uncertainty(metrics)
            rows.append({"value": label, **metrics})
        output[normalized_column] = rows
    return output


def _slice_observed_row_counts(
    evaluation_slices: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {
        column: {
            "rows": sum(int(row["n"]) for row in rows),
            "labels": {
                str(row["value"]): int(row["n"])
                for row in rows
            },
        }
        for column, rows in sorted(evaluation_slices.items())
    }


def render_model_card(
    *,
    artifact: str,
    meta: dict[str, Any],
    holdout_metrics: dict[str, Any] | None = None,
    sweep_metrics: list[dict[str, Any]] | None = None,
    evaluation_manifest: dict[str, Any] | None = None,
    evaluation_slices: dict[str, list[dict[str, Any]]] | None = None,
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
            "pr_auc",
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

    calibration_comparison = meta.get("calibration_metrics")
    if isinstance(calibration_comparison, dict):
        comparison_rows = []
        for label in ("raw", "calibrated"):
            metrics = calibration_comparison.get(label)
            if isinstance(metrics, dict):
                comparison_rows.append(
                    [
                        label,
                        metrics.get("n"),
                        metrics.get("brier_score"),
                        metrics.get("expected_calibration_error"),
                        metrics.get("mean_predicted_probability"),
                        metrics.get("mean_observed_rate"),
                    ]
                )
        if comparison_rows:
            parts.extend(
                [
                    "",
                    "## Calibration Fit Comparison",
                    "",
                    _markdown_table(
                        [
                            "probability_source",
                            "n",
                            "brier_score",
                            "expected_calibration_error",
                            "mean_predicted_probability",
                            "mean_observed_rate",
                        ],
                        comparison_rows,
                    ),
                ]
            )

    if evaluation_slices:
        parts.extend(
            [
                "",
                "## Evaluation Slices",
                "",
                _SLICE_UNCERTAINTY_NOTE,
                "",
                _markdown_table(_SLICE_HEADERS, _slice_table_rows(evaluation_slices)),
            ]
        )

    if evaluation_manifest is not None:
        parts.extend(
            [
                "",
                "## Reproducibility",
                "",
                "```json",
                json.dumps(evaluation_manifest, indent=2, sort_keys=True),
                "```",
            ]
        )

    parts.extend(
        [
            "",
            "## Serving Contract",
            "",
            (
                "`POST /predict` accepts `question1` and `question2` arrays of equal length "
                "and returns raw, optional calibrated, and effective duplicate probabilities, "
                "alongside `is_duplicate`, `decision_threshold`, and their source fields. "
                "`GET /metrics` exposes Prometheus counters and latency histograms."
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


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row.get("f1", 0.0), reverse=True)


def render_comparison_report(
    rows: list[dict[str, Any]],
    *,
    evaluation_manifest: dict[str, Any] | None = None,
    evaluation_slices: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Render comparable artifact metrics as a Markdown table."""
    metric_rows = [
        [
            row["artifact"],
            row.get("feature_backend", ""),
            row.get("threshold", ""),
            row.get("f1", ""),
            row.get("precision", ""),
            row.get("recall", ""),
            row.get("accuracy", ""),
            row.get("roc_auc", ""),
            row.get("pr_auc", ""),
            row.get("log_loss", ""),
        ]
        for row in _comparison_rows(rows)
    ]
    parts = [
            "# Quorabust Model Comparison",
            "",
            "## Backend Comparison",
            "",
            _markdown_table(
                [
                    "artifact",
                    "feature_backend",
                    "threshold",
                    "f1",
                    "precision",
                    "recall",
                    "accuracy",
                    "roc_auc",
                    "pr_auc",
                    "log_loss",
                ],
                metric_rows,
            ),
            "",
            "## Caveats",
            "",
            (
                "Compare rows only when every artifact was evaluated against the same "
                "holdout CSV, threshold policy, and metric code."
            ),
            "",
        ]
    if evaluation_manifest is not None:
        parts.extend(
            [
                "## Reproducibility",
                "",
                "```json",
                json.dumps(evaluation_manifest, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if evaluation_slices:
        parts.extend(
            [
                "## Reference Model Evaluation Slices",
                "",
                _SLICE_UNCERTAINTY_NOTE,
                "",
                _markdown_table(_SLICE_HEADERS, _slice_table_rows(evaluation_slices)),
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
    evaluation_manifest: dict[str, Any] | None = None,
    evaluation_slices: dict[str, list[dict[str, Any]]] | None = None,
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
                "raw_proba_duplicate": "list[float]",
                "calibrated_proba_duplicate": "list[float] | null",
                "proba_duplicate": "list[float]",
                "is_duplicate": "list[bool]",
                "decision_threshold": "float",
                "decision_threshold_source": "string",
                "probability_source": "raw|calibrated",
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
    calibration_comparison = meta.get("calibration_metrics")
    if isinstance(calibration_comparison, dict):
        payload["calibration_comparison"] = calibration_comparison
    if evaluation_slices:
        payload["evaluation_slices"] = evaluation_slices
    if evaluation_manifest is not None:
        payload["evaluation_manifest"] = evaluation_manifest
    return payload


def _parse_compare_model(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--compare-model must use label=path")
    label, value = raw.split("=", 1)
    label = label.strip()
    path = Path(value.strip())
    if not label:
        raise ValueError("--compare-model label cannot be empty")
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    return label, path


def _comparison_metrics(
    label: str,
    path: Path,
    eval_df: pd.DataFrame,
    *,
    threshold: float,
    calibration_bins: int,
) -> dict[str, Any]:
    builder, clf, meta = load_classifier(path)
    metrics = evaluate_holdout(
        builder,
        clf,
        eval_df,
        threshold=threshold,
        calibration_bins=calibration_bins,
    )
    return {
        "artifact": label,
        "feature_backend": meta.get("feature_backend", ""),
        "threshold": metrics.get("threshold"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "log_loss": metrics.get("log_loss"),
        "positive_rate": metrics.get("positive_rate"),
        "predicted_positive_rate": metrics.get("predicted_positive_rate"),
    }


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
        "--slice-column",
        action="append",
        default=[],
        help="Optional evaluation slice column; repeat for language/domain/category slices",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=_DEFAULT_MAX_SLICES,
        help="Maximum distinct labels allowed per requested slice column",
    )
    parser.add_argument(
        "--slice-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON provenance sidecar bound to the evaluated CSV; requires "
            "--slice-column"
        ),
    )
    parser.add_argument(
        "--artifact-label",
        default=None,
        help="Public artifact label to print instead of the local model path",
    )
    parser.add_argument(
        "--compare-model",
        action="append",
        default=[],
        help="Compare another artifact on the same eval CSV, formatted as label=path",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Report format to print or write",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write report here")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Write the holdout reproducibility manifest as JSON",
    )
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
    if args.max_slices < 1:
        print("--max-slices must be at least 1", file=sys.stderr)
        return 1
    try:
        thresholds = _parse_thresholds(args.thresholds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.manifest_out is not None and args.eval_csv is None:
        print("--manifest-out requires --eval-csv", file=sys.stderr)
        return 1
    if args.slice_column and args.eval_csv is None:
        print("--slice-column requires --eval-csv", file=sys.stderr)
        return 1
    if args.slice_manifest is not None and args.eval_csv is None:
        print("--slice-manifest requires --eval-csv", file=sys.stderr)
        return 1
    if args.slice_manifest is not None and not args.slice_column:
        print("--slice-manifest requires --slice-column", file=sys.stderr)
        return 1

    builder, clf, meta = load_classifier(args.model)
    artifact = args.artifact_label or str(args.model.resolve())
    eval_df = None
    holdout_metrics = None
    sweep_metrics = None
    comparison_metrics = None
    evaluation_manifest = None
    evaluation_slices = None
    slice_provenance = None
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
            if args.slice_column:
                evaluation_slices = evaluate_slices(
                    builder,
                    clf,
                    eval_df,
                    slice_columns=args.slice_column,
                    threshold=args.threshold,
                    calibration_bins=args.calibration_bins,
                    max_slices=args.max_slices,
                )
                if args.slice_manifest is not None:
                    slice_provenance = {
                        "manifest": load_slice_manifest(
                            args.slice_manifest,
                            eval_path=args.eval_csv,
                            eval_rows=len(eval_df),
                            slice_columns=args.slice_column,
                        ),
                        "observed_row_counts": _slice_observed_row_counts(evaluation_slices),
                    }
            if args.compare_model:
                comparison_metrics = [
                    _comparison_metrics(
                        label,
                        path,
                        eval_df,
                        threshold=args.threshold,
                        calibration_bins=args.calibration_bins,
                    )
                    for label, path in (_parse_compare_model(raw) for raw in args.compare_model)
                ]
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    elif args.compare_model:
        print("--compare-model requires --eval-csv", file=sys.stderr)
        return 1

    if eval_df is not None:
        evaluation_manifest = build_evaluation_manifest(
            artifact_path=args.model,
            artifact_label=artifact,
            eval_path=args.eval_csv,
            eval_df=eval_df,
            meta=meta,
            threshold=args.threshold,
            thresholds=thresholds,
            calibration_bins=args.calibration_bins,
            command=_report_command(argv),
            slice_columns=args.slice_column,
            slice_provenance=slice_provenance,
        )

    if args.format == "json":
        payload = build_report_payload(
            artifact=artifact,
            meta=meta,
            holdout_metrics=holdout_metrics,
            sweep_metrics=sweep_metrics,
            evaluation_manifest=evaluation_manifest,
            evaluation_slices=evaluation_slices,
        )
        if comparison_metrics is not None:
            payload["comparison"] = _comparison_rows(comparison_metrics)
        report = json.dumps(payload, indent=2, sort_keys=True)
    elif comparison_metrics is not None:
        report = render_comparison_report(
            comparison_metrics,
            evaluation_manifest=evaluation_manifest,
            evaluation_slices=evaluation_slices,
        )
    else:
        report = render_model_card(
            artifact=artifact,
            meta=meta,
            holdout_metrics=holdout_metrics,
            sweep_metrics=sweep_metrics,
            evaluation_manifest=evaluation_manifest,
            evaluation_slices=evaluation_slices,
        )
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.out.resolve()}")
    if args.manifest_out is not None and evaluation_manifest is not None:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.manifest_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
