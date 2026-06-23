from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score

from quorabust.model import predict_proba_duplicate
from quorabust.persist import load_classifier

_REQUIRED_COLUMNS = {"question1", "question2", "is_duplicate"}


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


def evaluate_holdout(
    builder: Any,
    clf: Any,
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a loaded artifact against a labeled Quora-style dataframe."""
    y = df["is_duplicate"].astype(int).to_numpy()
    proba = predict_proba_duplicate(
        builder,
        clf,
        df["question1"].astype(str).tolist(),
        df["question2"].astype(str).tolist(),
    )[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "n": int(len(df)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(pred)),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    return metrics


def render_model_card(
    *,
    artifact: str,
    meta: dict[str, Any],
    holdout_metrics: dict[str, Any] | None = None,
) -> str:
    """Render artifact metadata and optional holdout metrics as Markdown."""
    artifact_rows = [
        ["artifact", artifact],
        ["generated_by", f"Quorabust {_package_version()}"],
    ]

    metadata_keys = [
        "feature_backend",
        "feature_schema",
        "n_train",
        "n_eval",
        "seed",
        "quorabust_version",
        "git_revision",
        "csv_sha256",
    ]
    metadata_rows = [[k, meta[k]] for k in metadata_keys if k in meta]
    persisted_metric_rows = [
        [k.removeprefix("eval_"), meta[k]]
        for k in sorted(meta)
        if k.startswith("eval_") and isinstance(meta[k], int | float)
    ]

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
        metric_keys = [
            "n",
            "threshold",
            "accuracy",
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

    parts.extend(
        [
            "",
            "## Serving Contract",
            "",
            (
                "`POST /predict` accepts `question1` and `question2` arrays of equal length "
                "and returns `proba_duplicate` in the same order. `GET /metrics` exposes "
                "Prometheus counters and latency histograms."
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown model card for a Quorabust artifact.",
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
    parser.add_argument("--out", type=Path, default=None, help="Write Markdown here")
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(f"File not found: {args.model}", file=sys.stderr)
        return 1
    if not 0.0 < args.threshold < 1.0:
        print("--threshold must be between 0 and 1", file=sys.stderr)
        return 1

    builder, clf, meta = load_classifier(args.model)
    holdout_metrics = None
    if args.eval_csv is not None:
        if not args.eval_csv.is_file():
            print(f"File not found: {args.eval_csv}", file=sys.stderr)
            return 1
        try:
            holdout_metrics = evaluate_holdout(
                builder,
                clf,
                _load_eval_csv(args.eval_csv),
                threshold=args.threshold,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    card = render_model_card(
        artifact=str(args.model.resolve()),
        meta=meta,
        holdout_metrics=holdout_metrics,
    )
    if args.out is None:
        print(card)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(card, encoding="utf-8")
        print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
