from __future__ import annotations

import argparse
import shlex
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.data_audit import OPTIONAL_QUESTION_ID_COLUMNS
from quorabust.drift import feature_means_from_matrix
from quorabust.lineage import git_revision, sha256_file
from quorabust.model import (
    eval_classification_metrics,
    predict_proba_duplicate,
    select_decision_threshold,
    train_duplicate_classifier,
    validate_threshold_costs,
)
from quorabust.persist import save_classifier, save_metadata_sidecar
from quorabust.registry import append_model_record
from quorabust.split import split_train_eval


def _package_version() -> str:
    try:
        return version("Quorabust")
    except PackageNotFoundError:
        return "0.0.0"


def _load_quora_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {"question1", "question2", "is_duplicate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(df.columns)}")
    return df


def _parse_thresholds(raw: str) -> list[float]:
    thresholds: list[float] = []
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
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def _missing_or_incomplete_question_ids(df: pd.DataFrame) -> list[str]:
    """Return qid columns that are absent or contain blank values."""
    return [
        column
        for column in OPTIONAL_QUESTION_ID_COLUMNS
        if column not in df.columns
        or df[column].isna().any()
        or df[column].astype("string").str.strip().eq("").any()
    ]


def _question_ids_are_disjoint(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Check that complete question-component IDs do not cross an explicit holdout."""
    if any(_missing_or_incomplete_question_ids(frame) for frame in (left, right)):
        return False
    left_ids = set(
        left[list(OPTIONAL_QUESTION_ID_COLUMNS)].astype("string").to_numpy().reshape(-1)
    )
    right_ids = set(
        right[list(OPTIONAL_QUESTION_ID_COLUMNS)].astype("string").to_numpy().reshape(-1)
    )
    return left_ids.isdisjoint(right_ids)


def _training_command(argv: list[str] | None) -> str:
    """Return a copy-pasteable command that records the training configuration."""
    arguments = sys.argv[1:] if argv is None else argv
    return shlex.join(["quorabust-train", *(str(value) for value in arguments)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train Quorabust duplicate classifier from a Quora-style CSV.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to train.csv (Kaggle Quora Question Pairs)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output .pkl (builder + XGBoost)")
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows after shuffle (subset / debug)",
    )
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Holdout fraction for early stopping and log loss (use 0 for no holdout)",
    )
    p.add_argument(
        "--eval-out",
        type=Path,
        default=None,
        help="Write the exact training holdout CSV used for evaluation and threshold selection",
    )
    p.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help=(
            "Use an independent labeled CSV as the evaluation/threshold role instead of "
            "splitting the training CSV"
        ),
    )
    p.add_argument(
        "--require-question-ids",
        action="store_true",
        help="Fail unless qid1 and qid2 are complete; recommended for benchmark runs",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature-backend",
        choices=["tfidf", "embedding", "cross-encoder"],
        default="tfidf",
        help=(
            "tfidf (default), sentence-transformer embeddings, or cross-encoder pair scores "
            "(requires nlp extra)"
        ),
    )
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="When --feature-backend=embedding, SentenceTransformer model id",
    )
    p.add_argument(
        "--cross-encoder-model",
        default="cross-encoder/quora-distilroberta-base",
        help="When --feature-backend=cross-encoder, CrossEncoder model id",
    )
    p.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="If set, append a JSONL record under this directory after training",
    )
    p.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="If set, write artifact metadata JSON here without requiring pickle loading",
    )
    p.add_argument(
        "--thresholds",
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated candidate thresholds for holdout-based decision selection",
    )
    p.add_argument(
        "--threshold-metric",
        choices=["accuracy", "precision", "recall", "f1", "expected_cost"],
        default="f1",
        help="Metric to optimize when selecting a decision threshold from the holdout",
    )
    p.add_argument(
        "--false-positive-cost",
        type=float,
        default=1.0,
        help="Cost assigned to a false-positive decision",
    )
    p.add_argument(
        "--false-negative-cost",
        type=float,
        default=1.0,
        help="Cost assigned to a false-negative decision",
    )
    args = p.parse_args(argv)

    if not args.csv.is_file():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 1
    try:
        threshold_candidates = _parse_thresholds(args.thresholds)
        threshold_costs = validate_threshold_costs(
            args.false_positive_cost,
            args.false_negative_cost,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    try:
        df = _load_quora_csv(args.csv)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if args.require_question_ids:
        missing_or_incomplete = _missing_or_incomplete_question_ids(df)
        if missing_or_incomplete:
            print(
                "--require-question-ids needs complete qid1/qid2 columns; "
                f"missing or incomplete: {', '.join(missing_or_incomplete)}",
                file=sys.stderr,
            )
            return 1

    eval_csv_sha256: str | None = None
    eval_split_source = "generated_from_source"
    if args.eval_csv is not None:
        if args.eval_out is not None:
            print("--eval-out cannot be combined with --eval-csv", file=sys.stderr)
            return 1
        if not args.eval_csv.is_file():
            print(f"File not found: {args.eval_csv}", file=sys.stderr)
            return 1
        if args.eval_csv.resolve() == args.csv.resolve():
            print("--eval-csv must be different from --csv", file=sys.stderr)
            return 1
        try:
            eval_df = _load_quora_csv(args.eval_csv)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.require_question_ids:
            eval_missing_ids = _missing_or_incomplete_question_ids(eval_df)
            if eval_missing_ids:
                print(
                    "--require-question-ids needs complete qid1/qid2 columns in --eval-csv; "
                    f"missing or incomplete: {', '.join(eval_missing_ids)}",
                    file=sys.stderr,
                )
                return 1
        if _missing_or_incomplete_question_ids(df) or _missing_or_incomplete_question_ids(
            eval_df
        ):
            split_strategy = "explicit_holdout"
        elif not _question_ids_are_disjoint(df, eval_df):
            print(
                "--csv and --eval-csv question IDs overlap; provide disjoint components",
                file=sys.stderr,
            )
            return 1
        else:
            split_strategy = "question_component_holdout"
        train_df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        if args.max_rows is not None:
            train_df = train_df.head(args.max_rows).copy()
        eval_csv_sha256 = sha256_file(args.eval_csv)
        eval_split_source = "explicit_csv"
    else:
        try:
            train_df, eval_df, split_strategy = split_train_eval(
                df,
                eval_fraction=args.eval_fraction,
                seed=args.seed,
                max_rows=args.max_rows,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.eval_out is not None:
        if eval_df is None:
            print(
                "--eval-out requires a non-empty holdout; increase --eval-fraction "
                "and provide at least 20 rows",
                file=sys.stderr,
            )
            return 1
        if args.eval_out.resolve() == args.csv.resolve():
            print("--eval-out must be different from --csv", file=sys.stderr)
            return 1
        try:
            args.eval_out.parent.mkdir(parents=True, exist_ok=True)
            eval_df.to_csv(args.eval_out, index=False)
            eval_csv_sha256 = sha256_file(args.eval_out)
        except OSError as exc:
            print(f"Unable to write holdout CSV {args.eval_out}: {exc}", file=sys.stderr)
            return 1

    feature_builder: Any | None = None
    if args.feature_backend == "embedding":
        from quorabust.embedding_features import PairEmbeddingBuilder

        feature_builder = PairEmbeddingBuilder(model_name=args.embedding_model)
    elif args.feature_backend == "cross-encoder":
        from quorabust.cross_encoder_features import PairCrossEncoderBuilder

        feature_builder = PairCrossEncoderBuilder(model_name=args.cross_encoder_model)

    builder, clf = train_duplicate_classifier(
        train_df,
        eval_df=eval_df,
        random_state=args.seed,
        feature_builder=feature_builder,
    )

    feat_names = (
        builder.feature_names()
        if hasattr(builder, "feature_names")
        else ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
    )

    meta: dict[str, Any] = {
        "n_train": len(train_df),
        "n_eval": len(eval_df) if eval_df is not None else 0,
        "eval_fraction": args.eval_fraction,
        "eval_split_source": eval_split_source,
        "split_strategy": split_strategy,
        "question_id_columns": ["qid1", "qid2"]
        if split_strategy == "question_component_holdout"
        else [],
        "max_rows": args.max_rows,
        "require_question_ids": args.require_question_ids,
        "csv": str(args.csv.resolve()),
        "csv_sha256": sha256_file(args.csv),
        "seed": args.seed,
        "threshold_candidates": threshold_candidates,
        "threshold_metric": args.threshold_metric,
        "training_command": _training_command(argv),
        "quorabust_version": _package_version(),
        "git_revision": git_revision(),
        "feature_backend": args.feature_backend,
        "feature_schema": feat_names,
    }
    if eval_csv_sha256 is not None:
        meta["eval_csv_sha256"] = eval_csv_sha256
    eval_target = eval_df if eval_df is not None else train_df
    m = eval_classification_metrics(builder, clf, eval_target)
    for k, v in m.items():
        meta[f"eval_{k}"] = v
    if eval_df is not None:
        y_eval = eval_df["is_duplicate"].astype(int).to_numpy()
        p_eval = predict_proba_duplicate(
            builder,
            clf,
            eval_df["question1"].astype(str).tolist(),
            eval_df["question2"].astype(str).tolist(),
        )[:, 1]
        selected_threshold = select_decision_threshold(
            y_eval,
            p_eval,
            thresholds=threshold_candidates,
            optimize_for=args.threshold_metric,
            false_positive_cost=threshold_costs["false_positive_cost"],
            false_negative_cost=threshold_costs["false_negative_cost"],
        )
        meta["decision_threshold"] = selected_threshold["threshold"]
        meta["decision_threshold_source"] = "eval_holdout"
        meta["decision_threshold_metric"] = args.threshold_metric
        meta["decision_threshold_metrics"] = {
            k: v for k, v in selected_threshold.items() if k != "threshold"
        }
        if args.threshold_metric == "expected_cost":
            meta["decision_threshold_costs"] = threshold_costs
    print(
        "metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in sorted(m.items())),
        f"(n={len(eval_target)})",
    )

    X_ref = builder.transform_frame(train_df)
    meta["reference_feature_means"] = feature_means_from_matrix(feat_names, X_ref)

    save_classifier(args.out, builder, clf, meta=meta)
    artifact_sha256 = sha256_file(args.out)
    if args.metadata_out is not None:
        save_metadata_sidecar(
            args.metadata_out,
            {**meta, "artifact_sha256": artifact_sha256},
        )
    if args.registry_dir is not None:
        registry_record = {
            "artifact": str(args.out.resolve()),
            "artifact_sha256": artifact_sha256,
            "feature_backend": args.feature_backend,
            "git_revision": meta.get("git_revision"),
            "quorabust_version": meta.get("quorabust_version"),
            "decision_threshold": meta.get("decision_threshold"),
            "decision_threshold_metric": meta.get("decision_threshold_metric"),
            "eval_csv_sha256": meta.get("eval_csv_sha256"),
            "eval_metrics": {
                k: meta[k]
                for k in ("eval_accuracy", "eval_log_loss", "eval_roc_auc")
                if k in meta
            },
        }
        if "decision_threshold_costs" in meta:
            registry_record["decision_threshold_costs"] = meta["decision_threshold_costs"]
        append_model_record(
            args.registry_dir,
            registry_record,
        )
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
