from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.drift import feature_means_from_matrix
from quorabust.lineage import git_revision, sha256_file
from quorabust.model import eval_classification_metrics, train_duplicate_classifier
from quorabust.persist import save_classifier, save_metadata_sidecar
from quorabust.registry import append_model_record


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature-backend",
        choices=["tfidf", "embedding"],
        default="tfidf",
        help="tfidf (default) or sentence-transformer embeddings (requires nlp extra)",
    )
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="When --feature-backend=embedding, SentenceTransformer model id",
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
    args = p.parse_args(argv)

    if not args.csv.is_file():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 1

    try:
        df = _load_quora_csv(args.csv)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    n = len(df)
    eval_df = None
    train_df = df
    if args.eval_fraction > 0 and n >= 20:
        n_eval = max(1, min(int(n * args.eval_fraction), n - 1))
        eval_df = df.iloc[:n_eval].copy()
        train_df = df.iloc[n_eval:].copy()

    feature_builder: Any | None = None
    if args.feature_backend == "embedding":
        from quorabust.embedding_features import PairEmbeddingBuilder

        feature_builder = PairEmbeddingBuilder(model_name=args.embedding_model)

    builder, clf = train_duplicate_classifier(
        train_df,
        eval_df=eval_df,
        random_state=args.seed,
        feature_builder=feature_builder,
    )

    if args.feature_backend == "embedding":
        feat_names = ["cos", "l2", "mad", "len_ratio", "len_sum"]
    else:
        feat_names = ["cos", "jaccard", "len_ratio", "abs_len_diff", "len_sum"]

    meta: dict[str, Any] = {
        "n_train": len(train_df),
        "n_eval": len(eval_df) if eval_df is not None else 0,
        "csv": str(args.csv.resolve()),
        "csv_sha256": sha256_file(args.csv),
        "seed": args.seed,
        "quorabust_version": _package_version(),
        "git_revision": git_revision(),
        "feature_backend": args.feature_backend,
        "feature_schema": feat_names,
    }
    eval_target = eval_df if eval_df is not None else train_df
    m = eval_classification_metrics(builder, clf, eval_target)
    for k, v in m.items():
        meta[f"eval_{k}"] = v
    print(
        "metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in sorted(m.items())),
        f"(n={len(eval_target)})",
    )

    X_ref = builder.transform_frame(train_df)
    meta["reference_feature_means"] = feature_means_from_matrix(feat_names, X_ref)

    save_classifier(args.out, builder, clf, meta=meta)
    if args.metadata_out is not None:
        save_metadata_sidecar(args.metadata_out, meta)
    if args.registry_dir is not None:
        append_model_record(
            args.registry_dir,
            {
                "artifact": str(args.out.resolve()),
                "feature_backend": args.feature_backend,
                "git_revision": meta.get("git_revision"),
                "quorabust_version": meta.get("quorabust_version"),
                "eval_metrics": {k: meta[k] for k in meta if k.startswith("eval_")},
            },
        )
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
