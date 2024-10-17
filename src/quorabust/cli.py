from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quorabust.model import eval_log_loss, train_duplicate_classifier
from quorabust.persist import save_classifier


def _load_quora_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {"question1", "question2", "is_duplicate"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns {sorted(missing)}; found {list(df.columns)}")
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train Quorabust duplicate classifier from a Quora-style CSV.",
    )
    p.add_argument("--csv", type=Path, required=True, help="Path to train.csv (Kaggle Quora Question Pairs)")
    p.add_argument("--out", type=Path, required=True, help="Output .pkl (builder + XGBoost)")
    p.add_argument("--max-rows", type=int, default=None, help="Cap rows after shuffle (debug / subset)")
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Holdout fraction for early stopping and reported log loss (0 disables eval split)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not args.csv.is_file():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 1

    df = _load_quora_csv(args.csv)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    eval_df = None
    if args.eval_fraction > 0:
        n_eval = max(1, int(len(df) * args.eval_fraction))
        if n_eval >= len(df):
            n_eval = max(1, len(df) // 5)
        eval_df = df.iloc[:n_eval].copy()
        train_df = df.iloc[n_eval:].copy()
        if len(train_df) < 10:
            train_df = df.copy()
            eval_df = None

    builder, clf = train_duplicate_classifier(
        train_df if eval_df is not None else df,
        eval_df=eval_df,
        random_state=args.seed,
    )

    meta = {
        "n_train": int(len(train_df) if eval_df is not None else len(df)),
        "n_eval": int(len(eval_df)) if eval_df is not None else 0,
        "csv": str(args.csv.resolve()),
        "seed": args.seed,
    }
    if eval_df is not None:
        ll = eval_log_loss(builder, clf, eval_df)
        meta["eval_log_loss"] = ll
        print(f"eval log_loss: {ll:.4f} (n_eval={len(eval_df)})")
    else:
        ll = eval_log_loss(builder, clf, df)
        meta["train_log_loss"] = ll
        print(f"train log_loss (no holdout): {ll:.4f}")

    save_classifier(args.out, builder, clf, meta=meta)
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
