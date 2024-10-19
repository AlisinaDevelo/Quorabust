from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.model import eval_log_loss, train_duplicate_classifier
from quorabust.persist import save_classifier


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

    builder, clf = train_duplicate_classifier(
        train_df,
        eval_df=eval_df,
        random_state=args.seed,
    )

    meta: dict[str, Any] = {
        "n_train": len(train_df),
        "n_eval": len(eval_df) if eval_df is not None else 0,
        "csv": str(args.csv.resolve()),
        "seed": args.seed,
    }
    if eval_df is not None:
        ll = eval_log_loss(builder, clf, eval_df)
        meta["eval_log_loss"] = ll
        print(f"eval log_loss: {ll:.4f} (n_eval={len(eval_df)})")
    else:
        ll = eval_log_loss(builder, clf, train_df)
        meta["train_log_loss"] = ll
        print(f"log_loss (no holdout): {ll:.4f}")

    save_classifier(args.out, builder, clf, meta=meta)
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
