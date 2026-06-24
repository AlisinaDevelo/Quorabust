#!/usr/bin/env python3
"""Generate a small, fully synthetic duplicate-question dataset.

This is NOT the Kaggle Quora Question Pairs dataset and carries no real labels.
It exists only so the training, reporting, and serving paths can be exercised
end-to-end (and a sample model card produced) without downloading or
redistributing third-party data. Numbers produced from it describe the toy
dataset, not real-world model quality.

Deterministic: a fixed seed yields the same CSV every run.

    python scripts/make_sample_data.py --out data/sample/synthetic_pairs.csv --rows 2000
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

TOPICS = [
    "Python", "machine learning", "the stock market", "climate change",
    "remote work", "Kubernetes", "the violin", "espresso", "PostgreSQL",
    "rock climbing", "the immune system", "quantum computing", "Go modules",
    "sourdough bread", "electric cars", "the Roman empire", "REST APIs",
    "sleep hygiene", "compound interest", "the water cycle",
]

# Paraphrase frames that ask essentially the same thing about a topic.
SIMILAR_FRAMES = [
    "How do I get started with {t}?",
    "What is the best way to begin learning {t}?",
    "How can a beginner start out with {t}?",
    "Where should someone new to {t} begin?",
]

# Frames with a clearly different intent about the same topic.
DIFFERENT_FRAMES = [
    "What is the history of {t}?",
    "Why is {t} controversial?",
    "How much does {t} cost?",
    "Who invented {t}?",
    "What are the risks of {t}?",
]


def build_rows(rows: int, seed: int) -> list[tuple[str, str, int]]:
    rng = random.Random(seed)
    out: list[tuple[str, str, int]] = []
    for _ in range(rows):
        if rng.random() < 0.5:
            # Duplicate: same topic, two different paraphrase frames.
            topic = rng.choice(TOPICS)
            f1, f2 = rng.sample(SIMILAR_FRAMES, 2)
            out.append((f1.format(t=topic), f2.format(t=topic), 1))
        else:
            # Non-duplicate: either different topics, or same topic / different intent.
            if rng.random() < 0.5:
                t1, t2 = rng.sample(TOPICS, 2)
                q1 = rng.choice(SIMILAR_FRAMES + DIFFERENT_FRAMES).format(t=t1)
                q2 = rng.choice(SIMILAR_FRAMES + DIFFERENT_FRAMES).format(t=t2)
            else:
                topic = rng.choice(TOPICS)
                q1 = rng.choice(SIMILAR_FRAMES).format(t=topic)
                q2 = rng.choice(DIFFERENT_FRAMES).format(t=topic)
            out.append((q1, q2, 0))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/sample/synthetic_pairs.csv"))
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.rows, args.seed)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["question1", "question2", "is_duplicate"])
        w.writerows(rows)

    dup = sum(r[2] for r in rows)
    print(
        f"wrote {len(rows)} rows to {args.out} "
        f"({dup} duplicate, {len(rows) - dup} non-duplicate)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
