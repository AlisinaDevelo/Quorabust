from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd


def iter_csv_batches(
    path: str | Path,
    chunksize: int = 50_000,
    **read_csv_kw: Any,
) -> Iterator[pd.DataFrame]:
    """
    Stream a large CSV as DataFrame chunks (pandas ``chunksize``).

    Use for out-of-core preprocessing, sampling, or custom training loops.
    """
    p = Path(path)
    yield from pd.read_csv(p, chunksize=chunksize, **read_csv_kw)
