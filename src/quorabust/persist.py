from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier


def save_classifier(
    path: str | Path,
    builder: Any,
    clf: XGBClassifier,
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist vectorizer+model together (same pickle; load before scoring)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"builder": builder, "clf": clf, "meta": meta or {}}
    with p.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_classifier(path: str | Path) -> tuple[Any, XGBClassifier, dict[str, Any]]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    return data["builder"], data["clf"], data.get("meta", {})
