from __future__ import annotations

import hmac
import json
import pickle
from pathlib import Path
from typing import Any

from quorabust.lineage import sha256_file


def save_classifier(
    path: str | Path,
    builder: Any,
    clf: Any,
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist vectorizer+model together (same pickle; load before scoring)."""
    p = Path(path)
    if p.suffix == ".qmodel":
        from quorabust.safe_artifact import save_safe_classifier

        save_safe_classifier(p, builder, clf, meta=meta)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"builder": builder, "clf": clf, "meta": meta or {}}
    with p.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _normalize_expected_sha256(expected_sha256: str | None) -> str | None:
    if expected_sha256 is None:
        return None
    normalized = expected_sha256.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("expected_sha256 must be a 64 hexadecimal character string")
    return normalized


def load_classifier(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    p = Path(path)
    expected = _normalize_expected_sha256(expected_sha256)
    if expected is not None:
        actual = sha256_file(p)
        if not hmac.compare_digest(actual, expected):
            raise ValueError(f"artifact SHA-256 mismatch for {p}")
    if p.suffix == ".qmodel":
        from quorabust.safe_artifact import load_safe_classifier

        return load_safe_classifier(p)
    with p.open("rb") as f:
        data = pickle.load(f)
    return data["builder"], data["clf"], data.get("meta", {})


def save_metadata_sidecar(
    path: str | Path,
    meta: dict[str, Any],
) -> Path:
    """Write artifact metadata as JSON so reviewers do not need to load a pickle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
