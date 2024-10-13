from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def clean_text(text: str | None) -> str:
    """Lowercase, normalize unicode, strip noise, collapse whitespace."""
    if text is None or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text).lower().strip()
    s = _NON_WORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization after cleaning."""
    c = clean_text(text)
    if not c:
        return []
    return c.split()
