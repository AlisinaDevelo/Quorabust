from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_revision(cwd: Path | None = None, *, fallback: str = "unknown") -> str:
    """Best-effort `git rev-parse HEAD` for training provenance."""
    root = cwd or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return fallback
