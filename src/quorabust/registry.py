from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_model_record(registry_dir: str | Path, record: dict[str, Any]) -> Path:
    """Append one JSON object per line (simple model registry)."""
    d = Path(registry_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "models.jsonl"
    rec = {**record, "registered_at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return path


def load_model_records(registry_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(registry_dir) / "models.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
