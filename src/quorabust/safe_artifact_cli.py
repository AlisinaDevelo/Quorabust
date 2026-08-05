from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from quorabust.lineage import git_revision, sha256_file
from quorabust.persist import load_classifier, save_metadata_sidecar
from quorabust.safe_artifact import SAFE_ARTIFACT_FORMAT, safe_metadata, save_safe_classifier


def _command(argv: list[str] | None) -> str:
    arguments = sys.argv[1:] if argv is None else argv
    return shlex.join(["quorabust-export-safe", *(str(value) for value in arguments)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a trusted artifact to the non-pickle TF-IDF/XGBoost format.",
    )
    parser.add_argument("--model", type=Path, required=True, help="Trusted source artifact")
    parser.add_argument("--out", type=Path, required=True, help="Output .qmodel artifact")
    parser.add_argument("--model-sha256", default=None, help="Optional source artifact digest")
    parser.add_argument("--metadata-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(f"File not found: {args.model}", file=sys.stderr)
        return 1
    try:
        builder, classifier, source_meta = load_classifier(
            args.model,
            expected_sha256=args.model_sha256,
        )
        source_digest = sha256_file(args.model)
        export_meta = {
            **safe_metadata(source_meta),
            "artifact_format": SAFE_ARTIFACT_FORMAT,
            "source_artifact_sha256": source_digest,
            "safe_export_git_revision": git_revision(),
            "safe_exported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        save_safe_classifier(args.out, builder, classifier, meta=export_meta)
        output_digest = sha256_file(args.out)
    except (OSError, TypeError, ValueError) as exc:
        print(f"Unable to export safe artifact: {exc}", file=sys.stderr)
        return 1

    if args.metadata_out is not None:
        save_metadata_sidecar(
            args.metadata_out,
            {
                **export_meta,
                "artifact_sha256": output_digest,
                "safe_export_command": _command(argv),
            },
        )
    print(
        json.dumps(
            {
                "artifact_format": SAFE_ARTIFACT_FORMAT,
                "artifact_sha256": output_digest,
                "source_artifact_sha256": source_digest,
                "out": args.out.name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
