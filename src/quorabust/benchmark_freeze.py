"""Materialize deterministic, leakage-aware benchmark role artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from quorabust.data_audit import audit_dataframe
from quorabust.lineage import git_revision, sha256_file
from quorabust.split import split_question_component_roles

ROLE_NAMES = ("train", "tuning", "calibration", "final_holdout")


def _require_text(value: str | None, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _path_reference(path: Path, prefix: str) -> str:
    return f"{prefix}://{path.name}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _output_paths(out_dir: Path, audit_out: Path, split_out: Path) -> dict[str, Path]:
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"role output directory is not a directory: {out_dir}")

    resolved = {
        "audit": audit_out.resolve(),
        "split": split_out.resolve(),
        **{role: (out_dir / f"{role}.csv").resolve() for role in ROLE_NAMES},
    }
    by_path: dict[Path, list[str]] = {}
    for label, path in resolved.items():
        by_path.setdefault(path, []).append(label)
    collisions = [
        f"{path}: {', '.join(sorted(labels))}"
        for path, labels in by_path.items()
        if len(labels) > 1
    ]
    existing = [f"{label}: {path}" for label, path in resolved.items() if path.exists()]
    if collisions or existing:
        details = collisions + existing
        raise ValueError("output collision; refusing to overwrite: " + "; ".join(details))
    return resolved


def _command_reference(
    *,
    csv_path: Path,
    out_dir: Path,
    audit_out: Path,
    split_out: Path,
    seed: int,
    tuning_fraction: float,
    calibration_fraction: float,
    final_holdout_fraction: float,
) -> str:
    args = [
        "quorabust-freeze-protocol",
        "--csv",
        csv_path.name,
        "--out-dir",
        out_dir.name,
        "--audit-out",
        audit_out.name,
        "--split-out",
        split_out.name,
        "--seed",
        str(seed),
        "--tuning-fraction",
        str(tuning_fraction),
        "--calibration-fraction",
        str(calibration_fraction),
        "--final-holdout-fraction",
        str(final_holdout_fraction),
    ]
    return shlex.join(args)


def freeze_protocol(
    csv_path: Path,
    out_dir: Path,
    audit_out: Path,
    split_out: Path,
    *,
    tuning_fraction: float = 0.1,
    calibration_fraction: float = 0.1,
    final_holdout_fraction: float = 0.1,
    seed: int = 42,
    source_reference: str | None = None,
    audit_reference: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Audit a source CSV and write deterministic role and split artifacts.

    The source and all generated artifacts stay outside the repository. Existing output
    files are never overwritten, which makes an accidental rerun visible and reviewable.
    """
    csv_path = csv_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    audit_out = audit_out.expanduser().resolve()
    split_out = split_out.expanduser().resolve()
    if not csv_path.is_file():
        raise ValueError(f"source CSV does not exist or is not a file: {csv_path}")
    output_paths = _output_paths(out_dir, audit_out, split_out)
    if csv_path in output_paths.values():
        raise ValueError("source CSV must not be one of the generated output files")
    source_ref = (
        _require_text(source_reference, "source_reference")
        if source_reference is not None
        else _path_reference(csv_path, "external")
    )
    audit_ref = (
        _require_text(audit_reference, "audit_reference")
        if audit_reference is not None
        else _path_reference(audit_out, "external")
    )
    command_ref = (
        _require_text(command, "command")
        if command is not None
        else _command_reference(
            csv_path=csv_path,
            out_dir=out_dir,
            audit_out=audit_out,
            split_out=split_out,
            seed=seed,
            tuning_fraction=tuning_fraction,
            calibration_fraction=calibration_fraction,
            final_holdout_fraction=final_holdout_fraction,
        )
    )

    source_sha256 = sha256_file(csv_path)
    df = pd.read_csv(csv_path, dtype={"qid1": "string", "qid2": "string"})
    df.columns = [str(column).strip() for column in df.columns]
    audit = audit_dataframe(
        df,
        source_name=csv_path.name,
        source_sha256=source_sha256,
        require_question_ids=True,
        require_question_text=True,
    )
    _write_json(audit_out, audit)
    if audit["status"] != "pass":
        raise ValueError("source audit failed; no benchmark role files were written")

    roles, stats = split_question_component_roles(
        df,
        tuning_fraction=tuning_fraction,
        calibration_fraction=calibration_fraction,
        final_holdout_fraction=final_holdout_fraction,
        seed=seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for role in ROLE_NAMES:
        roles[role].to_csv(output_paths[role], index=False, lineterminator="\n")

    role_records: dict[str, dict[str, Any]] = {}
    for role in ROLE_NAMES:
        role_stats = stats["roles"][role]
        role_records[role] = {
            "reference": f"roles/{role}.csv",
            "sha256": sha256_file(output_paths[role]),
            "rows": role_stats["rows"],
            "components": role_stats["components"],
            "label_counts": role_stats["label_counts"],
            "observed_fraction": role_stats["observed_fraction"],
        }

    split_payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest": "quorabust.question_component_split",
        "strategy": stats["strategy"],
        "question_id_columns": stats["question_id_columns"],
        "seed": stats["seed"],
        "source": {
            "reference": source_ref,
            "sha256": source_sha256,
            "rows": len(df),
        },
        "audit": {
            "reference": audit_ref,
            "sha256": sha256_file(audit_out),
            "status": audit["status"],
            "source_sha256": source_sha256,
        },
        "requested_fractions": stats["requested_fractions"],
        "roles_are_disjoint": stats["roles_are_disjoint"],
        "safeguards": {
            "final_holdout_used_for_tuning": False,
            "final_holdout_used_for_calibration": False,
            "final_holdout_used_for_model_selection": False,
            "raw_data_committed": False,
        },
        "roles": role_records,
        "provenance": {
            "git_revision": git_revision(),
            "python_version": platform.python_version(),
            "machine": f"{platform.system()}/{platform.machine()}",
            "command": command_ref,
        },
    }
    _write_json(split_out, split_payload)
    return split_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit and materialize deterministic Quorabust benchmark role CSVs.",
    )
    parser.add_argument("--csv", type=Path, required=True, help="Source question-pair CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for train, tuning, calibration, and final_holdout CSVs",
    )
    parser.add_argument("--audit-out", type=Path, required=True, help="Audit JSON output")
    parser.add_argument("--split-out", type=Path, required=True, help="Split manifest JSON output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tuning-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--final-holdout-fraction", type=float, default=0.1)
    parser.add_argument("--source-reference")
    parser.add_argument("--audit-reference")
    args = parser.parse_args(argv)

    try:
        payload = freeze_protocol(
            args.csv,
            args.out_dir,
            args.audit_out,
            args.split_out,
            tuning_fraction=args.tuning_fraction,
            calibration_fraction=args.calibration_fraction,
            final_holdout_fraction=args.final_holdout_fraction,
            seed=args.seed,
            source_reference=args.source_reference,
            audit_reference=args.audit_reference,
        )
    except (OSError, ValueError) as exc:
        print(f"Unable to freeze benchmark protocol: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.split_out.resolve()} source_sha256={payload['source']['sha256']} "
        f"roles={','.join(ROLE_NAMES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
