import json
from pathlib import Path

import pandas as pd
import pytest

from quorabust.benchmark_freeze import ROLE_NAMES, freeze_protocol, main
from quorabust.lineage import sha256_file


def _write_source(path: Path, rows: int = 24) -> None:
    pd.DataFrame(
        {
            "question1": [f"question one {index}" for index in range(rows)],
            "question2": [f"question two {index}" for index in range(rows)],
            "is_duplicate": [index % 2 for index in range(rows)],
            "qid1": [f"q{index * 2}" for index in range(rows)],
            "qid2": [f"q{index * 2 + 1}" for index in range(rows)],
        }
    ).to_csv(path, index=False)


def _freeze_paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "roles", root / "audit.json", root / "split.json"


def test_freeze_is_byte_deterministic_and_records_role_hashes(tmp_path):
    source = tmp_path / "source.csv"
    _write_source(source)
    first_roles, first_audit, first_split = _freeze_paths(tmp_path / "first")
    second_roles, second_audit, second_split = _freeze_paths(tmp_path / "second")

    first = freeze_protocol(source, first_roles, first_audit, first_split, seed=17)
    second = freeze_protocol(source, second_roles, second_audit, second_split, seed=17)

    assert first == second
    assert first_audit.read_bytes() == second_audit.read_bytes()
    assert first_split.read_bytes() == second_split.read_bytes()
    for role in ROLE_NAMES:
        first_path = first_roles / f"{role}.csv"
        second_path = second_roles / f"{role}.csv"
        assert first_path.read_bytes() == second_path.read_bytes()
        assert first["roles"][role]["sha256"] == sha256_file(first_path)

    role_ids: dict[str, set[str]] = {}
    for role in ROLE_NAMES:
        frame = pd.read_csv(first_roles / f"{role}.csv", dtype={"qid1": "string", "qid2": "string"})
        role_ids[role] = set(frame["qid1"]) | set(frame["qid2"])
    for index, role in enumerate(ROLE_NAMES):
        for other_role in ROLE_NAMES[index + 1 :]:
            assert role_ids[role].isdisjoint(role_ids[other_role])

    manifest = json.loads(first_split.read_text(encoding="utf-8"))
    assert manifest["source"]["sha256"] == sha256_file(source)
    assert manifest["audit"]["sha256"] == sha256_file(first_audit)
    assert manifest["roles_are_disjoint"] is True
    assert manifest["safeguards"]["final_holdout_used_for_tuning"] is False
    assert manifest["safeguards"]["final_holdout_used_for_calibration"] is False
    assert manifest["safeguards"]["final_holdout_used_for_model_selection"] is False
    assert sum(manifest["roles"][role]["rows"] for role in ROLE_NAMES) == len(pd.read_csv(source))


def test_freeze_writes_failing_audit_without_role_files(tmp_path):
    source = tmp_path / "missing-qids.csv"
    pd.DataFrame(
        {
            "question1": ["a", "b"],
            "question2": ["c", "d"],
            "is_duplicate": [0, 1],
        }
    ).to_csv(source, index=False)
    roles, audit, split = _freeze_paths(tmp_path / "outputs")

    with pytest.raises(ValueError, match="source audit failed"):
        freeze_protocol(source, roles, audit, split)

    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert not roles.exists()
    assert not split.exists()


def test_freeze_rejects_insufficient_components_after_audit(tmp_path):
    source = tmp_path / "too-small.csv"
    _write_source(source, rows=3)
    roles, audit, split = _freeze_paths(tmp_path / "outputs")

    with pytest.raises(ValueError, match="at least four question components"):
        freeze_protocol(source, roles, audit, split)

    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "pass"
    assert not roles.exists()
    assert not split.exists()


def test_freeze_refuses_output_collision_without_changing_artifacts(tmp_path):
    source = tmp_path / "source.csv"
    _write_source(source)
    roles, audit, split = _freeze_paths(tmp_path / "outputs")
    freeze_protocol(source, roles, audit, split)
    before = {
        path: path.read_bytes()
        for path in [audit, split, *(roles / f"{role}.csv" for role in ROLE_NAMES)]
    }

    with pytest.raises(ValueError, match="output collision"):
        freeze_protocol(source, roles, audit, split)

    assert {path: path.read_bytes() for path in before} == before


def test_freeze_cli_materializes_roles(tmp_path, capsys):
    source = tmp_path / "source.csv"
    _write_source(source)
    roles, audit, split = _freeze_paths(tmp_path / "cli")

    assert (
        main(
            [
                "--csv",
                str(source),
                "--out-dir",
                str(roles),
                "--audit-out",
                str(audit),
                "--split-out",
                str(split),
                "--seed",
                "9",
            ]
        )
        == 0
    )
    assert "roles=train,tuning,calibration,final_holdout" in capsys.readouterr().out
    assert all((roles / f"{role}.csv").is_file() for role in ROLE_NAMES)
