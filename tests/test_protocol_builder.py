import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from quorabust.benchmark_protocol import validate_protocol_payload
from quorabust.data_audit import audit_csv
from quorabust.lineage import sha256_file
from quorabust.protocol_builder import build_protocol, build_protocol_payload, main

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> tuple[Path, dict, dict[str, Path]]:
    source_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "question1": ["a", "b", "d", "f"],
            "question2": ["a", "c", "e", "g"],
            "is_duplicate": [1, 0, 0, 1],
            "qid1": ["q1", "q2", "q4", "q6"],
            "qid2": ["q1", "q3", "q5", "q7"],
        }
    ).to_csv(source_path, index=False)

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(audit_csv(source_path, require_question_ids=True), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    role_paths: dict[str, Path] = {}
    for role in ("train", "tuning", "calibration", "final_holdout"):
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
        role_paths[role] = path

    split_path = tmp_path / "split.json"
    split_path.write_text('{"strategy": "question_component_holdout"}\n', encoding="utf-8")
    config = {
        "protocol_name": "quorabust-synthetic-builder-v1",
        "dataset": {
            "name": "synthetic protocol builder fixture",
            "source_path": source_path.name,
            "source_reference": "synthetic://source.csv",
            "license": "repository test fixture",
            "terms": "no external benchmark claim",
            "audit_path": audit_path.name,
            "audit_reference": "synthetic://audit.json",
        },
        "roles": {
            role: {
                "purpose": f"{role} purpose",
                "path": path.name,
                "reference": f"synthetic://{role}.json",
            }
            for role, path in role_paths.items()
        },
        "split": {
            "manifest_path": split_path.name,
            "manifest_reference": "synthetic://split.json",
            "seed": 42,
            "eval_fraction": 0.2,
            "question_id_columns": ["qid1", "qid2"],
        },
        "decision_policy": {
            "threshold_metric": "f1",
            "threshold_candidates": [0.2, 0.5, 0.8],
            "calibration_method": "sigmoid",
        },
        "dependency_lock_path": str(REPOSITORY_ROOT / "requirements.txt"),
        "dependency_lock_reference": "repo://requirements.txt",
        "repository_path": str(REPOSITORY_ROOT),
        "command": "quorabust-build-protocol --config protocol.json",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    paths = {
        "source": source_path,
        "audit": audit_path,
        "split": split_path,
        **role_paths,
    }
    return config_path, config, paths


def test_build_protocol_payload_hashes_audited_artifacts(tmp_path):
    config_path, config, paths = _fixture(tmp_path)

    payload = build_protocol_payload(config, base_dir=config_path.parent)

    assert validate_protocol_payload(payload) == []
    assert payload["dataset"]["sha256"] == sha256_file(paths["source"])
    assert payload["dataset"]["audit"]["sha256"] == sha256_file(paths["audit"])
    assert payload["roles"]["train"]["artifact"]["sha256"] == sha256_file(paths["train"])
    assert payload["split"]["manifest"]["sha256"] == sha256_file(paths["split"])
    assert len(payload["provenance"]["git_revision"]) == 40


def test_build_protocol_writes_canonical_manifest_and_cli_output(tmp_path, capsys):
    config_path, _, _ = _fixture(tmp_path)
    output_path = tmp_path / "reports" / "protocol.json"

    payload = build_protocol(config_path, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert main(["--config", str(config_path), "--out", str(output_path)]) == 0
    assert "wrote" in capsys.readouterr().out


def test_build_protocol_rehashes_changed_role_artifact(tmp_path):
    config_path, config, paths = _fixture(tmp_path)
    first = build_protocol_payload(config, base_dir=config_path.parent)

    paths["train"].write_text('{"role": "train", "revision": 2}\n', encoding="utf-8")
    second = build_protocol_payload(config, base_dir=config_path.parent)

    first_train_sha = first["roles"]["train"]["artifact"]["sha256"]
    second_train_sha = second["roles"]["train"]["artifact"]["sha256"]
    assert first_train_sha != second_train_sha
    assert first["roles"]["tuning"]["artifact"] == second["roles"]["tuning"]["artifact"]


def test_build_protocol_rejects_source_changed_after_audit(tmp_path):
    config_path, config, paths = _fixture(tmp_path)
    source_text = paths["source"].read_text(encoding="utf-8")
    paths["source"].write_text(source_text + "x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source SHA-256"):
        build_protocol_payload(config, base_dir=config_path.parent)


def test_build_protocol_requires_repository_path(tmp_path):
    config_path, config, _ = _fixture(tmp_path)
    incomplete = copy.deepcopy(config)
    del incomplete["repository_path"]

    with pytest.raises(ValueError, match=r"missing config field\(s\): repository_path"):
        build_protocol_payload(incomplete, base_dir=config_path.parent)


def test_build_protocol_cli_fails_for_invalid_config(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text("{", encoding="utf-8")

    assert main(["--config", str(config_path), "--out", str(tmp_path / "protocol.json")]) == 1
    assert "invalid builder config JSON" in capsys.readouterr().err
