from pathlib import Path

from quorabust.lineage import git_revision, sha256_file


def test_sha256_file_stable(tmp_path: Path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == sha256_file(p)
    assert len(sha256_file(p)) == 64


def test_git_revision_returns_string():
    r = git_revision(cwd=Path(__file__).resolve().parents[1])
    assert isinstance(r, str)
    assert len(r) >= 7 or r == "unknown"
