import re
from pathlib import Path

_FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_github_actions_use_immutable_commit_shas():
    workflow_dir = Path(__file__).parents[1] / ".github" / "workflows"
    references = []
    for workflow in sorted(workflow_dir.glob("*.yml")):
        for line_number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), 1):
            match = re.search(r"^\s*- uses:\s*([^@\s]+)@([^\s#]+)", line)
            if match:
                references.append((workflow, line_number, match.group(1), match.group(2)))

    assert references
    mutable = [
        f"{path}:{line_number}: {owner}@{ref}"
        for path, line_number, owner, ref in references
        if not _FULL_COMMIT_SHA.fullmatch(ref)
    ]
    assert not mutable, "mutable GitHub Actions references found: " + ", ".join(mutable)
