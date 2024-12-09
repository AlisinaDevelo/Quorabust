# Enterprise operations

## Supply chain

- **Dependabot** updates GitHub Actions and pip dependencies weekly (see `.github/dependabot.yml`).
- **Audit** workflow runs `pip-audit` on pushes, PRs, and weekly; install pins `pip>=26` before auditing.
- **Pre-commit** (optional locally): Ruff and Mypy on `src/quorabust` (see `.pre-commit-config.yaml`).

## Builds

- **Python package**: PEP 621 metadata in `pyproject.toml`; install with `pip install .` or `pip install -e ".[dev]"`.
- **Container**: `docker build -t quorabust .` then mount data and pass CLI flags, e.g.  
  `docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/models:/models" quorabust --csv /data/raw/train.csv --out /models/model.pkl`

## Lineage and artifacts

Training writes `csv_sha256`, `git_revision`, `quorabust_version`, and metric fields into the pickle `meta` dict. Treat `.pkl` files as **trusted** (pickle); load only from controlled storage.

## Releases

Tag versions, update `CHANGELOG.md`, and align `[project].version` in `pyproject.toml`. Publish to an internal index or PyPI as appropriate.
