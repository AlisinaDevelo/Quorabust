# Contributing

Use a virtual environment, install dev extras, and run checks before opening a PR:

```bash
pip install -e ".[dev]"
ruff check src tests
mypy src/quorabust
pytest -q --cov=quorabust --cov-fail-under=70
```

Optional: `pre-commit install` then `pre-commit run --all-files`.

Keep commits focused; follow existing style and typing patterns in `src/quorabust`.
