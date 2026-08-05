from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from quorabust.features import PairFeatureBuilder
from quorabust.lineage import sha256_file
from quorabust.persist import save_classifier

DEMO_REQUEST = {
    "question1": [
        "How do I learn Python?",
        "How should I cache API responses?",
    ],
    "question2": [
        "What is the best way to learn Python?",
        "Where can I buy train tickets?",
    ],
}

_JSON_FLOAT_TOLERANCE = 1e-3
_RUNTIME_ARTIFACT_SHA256 = "<runtime-artifact-sha256>"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_smoke_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"question1", "question2", "is_duplicate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)} in {path}")
    return df


class _DemoDuplicateClassifier:
    """Deterministic scorer for API snapshots; not a benchmark model."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cos = x[:, 0]
        duplicate = np.where(cos > 0.1, 0.95, 0.25)
        return np.column_stack([1.0 - duplicate, duplicate])


def _write_smoke_artifact(csv_path: Path, model_path: Path, seed: int) -> None:
    df = _load_smoke_csv(csv_path)
    builder = PairFeatureBuilder()
    builder.fit_from_frame(df)
    clf = _DemoDuplicateClassifier()
    feature_schema = builder.feature_names()
    save_classifier(
        model_path,
        builder,
        clf,
        meta={
            "feature_backend": "tfidf",
            "feature_schema": feature_schema,
            "n_train": len(df),
            "n_eval": 0,
            "seed": seed,
            "csv_sha256": sha256_file(csv_path),
            "decision_threshold": 0.5,
            "decision_threshold_source": "demo_default",
            "demo_scorer": "deterministic_tfidf_contract",
        },
    )


def _openapi_excerpt(spec: dict[str, Any]) -> dict[str, Any]:
    components = spec.get("components", {}).get("schemas", {})
    return {
        "openapi": spec.get("openapi"),
        "info": spec.get("info"),
        "paths": {
            "/predict": spec.get("paths", {}).get("/predict"),
            "/models": spec.get("paths", {}).get("/models"),
            "/metrics": spec.get("paths", {}).get("/metrics"),
        },
        "components": {
            "schemas": {
                name: components[name]
                for name in ("PredictBody", "PredictOut", "HTTPValidationError")
                if name in components
            }
        },
    }


def _snapshot_models_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep temporary pickle identity out of committed, cross-process snapshots."""
    variants = payload.get("variants")
    if isinstance(variants, dict):
        for metadata in variants.values():
            if isinstance(metadata, dict) and "artifact_sha256" in metadata:
                metadata["artifact_sha256"] = _RUNTIME_ARTIFACT_SHA256
    return payload


def build_demo_assets(csv_path: Path, out_dir: Path, seed: int = 7) -> list[Path]:
    try:
        from starlette.testclient import TestClient

        from quorabust.serve import create_app
    except ImportError as exc:
        raise RuntimeError(
            'demo asset generation requires serving/test dependencies; install ".[dev]"'
        ) from exc

    if not csv_path.is_file():
        raise FileNotFoundError(f"demo CSV not found: {csv_path}")

    written: list[Path] = []
    with TemporaryDirectory(prefix="quorabust-demo-") as tmp:
        model_path = Path(tmp) / "quorabust-smoke.pkl"
        _write_smoke_artifact(csv_path, model_path, seed)
        app = create_app(model_path_a=str(model_path))
        with TestClient(app) as client:
            predict_response = client.post(
                "/predict?explain=true&threshold=0.9",
                json=DEMO_REQUEST,
            )
            predict_response.raise_for_status()
            models_response = client.get("/models")
            models_response.raise_for_status()
            openapi_response = client.get("/openapi.json")
            openapi_response.raise_for_status()

    assets = {
        "predict-request.json": DEMO_REQUEST,
        "predict-response.json": predict_response.json(),
        "models-response.json": _snapshot_models_response(models_response.json()),
        "openapi-excerpt.json": _openapi_excerpt(openapi_response.json()),
    }
    for filename, payload in assets.items():
        path = out_dir / filename
        _write_json(path, payload)
        written.append(path)

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Demo API snapshots",
                "",
                "Generated from `examples/smoke_pairs.csv` with `quorabust-demo-assets`.",
                "These files demonstrate the serving contract and are not model-quality claims.",
                "The snapshot uses a deterministic TF-IDF demo scorer so CI stays stable.",
                "The temporary demo pickle uses <runtime-artifact-sha256>; live deployments "
                "expose the exact digest.",
                "",
                "- `predict-request.json`: sample batch scoring payload.",
                "- `predict-response.json`: response shape with a request threshold override "
                "and feature values.",
                "- `models-response.json`: safe public model metadata returned by `/models`.",
                "- `openapi-excerpt.json`: focused OpenAPI slice for `/predict`, "
                "`/models`, and `/metrics`.",
                "",
                "Regenerate:",
                "",
                "```bash",
                "quorabust-demo-assets --out docs/demo-assets",
                "```",
                "",
                "CI freshness check:",
                "",
                "```bash",
                "quorabust-demo-assets --out docs/demo-assets --check",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    written.append(readme)
    return written


def _check_demo_assets(csv_path: Path, out_dir: Path, seed: int) -> list[Path]:
    with TemporaryDirectory(prefix="quorabust-demo-check-") as tmp:
        expected_dir = Path(tmp) / "assets"
        expected = build_demo_assets(csv_path, expected_dir, seed=seed)
        stale: list[Path] = []
        for expected_path in expected:
            actual_path = out_dir / expected_path.relative_to(expected_dir)
            if not actual_path.is_file():
                stale.append(actual_path)
                continue
            if not _asset_matches(actual_path, expected_path):
                stale.append(actual_path)
    return stale


def _asset_matches(actual_path: Path, expected_path: Path) -> bool:
    if actual_path.suffix == ".json" and expected_path.suffix == ".json":
        try:
            actual = json.loads(actual_path.read_text(encoding="utf-8"))
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return _json_matches(actual, expected)
    return actual_path.read_bytes() == expected_path.read_bytes()


def _json_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=_JSON_FLOAT_TOLERANCE,
            abs_tol=_JSON_FLOAT_TOLERANCE,
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return False
        return all(_json_matches(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return False
        return all(_json_matches(a, e) for a, e in zip(actual, expected, strict=True))
    return bool(actual == expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate JSON snapshots for the local Quorabust demo API.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("examples/smoke_pairs.csv"),
        help="Smoke CSV used to train the demo artifact.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/demo-assets"),
        help="Directory for generated JSON and README snapshots.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated snapshots differ from the files already in --out.",
    )
    args = parser.parse_args(argv)

    try:
        if args.check:
            stale = _check_demo_assets(args.csv, args.out, seed=args.seed)
            if stale:
                for path in stale:
                    print(f"stale demo asset: {path}", file=sys.stderr)
                return 1
            print(f"demo assets are current in {args.out}")
            return 0

        written = build_demo_assets(args.csv, args.out, seed=args.seed)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
