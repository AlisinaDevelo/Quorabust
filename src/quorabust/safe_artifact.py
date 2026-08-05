from __future__ import annotations

import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from quorabust.features import PairFeatureBuilder
from quorabust.lineage import sha256_file

SAFE_ARTIFACT_FORMAT = "quorabust.safe.tfidf_xgboost"
SAFE_ARTIFACT_SUFFIX = ".qmodel"
_MEMBERS = {"manifest.json", "builder.json", "booster.json"}
_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_PATH_METADATA_KEYS = {
    "csv",
    "eval_csv",
    "calibration_csv",
    "threshold_csv",
    "training_command",
    "calibration_command",
}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("safe artifact metadata must contain finite numbers")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    raise TypeError(f"metadata value is not JSON serializable: {type(value).__name__}")


def safe_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    source = meta or {}
    filtered = {
        str(key): _json_value(value)
        for key, value in source.items()
        if str(key) not in _PATH_METADATA_KEYS and not str(key).endswith("_path")
    }
    filtered["artifact_format"] = SAFE_ARTIFACT_FORMAT
    return filtered


def _vectorizer_params(vectorizer: TfidfVectorizer) -> dict[str, Any]:
    keys = (
        "max_features",
        "ngram_range",
        "min_df",
        "stop_words",
        "norm",
        "use_idf",
        "smooth_idf",
        "sublinear_tf",
        "lowercase",
        "token_pattern",
    )
    return {
        key: _json_value(getattr(vectorizer, key))
        for key in keys
    }


def _builder_payload(builder: Any) -> dict[str, Any]:
    if not isinstance(builder, PairFeatureBuilder):
        raise TypeError("safe artifacts currently support PairFeatureBuilder only")
    if not builder._fitted:
        raise ValueError("feature builder must be fitted before safe export")
    vectorizer = builder._vec
    vocabulary = getattr(vectorizer, "vocabulary_", None)
    idf = getattr(vectorizer, "idf_", None)
    if not isinstance(vocabulary, dict) or idf is None:
        raise ValueError("fitted TF-IDF state is incomplete")
    normalized_vocabulary = {str(key): int(value) for key, value in vocabulary.items()}
    expected_indices = set(range(len(normalized_vocabulary)))
    if set(normalized_vocabulary.values()) != expected_indices:
        raise ValueError("TF-IDF vocabulary indices must be contiguous")
    idf_values = np.asarray(idf, dtype=np.float64).reshape(-1)
    if len(idf_values) != len(normalized_vocabulary):
        raise ValueError("TF-IDF vocabulary and IDF lengths must match")
    return {
        "type": "PairFeatureBuilder",
        "params": _vectorizer_params(vectorizer),
        "vocabulary": normalized_vocabulary,
        "idf": [_json_value(value) for value in idf_values.tolist()],
    }


def save_safe_classifier(
    path: str | Path,
    builder: Any,
    clf: Any,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a non-pickle TF-IDF/XGBoost artifact as a validated ZIP bundle."""
    if not isinstance(clf, XGBClassifier):
        raise TypeError("safe artifacts currently support XGBClassifier only")
    builder_payload = _builder_payload(builder)
    metadata = safe_metadata(meta)
    try:
        booster_bytes = bytes(clf.get_booster().save_raw(raw_format="json"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("XGBClassifier must be fitted before safe export") from exc

    manifest = {
        "schema_version": 1,
        "format": SAFE_ARTIFACT_FORMAT,
        "metadata": metadata,
        "builder_member": "builder.json",
        "booster_member": "booster.json",
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        archive.writestr("builder.json", json.dumps(builder_payload, sort_keys=True))
        archive.writestr("booster.json", booster_bytes)
    return output


def _validate_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or set(names) != _MEMBERS:
        raise ValueError("safe artifact must contain exactly the expected members")
    for info in infos:
        if info.is_dir() or info.file_size > _MAX_MEMBER_BYTES:
            raise ValueError("safe artifact contains an invalid member")


def _load_builder(payload: dict[str, Any]) -> PairFeatureBuilder:
    if payload.get("type") != "PairFeatureBuilder":
        raise ValueError("unsupported safe feature builder")
    params = payload.get("params")
    vocabulary = payload.get("vocabulary")
    idf = payload.get("idf")
    if (
        not isinstance(params, dict)
        or not isinstance(vocabulary, dict)
        or not isinstance(idf, list)
    ):
        raise ValueError("safe builder payload is malformed")
    ngram_range = params.get("ngram_range")
    if not isinstance(ngram_range, list) or len(ngram_range) != 2:
        raise ValueError("safe builder ngram_range is malformed")
    normalized_vocabulary = {str(key): int(value) for key, value in vocabulary.items()}
    if set(normalized_vocabulary.values()) != set(range(len(normalized_vocabulary))):
        raise ValueError("safe builder vocabulary indices are malformed")
    idf_values = np.asarray(idf, dtype=np.float64).reshape(-1)
    if len(idf_values) != len(normalized_vocabulary) or not np.isfinite(idf_values).all():
        raise ValueError("safe builder IDF values are malformed")

    vectorizer = TfidfVectorizer(
        max_features=params.get("max_features"),
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        min_df=params.get("min_df", 1),
        stop_words=params.get("stop_words"),
        norm=params.get("norm", "l2"),
        use_idf=bool(params.get("use_idf", True)),
        smooth_idf=bool(params.get("smooth_idf", True)),
        sublinear_tf=bool(params.get("sublinear_tf", False)),
        lowercase=bool(params.get("lowercase", True)),
        token_pattern=params.get("token_pattern"),
    ).fit(["quorabust placeholder"])
    vectorizer.vocabulary_ = normalized_vocabulary
    vectorizer.fixed_vocabulary_ = True
    vectorizer._tfidf.idf_ = idf_values
    vectorizer._tfidf.n_features_in_ = len(normalized_vocabulary)

    max_features = params.get("max_features")
    if not isinstance(max_features, int):
        raise ValueError("safe builder max_features is malformed")
    builder = PairFeatureBuilder(
        max_features=max_features,
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
    )
    builder._vec = vectorizer
    builder._fitted = True
    return builder


def _load_booster(raw: bytes) -> XGBClassifier:
    with tempfile.NamedTemporaryFile(suffix=".json") as temporary:
        temporary.write(raw)
        temporary.flush()
        classifier = XGBClassifier()
        classifier.load_model(temporary.name)
    return classifier


def load_safe_classifier(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[PairFeatureBuilder, XGBClassifier, dict[str, Any]]:
    """Load only the JSON/ZIP safe artifact format; no code deserialization occurs."""
    artifact = Path(path)
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("expected_sha256 must be a 64 hexadecimal character string")
        actual = sha256_file(artifact)
        if actual != expected:
            raise ValueError(f"artifact SHA-256 mismatch for {artifact}")
    with zipfile.ZipFile(artifact, mode="r") as archive:
        _validate_archive(archive)
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError("safe artifact manifest is malformed")
        if manifest.get("schema_version") != 1 or manifest.get("format") != SAFE_ARTIFACT_FORMAT:
            raise ValueError("unsupported safe artifact format")
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("safe artifact metadata is malformed")
        builder_payload = json.loads(archive.read("builder.json"))
        booster_bytes = archive.read("booster.json")
    builder = _load_builder(builder_payload)
    classifier = _load_booster(booster_bytes)
    return builder, classifier, dict(metadata)
