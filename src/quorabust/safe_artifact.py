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

from quorabust.calibration import CalibratedClassifier, ProbabilityCalibrator
from quorabust.features import PairFeatureBuilder
from quorabust.lineage import sha256_file

SAFE_ARTIFACT_FORMAT = "quorabust.safe.tfidf_xgboost"
SAFE_ARTIFACT_SUFFIX = ".qmodel"
_BASE_MEMBERS = {"manifest.json", "builder.json", "booster.json"}
_CALIBRATION_MEMBER = "calibrator.json"
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
    calibration_payload: dict[str, Any] | None = None
    base_classifier = clf
    if isinstance(clf, CalibratedClassifier):
        if not isinstance(clf.calibrator, ProbabilityCalibrator):
            raise TypeError("safe artifacts require ProbabilityCalibrator wrappers")
        base_classifier = clf.base_classifier
        calibration_payload = clf.calibrator.to_safe_payload()
    if not isinstance(base_classifier, XGBClassifier):
        raise TypeError(
            "safe artifacts currently support XGBClassifier or CalibratedClassifier[XGBClassifier]"
        )
    builder_payload = _builder_payload(builder)
    metadata = safe_metadata(meta)
    if calibration_payload is not None:
        if metadata.get("calibration_method") != calibration_payload.get("method"):
            raise ValueError(
                "safe calibrated artifacts require matching calibration_method metadata"
            )
    elif "calibration_method" in metadata:
        raise ValueError("raw safe artifacts must not declare calibration_method metadata")
    try:
        booster_bytes = bytes(base_classifier.get_booster().save_raw(raw_format="json"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("XGBClassifier must be fitted before safe export") from exc

    manifest = {
        "schema_version": 2 if calibration_payload is not None else 1,
        "format": SAFE_ARTIFACT_FORMAT,
        "metadata": metadata,
        "builder_member": "builder.json",
        "booster_member": "booster.json",
    }
    if calibration_payload is not None:
        manifest["calibrator_member"] = _CALIBRATION_MEMBER
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        archive.writestr("builder.json", json.dumps(builder_payload, sort_keys=True))
        archive.writestr("booster.json", booster_bytes)
        if calibration_payload is not None:
            archive.writestr(
                _CALIBRATION_MEMBER,
                json.dumps(calibration_payload, sort_keys=True),
            )
    return output


def _validate_archive(archive: zipfile.ZipFile, *, schema_version: int) -> None:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    expected_members = _BASE_MEMBERS | (
        {_CALIBRATION_MEMBER} if schema_version == 2 else set()
    )
    if len(names) != len(set(names)) or set(names) != expected_members:
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
) -> tuple[PairFeatureBuilder, XGBClassifier | CalibratedClassifier, dict[str, Any]]:
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
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError("safe artifact manifest is malformed")
        schema_version = manifest.get("schema_version")
        if type(schema_version) is not int or schema_version not in {1, 2}:
            raise ValueError("unsupported safe artifact schema")
        if manifest.get("format") != SAFE_ARTIFACT_FORMAT:
            raise ValueError("unsupported safe artifact format")
        if schema_version == 2 and manifest.get("calibrator_member") != _CALIBRATION_MEMBER:
            raise ValueError("safe calibrated artifact manifest is malformed")
        if schema_version == 1 and "calibrator_member" in manifest:
            raise ValueError("safe artifact manifest is malformed")
        _validate_archive(archive, schema_version=schema_version)
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("safe artifact metadata is malformed")
        builder_payload = json.loads(archive.read("builder.json"))
        booster_bytes = archive.read("booster.json")
        calibration_payload = (
            json.loads(archive.read(_CALIBRATION_MEMBER)) if schema_version == 2 else None
        )
    builder = _load_builder(builder_payload)
    classifier: XGBClassifier | CalibratedClassifier = _load_booster(booster_bytes)
    if schema_version == 2:
        calibrator = ProbabilityCalibrator.from_safe_payload(calibration_payload)
        if metadata.get("calibration_method") != calibrator.method:
            raise ValueError(
                "safe calibration metadata does not match calibration payload"
            )
        classifier = CalibratedClassifier(classifier, calibrator)
    elif "calibration_method" in metadata:
        raise ValueError("raw safe artifact metadata declares calibration_method")
    return builder, classifier, dict(metadata)
