from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_CALIBRATION_METHODS = {"sigmoid", "isotonic"}


def _validated_inputs(
    probabilities: Any,
    labels: Any,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(scores) != len(y):
        raise ValueError("probabilities and labels must have the same length")
    if len(scores) < 2:
        raise ValueError("at least two calibration rows are required")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities must be finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be between 0 and 1")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("labels must be binary values 0 or 1")
    if len(np.unique(y)) < 2:
        raise ValueError("calibration data must contain both label classes")
    return scores, y


class ProbabilityCalibrator:
    """Fit a monotonic mapping from model scores to probabilities."""

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in _CALIBRATION_METHODS:
            raise ValueError("method must be one of: sigmoid, isotonic")
        self.method = method
        self._model: Any | None = None

    def fit(self, probabilities: Any, labels: Any) -> ProbabilityCalibrator:
        scores, y = _validated_inputs(probabilities, labels)
        if self.method == "sigmoid":
            model = LogisticRegression(solver="lbfgs")
            model.fit(scores.reshape(-1, 1), y)
        else:
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(scores, y)
        self._model = model
        return self

    def predict(self, probabilities: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError("probabilities must be finite")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("probabilities must be between 0 and 1")
        if self.method == "sigmoid":
            calibrated = self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            calibrated = self._model.predict(scores)
        return np.asarray(calibrated, dtype=np.float64)


class CalibratedClassifier:
    """Delegate feature inference to a classifier and calibrate its probabilities."""

    def __init__(self, base_classifier: Any, calibrator: ProbabilityCalibrator) -> None:
        self.base_classifier = base_classifier
        self.calibrator = calibrator
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, features: Any) -> np.ndarray:
        raw = np.asarray(self.base_classifier.predict_proba(features), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError("base classifier must return binary predict_proba output")
        positive = self.calibrator.predict(raw[:, 1])
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: Any) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


def calibrate_classifier(
    classifier: Any,
    features: Any,
    labels: Any,
    *,
    method: str = "sigmoid",
) -> CalibratedClassifier:
    """Fit a probability calibrator on classifier outputs from independent data."""
    raw = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("classifier must return binary predict_proba output")
    calibrator = ProbabilityCalibrator(method)
    calibrator.fit(raw[:, 1], labels)
    return CalibratedClassifier(classifier, calibrator)
