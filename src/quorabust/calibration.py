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
        self._safe_payload: dict[str, Any] | None = None

    def fit(self, probabilities: Any, labels: Any) -> ProbabilityCalibrator:
        scores, y = _validated_inputs(probabilities, labels)
        if self.method == "sigmoid":
            model = LogisticRegression(solver="lbfgs")
            model.fit(scores.reshape(-1, 1), y)
        else:
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(scores, y)
        self._model = model
        self._safe_payload = self._payload_from_fitted_model(model)
        return self

    def predict(self, probabilities: Any) -> np.ndarray:
        safe_payload = getattr(self, "_safe_payload", None)
        if self._model is None and safe_payload is None:
            raise RuntimeError("fit() must be called before predict()")
        scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError("probabilities must be finite")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("probabilities must be between 0 and 1")
        if self.method == "sigmoid":
            if self._model is not None:
                calibrated = self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
            else:
                assert safe_payload is not None
                coefficient = float(safe_payload["coefficient"])
                intercept = float(safe_payload["intercept"])
                logits = np.clip(coefficient * scores + intercept, -709.0, 709.0)
                calibrated = np.where(
                    logits >= 0.0,
                    1.0 / (1.0 + np.exp(-logits)),
                    np.exp(logits) / (1.0 + np.exp(logits)),
                )
        else:
            if self._model is not None:
                calibrated = self._model.predict(scores)
            else:
                assert safe_payload is not None
                calibrated = np.interp(
                    scores,
                    np.asarray(safe_payload["x_thresholds"], dtype=np.float64),
                    np.asarray(safe_payload["y_thresholds"], dtype=np.float64),
                )
        return np.asarray(calibrated, dtype=np.float64)

    def to_safe_payload(self) -> dict[str, Any]:
        """Return JSON-only parameters for pickle-free artifact serialization."""
        safe_payload = getattr(self, "_safe_payload", None)
        if self._model is None and safe_payload is None:
            raise RuntimeError("fit() must be called before serialization")
        if safe_payload is None:
            assert self._model is not None
            self._safe_payload = self._payload_from_fitted_model(self._model)
            safe_payload = self._safe_payload
        assert safe_payload is not None
        return dict(safe_payload)

    @classmethod
    def from_safe_payload(cls, payload: Any) -> ProbabilityCalibrator:
        """Load validated JSON-only parameters without deserializing executable code."""
        if (
            not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
        ):
            raise ValueError("unsupported calibration payload")
        method = payload.get("method")
        if not isinstance(method, str) or method not in _CALIBRATION_METHODS:
            raise ValueError("calibration payload method is unsupported")
        calibrator = cls(method)
        if method == "sigmoid":
            coefficient = payload.get("coefficient")
            intercept = payload.get("intercept")
            if (
                isinstance(coefficient, bool)
                or not isinstance(coefficient, int | float)
                or not np.isfinite(coefficient)
            ):
                raise ValueError("calibration coefficient must be finite")
            if (
                isinstance(intercept, bool)
                or not isinstance(intercept, int | float)
                or not np.isfinite(intercept)
            ):
                raise ValueError("calibration intercept must be finite")
            calibrator._safe_payload = {
                "schema_version": 1,
                "method": method,
                "coefficient": float(coefficient),
                "intercept": float(intercept),
            }
            return calibrator

        try:
            x_thresholds = np.asarray(
                payload.get("x_thresholds"), dtype=np.float64
            ).reshape(-1)
            y_thresholds = np.asarray(
                payload.get("y_thresholds"), dtype=np.float64
            ).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("isotonic calibration thresholds are malformed") from exc
        if len(x_thresholds) == 0 or len(x_thresholds) != len(y_thresholds):
            raise ValueError("isotonic calibration thresholds are malformed")
        if (
            not np.isfinite(x_thresholds).all()
            or not np.isfinite(y_thresholds).all()
            or np.any((x_thresholds < 0.0) | (x_thresholds > 1.0))
            or np.any((y_thresholds < 0.0) | (y_thresholds > 1.0))
            or np.any(np.diff(x_thresholds) < 0.0)
            or np.any(np.diff(y_thresholds) < 0.0)
        ):
            raise ValueError("isotonic calibration thresholds are invalid")
        calibrator._safe_payload = {
            "schema_version": 1,
            "method": method,
            "x_thresholds": x_thresholds.tolist(),
            "y_thresholds": y_thresholds.tolist(),
        }
        return calibrator

    def _payload_from_fitted_model(self, model: Any) -> dict[str, Any]:
        if self.method == "sigmoid":
            coefficient = np.asarray(getattr(model, "coef_", []), dtype=np.float64).reshape(-1)
            intercept = np.asarray(getattr(model, "intercept_", []), dtype=np.float64).reshape(-1)
            if len(coefficient) != 1 or len(intercept) != 1:
                raise ValueError("sigmoid calibration model parameters are malformed")
            if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
                raise ValueError("sigmoid calibration model parameters must be finite")
            return {
                "schema_version": 1,
                "method": self.method,
                "coefficient": float(coefficient[0]),
                "intercept": float(intercept[0]),
            }

        x_thresholds = np.asarray(
            getattr(model, "X_thresholds_", []), dtype=np.float64
        ).reshape(-1)
        y_thresholds = np.asarray(
            getattr(model, "y_thresholds_", []), dtype=np.float64
        ).reshape(-1)
        if len(x_thresholds) == 0 or len(x_thresholds) != len(y_thresholds):
            raise ValueError("isotonic calibration thresholds are malformed")
        return {
            "schema_version": 1,
            "method": self.method,
            "x_thresholds": x_thresholds.tolist(),
            "y_thresholds": y_thresholds.tolist(),
        }


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
