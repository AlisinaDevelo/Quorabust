import numpy as np
import pytest

from quorabust.calibration import CalibratedClassifier, ProbabilityCalibrator, calibrate_classifier


class _BaseClassifier:
    def predict_proba(self, features):
        scores = np.asarray(features, dtype=float).reshape(-1)
        return np.column_stack((1.0 - scores, scores))


def test_sigmoid_calibrator_returns_binary_probabilities():
    scores = np.asarray([0.05, 0.2, 0.4, 0.7, 0.95])
    labels = np.asarray([0, 0, 0, 1, 1])

    calibrator = ProbabilityCalibrator("sigmoid").fit(scores, labels)

    calibrated = calibrator.predict(scores)
    assert calibrated.shape == scores.shape
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert np.all(np.diff(calibrated) >= 0.0)


def test_isotonic_calibrator_clips_out_of_range_scores():
    calibrator = ProbabilityCalibrator("isotonic").fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])

    calibrated = calibrator.predict([0.0, 0.5, 1.0])
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert np.all(np.diff(calibrated) >= 0.0)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_safe_calibration_payload_round_trip_preserves_predictions(method):
    scores = np.asarray([0.05, 0.2, 0.4, 0.7, 0.95])
    labels = np.asarray([0, 0, 0, 1, 1])
    calibrator = ProbabilityCalibrator(method).fit(scores, labels)

    restored = ProbabilityCalibrator.from_safe_payload(calibrator.to_safe_payload())

    assert np.allclose(restored.predict(scores), calibrator.predict(scores), atol=1e-12)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1, "method": "temperature"},
        {
            "schema_version": 1,
            "method": "sigmoid",
            "coefficient": "bad",
            "intercept": 0.0,
        },
        {
            "schema_version": 1,
            "method": "isotonic",
            "x_thresholds": [0.2, 0.1],
            "y_thresholds": [0.0, 1.0],
        },
    ],
)
def test_safe_calibration_payload_rejects_malformed_values(payload):
    with pytest.raises(ValueError, match="calibration|isotonic"):
        ProbabilityCalibrator.from_safe_payload(payload)


def test_calibrated_classifier_delegates_and_preserves_binary_shape():
    base = _BaseClassifier()
    wrapper = calibrate_classifier(
        base,
        features=[0.1, 0.3, 0.8, 0.9],
        labels=[0, 0, 1, 1],
    )

    assert isinstance(wrapper, CalibratedClassifier)
    probabilities = wrapper.predict_proba([0.2, 0.7])
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert wrapper.predict([0.2, 0.7]).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        ([0.1], [1], "at least two"),
        ([0.1, 0.2], [0], "same length"),
        ([0.1, 1.2], [0, 1], "between 0 and 1"),
        ([0.1, float("nan")], [0, 1], "finite"),
        ([0.1, 0.2], [1, 1], "both label classes"),
    ],
)
def test_calibrator_rejects_invalid_inputs(probabilities, labels, message):
    with pytest.raises(ValueError, match=message):
        ProbabilityCalibrator().fit(probabilities, labels)


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError, match="sigmoid, isotonic"):
        ProbabilityCalibrator("temperature")
