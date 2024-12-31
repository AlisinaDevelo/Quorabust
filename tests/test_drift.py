import numpy as np

from quorabust.drift import feature_means_from_matrix, max_mean_shift, mean_shift_scores


def test_mean_shift_scores():
    ref = {"a": 1.0, "b": 2.0}
    cur = {"a": 1.5, "b": 2.0}
    s = mean_shift_scores(ref, cur)
    assert abs(s["a"] - 0.5) < 1e-6
    assert s["b"] == 0.0


def test_max_mean_shift():
    assert max_mean_shift({"x": 10.0}, {"x": 12.0}) == 0.2


def test_feature_means_from_matrix():
    X = np.array([[0.0, 2.0], [2.0, 4.0]])
    m = feature_means_from_matrix(["f0", "f1"], X)
    assert m["f0"] == 1.0
    assert m["f1"] == 3.0
