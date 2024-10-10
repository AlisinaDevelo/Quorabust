import numpy as np
import pandas as pd
import pytest

from quorabust.features import PairFeatureBuilder, word_jaccard


def test_word_jaccard_identical():
    assert word_jaccard("same words here", "same words here") == 1.0


def test_word_jaccard_disjoint():
    assert word_jaccard("a b", "c d") == 0.0


def test_pair_feature_builder_shape_and_monotonic_similarity():
    corpus = [
        "what is python",
        "how to learn python",
        "recipe for cake",
    ]
    b = PairFeatureBuilder(max_features=256)
    b.fit(corpus)
    q1 = ["what is python", "what is python"]
    q2 = ["what is python programming", "recipe for cake"]
    X = b.transform_pairs(q1, q2)
    assert X.shape == (2, 5)
    assert X[0, 0] >= X[1, 0]


def test_fit_from_frame():
    df = pd.DataFrame(
        {
            "question1": ["a b", "c d"],
            "question2": ["a c", "e f"],
        }
    )
    b = PairFeatureBuilder(max_features=64)
    b.fit_from_frame(df)
    X = b.transform_frame(df)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 2


def test_transform_before_fit_raises():
    b = PairFeatureBuilder()
    with pytest.raises(RuntimeError, match="fit"):
        b.transform_pairs(["a"], ["b"])
