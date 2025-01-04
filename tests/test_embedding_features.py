import pandas as pd
import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")

from quorabust.embedding_features import PairEmbeddingBuilder  # noqa: E402
from quorabust.model import train_duplicate_classifier  # noqa: E402


def test_pair_embedding_builder_shapes():
    b = PairEmbeddingBuilder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    df = pd.DataFrame(
        {
            "question1": ["hello world", "foo bar"],
            "question2": ["hello there", "baz qux"],
        }
    )
    b.fit_from_frame(df)
    X = b.transform_frame(df)
    assert X.shape == (2, 5)


def test_train_with_embedding_backend():
    df = pd.DataFrame(
        {
            "question1": [f"text a {i}" for i in range(16)],
            "question2": [f"text b {i % 4}" for i in range(16)],
            "is_duplicate": [i % 2 for i in range(16)],
        }
    )
    builder = PairEmbeddingBuilder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    b, clf = train_duplicate_classifier(
        df,
        feature_builder=builder,
        xgb_params={"n_estimators": 20, "max_depth": 3},
    )
    proba = clf.predict_proba(b.transform_pairs(["a"], ["b"]))
    assert proba.shape == (1, 2)
