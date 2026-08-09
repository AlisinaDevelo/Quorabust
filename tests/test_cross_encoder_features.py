import numpy as np
import pandas as pd
import pytest

import quorabust.cross_encoder_features as cef


class _FakeCrossEncoder:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def predict(self, pairs, batch_size: int = 32, show_progress_bar: bool = False):
        assert batch_size == 32
        return np.asarray([0.9 if a.split()[0:1] == b.split()[0:1] else 0.1 for a, b in pairs])


def test_pair_cross_encoder_builder_requires_sentence_transformers(monkeypatch):
    monkeypatch.setattr(cef, "CrossEncoder", None)
    with pytest.raises(RuntimeError, match="nlp"):
        cef.PairCrossEncoderBuilder()


def test_pair_cross_encoder_builder_shapes(monkeypatch):
    monkeypatch.setattr(cef, "CrossEncoder", _FakeCrossEncoder)
    builder = cef.PairCrossEncoderBuilder(model_name="fake-cross-encoder")
    df = pd.DataFrame(
        {
            "question1": ["hello world", "foo bar"],
            "question2": ["hello there", "baz qux"],
        }
    )

    X = builder.fit_from_frame(df).transform_frame(df)

    assert builder.feature_names() == ["cross_score", "len_ratio", "abs_len_diff", "len_sum"]
    assert X.shape == (2, 4)
    assert X[0, 0] == 0.9
    assert X[1, 0] == 0.1


def test_pair_cross_encoder_builder_records_model_revision(monkeypatch):
    monkeypatch.setattr(cef, "CrossEncoder", _FakeCrossEncoder)
    builder = cef.PairCrossEncoderBuilder(revision="  abc123  ")

    assert builder.model_revision == "abc123"
    assert builder._model.kwargs == {"revision": "abc123"}


def test_pair_cross_encoder_builder_controls_batch_size(monkeypatch):
    monkeypatch.setattr(cef, "CrossEncoder", _FakeCrossEncoder)
    builder = cef.PairCrossEncoderBuilder(batch_size=7)

    assert builder.batch_size == 7


def test_pair_cross_encoder_builder_rejects_invalid_batch_size(monkeypatch):
    monkeypatch.setattr(cef, "CrossEncoder", _FakeCrossEncoder)
    with pytest.raises(ValueError, match="batch_size"):
        cef.PairCrossEncoderBuilder(batch_size=0)


def test_pair_cross_encoder_builder_rejects_blank_model_revision(monkeypatch):
    monkeypatch.setattr(cef, "CrossEncoder", _FakeCrossEncoder)
    with pytest.raises(ValueError, match="revision"):
        cef.PairCrossEncoderBuilder(revision=" ")
