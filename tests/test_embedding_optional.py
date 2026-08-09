import numpy as np
import pytest

import quorabust.embedding_features as ef


def test_pair_embedding_builder_requires_sentence_transformers(monkeypatch):
    monkeypatch.setattr(ef, "SentenceTransformer", None)
    with pytest.raises(RuntimeError, match="nlp"):
        ef.PairEmbeddingBuilder()


def test_pair_embedding_builder_batches_unique_texts_and_reuses_cache(monkeypatch):
    class FakeSentenceTransformer:
        calls: list[list[str]] = []

        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            return np.asarray(
                [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)],
                dtype=np.float32,
            )

    monkeypatch.setattr(ef, "SentenceTransformer", FakeSentenceTransformer)
    builder = ef.PairEmbeddingBuilder(batch_size=2, cache_size=10)

    first = builder.transform_pairs(["same", "same"], ["other", "other"])
    second = builder.transform_pairs(["same"], ["other"])

    assert first.shape == (2, 5)
    assert np.allclose(first[0], second[0])
    assert FakeSentenceTransformer.calls == [["same", "other"]]
    assert builder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
